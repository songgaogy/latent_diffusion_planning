"""LIBERO data module for the LDP pipeline.

Mirrors data/alohasim_data.py (no `next_obs`) but:
  * loads from many hdf5 files (a glob over LIBERO suite directories);
  * uses lazy hdf5 reads on every sample to avoid welding ~100k+ frames of
    256x256x3 imagery into RAM;
  * builds globally unique demo ids of the form ``"{file_stem}__{demo_key}"``
    so latent.hdf5 groups never collide across the 130 task files;
  * tolerates missing ``num_samples`` attrs (LIBERO hdf5s leave demo attrs
    empty) by falling back to ``actions.shape[0]``.
"""

import glob as _glob
import os
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data._utils.collate import default_collate
from omegaconf import OmegaConf


def _resolve_paths(spec):
    """Accept a glob string, a list of globs/paths, or null/empty -> [].

    OmegaConf passes lists as ListConfig; coerce + resolve.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        items = [spec]
    else:
        items = list(spec)
    paths = []
    repo_root = Path(__file__).resolve().parents[1]
    for it in items:
        it = os.path.expanduser(str(it))
        candidates = [it]
        if not os.path.isabs(it):
            candidates.append(str(repo_root / it))
        for candidate in candidates:
            if any(c in candidate for c in "*?[]"):
                matches = sorted(_glob.glob(candidate))
                if matches:
                    paths.extend(matches)
                    break
            elif os.path.exists(candidate):
                paths.append(candidate)
                break
        else:
            paths.append(it)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _collate_numpy_tree(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: _collate_numpy_tree([item[key] for item in batch]) for key in elem}
    if isinstance(elem, np.ndarray):
        arr = np.ascontiguousarray(np.stack(batch, axis=0))
        return torch.from_numpy(arr)
    return default_collate(batch)


class LiberoDataset(torch.utils.data.IterableDataset):
    """Iterable dataset over a list of LIBERO hdf5 files.

    Frame-stacking / seq-length / padding logic matches ALOHASimDataset.
    Differences:
      * registry of (file_path, demo_key, length, global_start_idx) instead of
        a single welded numpy array;
      * one h5py handle per worker per file, opened lazily, cached in a dict
        scoped to this dataset instance.
    """

    def __init__(
        self,
        hdf5_paths,
        obs_keys,
        rgb_keys,
        dataset_keys,
        frame_stack,
        seq_length,
        hdf5_use_swmr,
        n_overfit,
        optimal,
        per_file_n_overfit=None,
        cache_max_gb=0,
        sample_burst_length=1,
        rgb_shapes=None,
    ):
        super().__init__()

        self.hdf5_paths = list(hdf5_paths)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.optimal = optimal
        self.obs_keys = tuple(obs_keys)
        self.rgb_keys = tuple(rgb_keys)
        self.dataset_keys = tuple(dataset_keys)
        self.rgb_shapes = {k: tuple(v) for k, v in (rgb_shapes or {}).items()}
        self.n_overfit = n_overfit  # truncate global demo list
        self.per_file_n_overfit = per_file_n_overfit  # truncate within each file
        self.cache_max_bytes = int(float(cache_max_gb) * (1024 ** 3))
        self.sample_burst_length = max(1, int(sample_burst_length))
        self.n_frame_stack = int(frame_stack)
        self.seq_length = int(seq_length)
        assert self.n_frame_stack >= 1 and self.seq_length >= 1

        # per-process h5py handle cache (populated lazily; do NOT touch in __init__
        # so workers can fork cleanly — opening hdf5 in __init__ then forking is
        # the classic h5py crash).
        self._file_handles = {}
        self._demo_cache = OrderedDict()
        self._demo_cache_bytes = 0
        self._rng = None

        self._build_registry()

    def _build_registry(self):
        """Walk every hdf5 file once (read-only, no caching) to learn lengths.

        We open each file briefly and close it; per-worker handles are opened
        lazily later.
        """
        self.demos = []  # list of (file_path, demo_key, length, global_start_idx)
        self.total_n_sequences = 0
        for path in self.hdf5_paths:
            with h5py.File(path, "r", swmr=self.hdf5_use_swmr, libver="latest") as f:
                demo_keys = list(f["data"].keys())
                # sort numerically by the trailing index, like alohasim_data does
                demo_keys.sort(key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
                if self.per_file_n_overfit is not None:
                    demo_keys = demo_keys[: self.per_file_n_overfit]
                for dk in demo_keys:
                    grp = f["data"][dk]
                    L = grp.attrs.get("num_samples", None)
                    if L is None:
                        L = grp["actions"].shape[0]
                    L = int(L)
                    if L < 1:
                        continue
                    self.demos.append((path, dk, L, self.total_n_sequences))
                    self.total_n_sequences += L
        if self.n_overfit is not None:
            assert self.n_overfit <= len(self.demos), (
                f"n_overfit={self.n_overfit} but only {len(self.demos)} demos available"
            )
            self.demos = self.demos[: self.n_overfit]
            self.total_n_sequences = sum(L for _, _, L, _ in self.demos)
            # rebuild start indices after truncation
            running = 0
            new_demos = []
            for path, dk, L, _ in self.demos:
                new_demos.append((path, dk, L, running))
                running += L
            self.demos = new_demos
        self.n_demos = len(self.demos)
        # bookkeeping name shared with alohasim API
        self.demo_ids = [self._make_demo_id(p, k) for p, k, _, _ in self.demos]
        # quick reverse lookup: global frame index -> demo registry index
        # (built lazily; we use binary search rather than expanding a flat list
        # to keep memory tiny for VAE training over 5500 demos)
        self._cum_starts = np.array([d[3] for d in self.demos] + [self.total_n_sequences], dtype=np.int64)

        print(
            f"[LiberoDataset] {len(self.hdf5_paths)} hdf5 files, "
            f"{self.n_demos} demos, {self.total_n_sequences} total frames"
        )

    @staticmethod
    def _make_demo_id(path, demo_key):
        return f"{Path(path).stem}__{demo_key}"

    def _index_to_demo(self, index):
        # rightmost insertion point - 1 = demo idx
        i = int(np.searchsorted(self._cum_starts, index, side="right")) - 1
        i = max(0, min(i, self.n_demos - 1))
        return i

    def _get_file(self, path):
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(
                path, "r", swmr=self.hdf5_use_swmr, libver="latest"
            )
        return self._file_handles[path]

    def close_handles(self):
        for h in list(self._file_handles.values()):
            try:
                h.close()
            except Exception:
                pass
        self._file_handles.clear()
        self._demo_cache.clear()
        self._demo_cache_bytes = 0

    def __del__(self):
        try:
            self.close_handles()
        except Exception:
            pass

    @staticmethod
    def _pad_seq(arr, n_pad_start, n_pad_end):
        if n_pad_start > 0:
            arr = np.concatenate([np.expand_dims(arr[0], 0)] * n_pad_start + [arr], axis=0)
        if n_pad_end > 0:
            arr = np.concatenate([arr] + [np.expand_dims(arr[-1], 0)] * n_pad_end, axis=0)
        return arr

    def _read_demo_slice(self, demo_grp, key, seq_start, seq_end):
        # h5py supports fancy slicing; this loads only the needed window
        arr = demo_grp["obs"][key][seq_start:seq_end]
        return self._resize_rgb_if_needed(key, arr)

    def _resize_rgb_if_needed(self, key, arr):
        target_shape = self.rgb_shapes.get(key)
        if target_shape is None or tuple(arr.shape[1:]) == target_shape:
            return arr
        if key not in self.rgb_keys or len(target_shape) != 3:
            return arr
        target_h, target_w, target_c = target_shape
        if arr.shape[-1] != target_c:
            raise ValueError(f"RGB channel mismatch for {key}: got {arr.shape}, target={target_shape}")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        frames = [np.asarray(Image.fromarray(frame).resize((target_w, target_h), resample=resample)) for frame in arr]
        return np.stack(frames, axis=0).astype(arr.dtype, copy=False)

    def _rng_state(self):
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            seed = worker.seed if worker is not None else None
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _read_demo_arrays(self, demo_idx):
        path, dk, _L, _global_start = self.demos[demo_idx]
        f = self._get_file(path)
        demo_grp = f["data"][dk]
        arrays = {"data": {}, "obs": {}}
        for key in self.dataset_keys:
            arr = demo_grp[key][:]
            arrays["data"][key] = arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr
        for key in self.obs_keys:
            if key != "optimal":
                arr = demo_grp["obs"][key][:]
                arr = self._resize_rgb_if_needed(key, arr)
                arrays["obs"][key] = arr.astype(np.float32, copy=False) if arr.dtype != np.uint8 else arr
        return arrays

    def _get_demo_arrays(self, demo_idx):
        if self.cache_max_bytes <= 0:
            return None
        if demo_idx in self._demo_cache:
            arrays, nbytes = self._demo_cache.pop(demo_idx)
            self._demo_cache[demo_idx] = (arrays, nbytes)
            return arrays

        arrays = self._read_demo_arrays(demo_idx)
        nbytes = sum(arr.nbytes for group in arrays.values() for arr in group.values())
        if nbytes > self.cache_max_bytes:
            return arrays

        while self._demo_cache and self._demo_cache_bytes + nbytes > self.cache_max_bytes:
            _old_idx, (_old_arrays, old_nbytes) = self._demo_cache.popitem(last=False)
            self._demo_cache_bytes -= old_nbytes
        self._demo_cache[demo_idx] = (arrays, nbytes)
        self._demo_cache_bytes += nbytes
        return arrays

    def _get_batch(self, demo_idx, local_index):
        path, dk, L, _global_start = self.demos[demo_idx]
        arrays = self._get_demo_arrays(demo_idx)
        if arrays is None:
            f = self._get_file(path)
            demo_grp = f["data"][dk]
        else:
            demo_grp = None

        seq_start = max(local_index - self.n_frame_stack + 1, 0)
        seq_end = min(local_index + self.seq_length, L)
        n_pad_start = max(self.n_frame_stack - (local_index - seq_start + 1), 0)
        n_pad_end = max(self.seq_length - (seq_end - local_index), 0)

        batch = dict()
        for key in self.dataset_keys:
            seq = arrays["data"][key][seq_start:seq_end] if arrays is not None else demo_grp[key][seq_start:seq_end]
            seq = self._pad_seq(seq, n_pad_start, n_pad_end)
            seq = seq[self.n_frame_stack - 1 :]  # actions get trimmed to seq_length
            batch[key] = seq.astype(np.float32, copy=False) if seq.dtype != np.float32 else seq

        batch["obs"] = dict()
        for key in self.obs_keys:
            if key == "optimal":
                seq = self.optimal * np.ones((self.n_frame_stack + self.seq_length - 1, 1), dtype=np.float32)
            else:
                seq = arrays["obs"][key][seq_start:seq_end] if arrays is not None else self._read_demo_slice(demo_grp, key, seq_start, seq_end)
                seq = self._pad_seq(seq, n_pad_start, n_pad_end)
                # keep image dtype uint8 (saves bandwidth); cast lowdim to float32
                if seq.dtype != np.uint8:
                    seq = seq.astype(np.float32, copy=False)
            batch["obs"][key] = seq

        return batch

    def _sample(self):
        index = self._rng_state().integers(self.total_n_sequences)
        demo_idx = self._index_to_demo(index)
        local = index - self.demos[demo_idx][3]
        return self._get_batch(demo_idx, local)

    def get_item(self, index):
        demo_idx = self._index_to_demo(index)
        local = index - self.demos[demo_idx][3]
        return self._get_batch(demo_idx, local)

    def sample_traj(self, ep_id):
        path, dk, L, _ = self.demos[ep_id]
        f = self._get_file(path)
        demo_grp = f["data"][dk]
        batch = dict()
        for key in self.dataset_keys:
            batch[key] = demo_grp[key][:].astype(np.float32, copy=False)
        batch["obs"] = dict()
        for key in self.obs_keys:
            if key == "optimal":
                arr = self.optimal * np.ones((L, 1), dtype=np.float32)
            else:
                arr = demo_grp["obs"][key][:]
                arr = self._resize_rgb_if_needed(key, arr)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32, copy=False)
            batch["obs"][key] = np.expand_dims(arr, axis=1)
        return batch

    def iter_demo_obs(self, rgb_keys):
        """Yield (demo_id, {rgb_key: full-trajectory ndarray}) tuples.

        Used by process_sdvae_data.py to encode raw images into latents
        without poking ``hdf5_file`` directly (which assumed a single file).
        """
        for demo_idx, (path, dk, L, _) in enumerate(self.demos):
            f = self._get_file(path)
            demo_grp = f["data"][dk]
            out = {k: demo_grp["obs"][k][:] for k in rgb_keys}
            yield self.demo_ids[demo_idx], out

    def __iter__(self):
        while True:
            index = self._rng_state().integers(self.total_n_sequences)
            demo_idx = self._index_to_demo(index)
            L = self.demos[demo_idx][2]
            for _ in range(self.sample_burst_length):
                local = self._rng_state().integers(L)
                yield self._get_batch(demo_idx, local)


class LiberoData:
    """Hydra-instantiated wrapper. Mirrors ALOHASimData / RobomimicData API.

    Accepts either a single path or a glob/list of globs via train_paths /
    eval_paths. Set name = "libero_..." so train_bc.py:170 dispatches to the
    libero eval branch.
    """

    def __init__(
        self,
        name,
        train_paths,
        eval_paths,
        train_n_episode_overfit,
        eval_n_episode_overfit,
        batch_size,
        n_workers,
        prefetch_factor,
        obs_horizon,
        seq_length,
        hdf5_use_swmr,
        meta,
        env_params,
        train_per_file_n_overfit=None,
        eval_per_file_n_overfit=None,
        cache_max_gb_per_worker=0,
        eval_cache_max_gb_per_worker=0,
        sample_burst_length=1,
        pin_memory=False,
    ):
        self.name = name
        self.train_paths = _resolve_paths(train_paths)
        self.eval_paths = _resolve_paths(eval_paths)
        if not self.train_paths:
            raise ValueError(f"LiberoData: train_paths resolved to empty (spec={train_paths})")
        self.train_n_episode_overfit = train_n_episode_overfit
        self.eval_n_episode_overfit = eval_n_episode_overfit
        self.train_per_file_n_overfit = train_per_file_n_overfit
        self.eval_per_file_n_overfit = eval_per_file_n_overfit
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        self.obs_horizon = obs_horizon
        self.pin_memory = pin_memory
        self.cache_max_gb_per_worker = cache_max_gb_per_worker
        self.eval_cache_max_gb_per_worker = eval_cache_max_gb_per_worker
        self.sample_burst_length = sample_burst_length

        self.env_params = OmegaConf.to_container(env_params, resolve=True)
        self.meta = meta
        self.shape_meta = meta.shape_meta
        self._train_dataset = None
        self._val_dataset = None

        obs_keys = list(meta.lowdim_obs) + list(meta.rgb_obs)
        self.ds_kwargs = dict(
            obs_keys=obs_keys,
            dataset_keys=["actions"],
            frame_stack=self.obs_horizon,
            seq_length=seq_length,
            hdf5_use_swmr=hdf5_use_swmr,
            rgb_keys=list(meta.rgb_obs),
            rgb_shapes={k: tuple(meta.shape_meta.all_shapes[k]) for k in meta.rgb_obs},
            optimal=1,
        )

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = LiberoDataset(
                hdf5_paths=self.train_paths,
                n_overfit=self.train_n_episode_overfit,
                per_file_n_overfit=self.train_per_file_n_overfit,
                cache_max_gb=self.cache_max_gb_per_worker,
                sample_burst_length=self.sample_burst_length,
                **self.ds_kwargs,
            )
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            paths = self.eval_paths if self.eval_paths else self.train_paths
            self._val_dataset = LiberoDataset(
                hdf5_paths=paths,
                n_overfit=self.eval_n_episode_overfit,
                per_file_n_overfit=self.eval_per_file_n_overfit,
                cache_max_gb=self.eval_cache_max_gb_per_worker,
                sample_burst_length=1,
                **self.ds_kwargs,
            )
        return self._val_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.n_workers > 0,
            prefetch_factor=self.prefetch_factor if self.n_workers > 0 else None,
            collate_fn=_collate_numpy_tree,
        )

    def eval_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.n_workers > 0,
            prefetch_factor=self.prefetch_factor if self.n_workers > 0 else None,
            collate_fn=_collate_numpy_tree,
        )
