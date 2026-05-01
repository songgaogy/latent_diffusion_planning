"""LIBERO latent data module.

Reads:
  * raw lowdim/rgb obs from the per-suite LIBERO hdf5 files (via the same
    multi-file glob as ``data.libero_data.LiberoDataset``); and
  * pre-encoded SD-VAE latents from a single ``latent.hdf5`` produced by
    ``process_sdvae_data.py``'s ``run_libero`` branch, keyed by the same
    globally unique demo ids.

Latent rgb keys are prefixed with ``latent_`` (matching the rm_lift / aloha
convention); on read we strip the prefix to look up the latent group:
``latent.hdf5/data/<demo_id>/latent/<rgb_key>``.
"""

from __future__ import annotations

import glob as _glob
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from data.libero_data import LiberoDataset, _resolve_paths


class LiberoLatentDataset(LiberoDataset):
    """LiberoDataset variant that overlays a single latent.hdf5.

    For each obs key:
      * if the key is ``optimal`` -> synthesized.
      * if the key starts with ``latent_`` -> read from the latent.hdf5 at
        ``data/<demo_id>/latent/<stripped>`` (no slicing along time, then
        windowed/padded same as LiberoDataset).
      * else -> read from the raw hdf5 like the base class.
    """

    def __init__(self, *, latent_path, goal_keys=None, cache_all_in_ram=False, **kwargs):
        super().__init__(**kwargs)
        self.latent_path = os.path.expanduser(latent_path)
        self.goal_keys = tuple(goal_keys or ())
        self._latent_handle = None
        self.cache_all_in_ram = bool(cache_all_in_ram)
        self._ram_demo_arrays = None
        self._validate_latent_demo_ids()
        if self.cache_all_in_ram:
            self._preload_all_demo_arrays()

    @property
    def latent_file(self):
        if self._latent_handle is None:
            self._latent_handle = h5py.File(
                self.latent_path, "r", swmr=self.hdf5_use_swmr, libver="latest"
            )
        return self._latent_handle

    def close_handles(self):
        super().close_handles()
        if self._latent_handle is not None:
            try:
                self._latent_handle.close()
            except Exception:
                pass
            self._latent_handle = None

    def _validate_latent_demo_ids(self):
        """Cross-check that every (file_stem__demo_key) tuple from the raw
        registry has a matching group in the latent hdf5. Catches stale latent
        files early instead of failing at the first sample.
        """
        with h5py.File(self.latent_path, "r") as f:
            present = set(f["data"].keys())
        missing = [d for d in self.demo_ids if d not in present]
        if missing:
            raise RuntimeError(
                f"latent.hdf5 at {self.latent_path} is missing {len(missing)} "
                f"demo groups (e.g. {missing[:3]}); re-run process_sdvae_data.py."
            )

    def _read_demo_slice(self, demo_grp, key, seq_start, seq_end):
        # demo_grp is from the *raw* file. We need the demo_id to look up the
        # latent file. Walk back up — h5py groups expose `name` like
        # "/data/demo_3"; combine with the file stem of demo_grp.file.
        if not key.startswith("latent_"):
            return super()._read_demo_slice(demo_grp, key, seq_start, seq_end)
        # build globally unique demo id matching iter_demo_obs convention
        raw_path = demo_grp.file.filename
        demo_key = demo_grp.name.rsplit("/", 1)[-1]
        demo_id = f"{Path(raw_path).stem}__{demo_key}"
        latent_key = key[len("latent_"):]
        return self.latent_file["data"][demo_id]["latent"][latent_key][seq_start:seq_end]

    def _read_demo_arrays_from_groups(self, demo_grp, latent_grp):
        arrays = {"data": {}, "obs": {}}
        for key in self.dataset_keys:
            arr = demo_grp[key][:]
            arrays["data"][key] = arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr
        for key in self.obs_keys:
            if key == "optimal":
                continue
            if key.startswith("latent_"):
                latent_key = key[len("latent_"):]
                arr = latent_grp["latent"][latent_key][:]
            else:
                arr = demo_grp["obs"][key][:]
            arrays["obs"][key] = arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr
        return arrays

    def _read_demo_arrays(self, demo_idx):
        path, dk, _L, _global_start = self.demos[demo_idx]
        demo_id = self.demo_ids[demo_idx]
        f = self._get_file(path)
        demo_grp = f["data"][dk]
        latent_grp = self.latent_file["data"][demo_id]
        return self._read_demo_arrays_from_groups(demo_grp, latent_grp)

    def _preload_all_demo_arrays(self):
        ram_arrays = []
        total_bytes = 0
        with h5py.File(self.latent_path, "r", swmr=self.hdf5_use_swmr, libver="latest") as latent_f:
            raw_handles = {}
            try:
                for demo_idx, (path, dk, _L, _global_start) in enumerate(self.demos):
                    if path not in raw_handles:
                        raw_handles[path] = h5py.File(
                            path, "r", swmr=self.hdf5_use_swmr, libver="latest"
                        )
                    demo_grp = raw_handles[path]["data"][dk]
                    latent_grp = latent_f["data"][self.demo_ids[demo_idx]]
                    arrays = self._read_demo_arrays_from_groups(demo_grp, latent_grp)
                    total_bytes += sum(arr.nbytes for group in arrays.values() for arr in group.values())
                    ram_arrays.append(arrays)
            finally:
                for handle in raw_handles.values():
                    handle.close()
        self._ram_demo_arrays = ram_arrays
        print(
            f"[LiberoLatentDataset] cached {len(ram_arrays)} demos in RAM "
            f"({total_bytes / (1024 ** 2):.1f} MiB)"
        )

    def _get_demo_arrays(self, demo_idx):
        if self._ram_demo_arrays is not None:
            return self._ram_demo_arrays[demo_idx]
        return super()._get_demo_arrays(demo_idx)

    def _read_goal_obs(self, demo_idx):
        if not self.goal_keys:
            return None
        arrays = self._ram_demo_arrays[demo_idx] if self._ram_demo_arrays is not None else None
        demo_id = self.demo_ids[demo_idx]
        goal_obs = {}
        for key in self.goal_keys:
            if arrays is not None and key in arrays["obs"]:
                arr = arrays["obs"][key][-1]
            elif key.startswith("latent_"):
                latent_key = key[len("latent_"):]
                arr = self.latent_file["data"][demo_id]["latent"][latent_key][-1]
            else:
                path, dk, _L, _global_start = self.demos[demo_idx]
                arr = self._get_file(path)["data"][dk]["obs"][key][-1]
            goal_obs[key] = arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr
        return goal_obs

    def _get_batch(self, demo_idx, local_index):
        batch = super()._get_batch(demo_idx, local_index)
        goal_obs = self._read_goal_obs(demo_idx)
        if goal_obs is not None:
            batch["goal_obs"] = goal_obs
        return batch


class LiberoLatentData:
    """Hydra wrapper. Mirrors LiberoData API, swapping in LiberoLatentDataset."""

    def __init__(
        self,
        name,
        train_paths,
        eval_paths,
        train_latent_path,
        eval_latent_path,
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
        cache_all_in_ram=False,
    ):
        self.name = name
        self.train_paths = _resolve_paths(train_paths)
        self.eval_paths = _resolve_paths(eval_paths)
        if not self.train_paths:
            raise ValueError(f"LiberoLatentData: train_paths resolved to empty (spec={train_paths})")
        self.train_latent_path = train_latent_path
        self.eval_latent_path = eval_latent_path
        self.train_n_episode_overfit = train_n_episode_overfit
        self.eval_n_episode_overfit = eval_n_episode_overfit
        self.train_per_file_n_overfit = train_per_file_n_overfit
        self.eval_per_file_n_overfit = eval_per_file_n_overfit
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        self.obs_horizon = obs_horizon
        self.cache_all_in_ram = cache_all_in_ram

        self.meta = meta
        self.env_params = OmegaConf.to_container(env_params, resolve=True)
        self.shape_meta = meta.shape_meta
        self._train_dataset = None
        self._val_dataset = None

        goal_keys = list(getattr(meta, "goal_rgb_obs", []))
        obs_keys = list(meta.lowdim_obs) + list(meta.rgb_obs)
        self.ds_kwargs = dict(
            obs_keys=obs_keys,
            dataset_keys=["actions"],
            frame_stack=self.obs_horizon,
            seq_length=seq_length,
            hdf5_use_swmr=hdf5_use_swmr,
            rgb_keys=list(meta.rgb_obs),
            goal_keys=goal_keys,
            optimal=1,
        )

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = LiberoLatentDataset(
                hdf5_paths=self.train_paths,
                latent_path=self.train_latent_path,
                cache_all_in_ram=self.cache_all_in_ram,
                n_overfit=self.train_n_episode_overfit,
                per_file_n_overfit=self.train_per_file_n_overfit,
                **self.ds_kwargs,
            )
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            paths = self.eval_paths if self.eval_paths else self.train_paths
            latent = self.eval_latent_path if self.eval_latent_path else self.train_latent_path
            self._val_dataset = LiberoLatentDataset(
                hdf5_paths=paths,
                latent_path=latent,
                cache_all_in_ram=self.cache_all_in_ram,
                n_overfit=self.eval_n_episode_overfit,
                per_file_n_overfit=self.eval_per_file_n_overfit,
                **self.ds_kwargs,
            )
        return self._val_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=False,
            shuffle=False,
            persistent_workers=self.n_workers > 0,
            prefetch_factor=self.prefetch_factor if self.n_workers > 0 else None,
        )

    def eval_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=False,
            shuffle=False,
            persistent_workers=self.n_workers > 0,
            prefetch_factor=self.prefetch_factor if self.n_workers > 0 else None,
        )
