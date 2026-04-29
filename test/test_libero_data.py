"""Smoke tests for data/libero_data.py.

Run:
  /home/dodo/miniconda3/envs/ldp/bin/python -m pytest test/test_libero_data.py -q

These tests only inspect a single libero_10 hdf5 and use very small batches —
they're meant to catch the multi-file indexing / missing-num_samples / shape
class of bugs early, not to validate training correctness.
"""

import os
import sys
from pathlib import Path

import numpy as np

try:
    import pytest  # type: ignore
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.libero_data import LiberoData, LiberoDataset, _resolve_paths  # noqa: E402

LIBERO_10 = REPO_ROOT / "data" / "libero" / "libero_10"
GLOB_LIBERO_10 = str(LIBERO_10 / "*demo.hdf5")


if HAS_PYTEST:
    pytestmark = pytest.mark.skipif(
        not any(LIBERO_10.glob("*demo.hdf5")),
        reason="LIBERO_10 dataset not present at data/libero/libero_10",
    )


def test_resolve_paths_glob():
    paths = _resolve_paths(GLOB_LIBERO_10)
    assert len(paths) == 10, f"expected 10 libero_10 task files, got {len(paths)}"
    assert all(p.endswith(".hdf5") for p in paths)


def test_dataset_registry_unique_demo_ids():
    paths = _resolve_paths(GLOB_LIBERO_10)
    ds = LiberoDataset(
        hdf5_paths=paths[:2],
        obs_keys=["ee_states", "gripper_states", "agentview_rgb"],
        rgb_keys=["agentview_rgb"],
        dataset_keys=["actions"],
        frame_stack=1,
        seq_length=4,
        hdf5_use_swmr=True,
        n_overfit=None,
        optimal=1,
        per_file_n_overfit=2,
    )
    assert ds.n_demos == 4, ds.n_demos
    # globally unique
    assert len(set(ds.demo_ids)) == ds.n_demos
    # length cumulative starts strictly increasing
    starts = [d[3] for d in ds.demos]
    assert all(b > a for a, b in zip(starts, starts[1:]))


def test_sample_shapes():
    paths = _resolve_paths(GLOB_LIBERO_10)[:1]
    seq_length = 9
    obs_horizon = 2
    ds = LiberoDataset(
        hdf5_paths=paths,
        obs_keys=["ee_states", "gripper_states", "agentview_rgb", "eye_in_hand_rgb"],
        rgb_keys=["agentview_rgb", "eye_in_hand_rgb"],
        dataset_keys=["actions"],
        frame_stack=obs_horizon,
        seq_length=seq_length,
        hdf5_use_swmr=True,
        n_overfit=None,
        optimal=1,
        per_file_n_overfit=2,
    )
    expected_obs_T = obs_horizon + seq_length - 1
    s = ds.get_item(0)
    assert s["actions"].shape == (seq_length, 7), s["actions"].shape
    assert s["obs"]["ee_states"].shape == (expected_obs_T, 6)
    assert s["obs"]["gripper_states"].shape == (expected_obs_T, 2)
    assert s["obs"]["agentview_rgb"].shape == (expected_obs_T, 256, 256, 3)
    assert s["obs"]["agentview_rgb"].dtype == np.uint8
    assert s["obs"]["eye_in_hand_rgb"].shape == (expected_obs_T, 256, 256, 3)


def test_iter_demo_obs():
    paths = _resolve_paths(GLOB_LIBERO_10)[:1]
    ds = LiberoDataset(
        hdf5_paths=paths,
        obs_keys=["agentview_rgb"],
        rgb_keys=["agentview_rgb"],
        dataset_keys=["actions"],
        frame_stack=1,
        seq_length=1,
        hdf5_use_swmr=True,
        n_overfit=None,
        optimal=1,
        per_file_n_overfit=1,
    )
    seen = []
    for demo_id, obs in ds.iter_demo_obs(["agentview_rgb"]):
        seen.append((demo_id, obs["agentview_rgb"].shape))
    assert len(seen) == 1
    demo_id, shape = seen[0]
    assert "__demo_" in demo_id, demo_id
    assert shape[1:] == (256, 256, 3) and shape[0] > 0


def test_libero_data_wrapper_dataloader():
    from omegaconf import OmegaConf

    meta = OmegaConf.create(
        dict(
            lowdim_obs=["ee_states", "gripper_states"],
            rgb_obs=["agentview_rgb"],
            rgb_viz="agentview_rgb",
            shape_meta=dict(
                ac_dim=7,
                all_shapes=dict(
                    ee_states=[6],
                    gripper_states=[2],
                    agentview_rgb=[256, 256, 3],
                    optimal=[1],
                ),
                use_images=True,
            ),
            obs_normalization=dict(
                obs=dict(
                    ee_states=dict(min=[0] * 6, max=[1] * 6),
                    gripper_states=dict(min=[0] * 2, max=[1] * 2),
                    agentview_rgb=dict(min=0, max=255),
                    optimal=dict(min=0, max=1),
                ),
                actions=dict(clip_min=-1, clip_max=1),
            ),
        )
    )
    env_params = OmegaConf.create({})
    data = LiberoData(
        name="libero_long_test",
        train_paths=GLOB_LIBERO_10,
        eval_paths=None,
        train_n_episode_overfit=None,
        eval_n_episode_overfit=None,
        train_per_file_n_overfit=1,
        eval_per_file_n_overfit=1,
        batch_size=2,
        n_workers=0,
        prefetch_factor=2,
        obs_horizon=2,
        seq_length=9,
        hdf5_use_swmr=True,
        meta=meta,
        env_params=env_params,
    )
    loader = data.train_dataloader()
    it = iter(loader)
    batch = next(it)
    assert batch["actions"].shape == (2, 9, 7)
    assert batch["obs"]["agentview_rgb"].shape == (2, 10, 256, 256, 3)
    assert data.name.startswith("libero")


if __name__ == "__main__":
    fns = [
        test_resolve_paths_glob,
        test_dataset_registry_unique_demo_ids,
        test_sample_shapes,
        test_iter_demo_obs,
        test_libero_data_wrapper_dataloader,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK  {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f" FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(0 if failed == 0 else 1)
