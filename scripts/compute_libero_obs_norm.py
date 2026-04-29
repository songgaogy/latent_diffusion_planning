"""One-shot script: scan LIBERO hdf5s and print per-key min/max for obs_normalization.

Usage:
  python scripts/compute_libero_obs_norm.py \
      --glob 'data/libero/*/*demo.hdf5' \
      --keys ee_pos ee_ori ee_states gripper_states joint_states actions

Prints a YAML block ready to paste into the data config's
``meta.obs_normalization``.
"""
import argparse
import glob as _glob
import os

import h5py
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True, help="glob pattern over hdf5 files")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states"],
        help="obs keys to scan (lowdim only)",
    )
    parser.add_argument(
        "--include-actions", action="store_true", help="also compute action stats"
    )
    args = parser.parse_args()

    paths = sorted(_glob.glob(os.path.expanduser(args.glob)))
    if not paths:
        raise SystemExit(f"no files matched {args.glob}")
    print(f"# scanning {len(paths)} files")

    mins = {k: None for k in args.keys}
    maxs = {k: None for k in args.keys}
    if args.include_actions:
        mins["__actions"] = None
        maxs["__actions"] = None

    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            for dk in f["data"].keys():
                grp = f["data"][dk]
                for k in args.keys:
                    if k not in grp["obs"]:
                        continue
                    arr = grp["obs"][k][:]
                    lo = arr.min(axis=0)
                    hi = arr.max(axis=0)
                    mins[k] = lo if mins[k] is None else np.minimum(mins[k], lo)
                    maxs[k] = hi if maxs[k] is None else np.maximum(maxs[k], hi)
                if args.include_actions and "actions" in grp:
                    arr = grp["actions"][:]
                    lo = arr.min(axis=0)
                    hi = arr.max(axis=0)
                    mins["__actions"] = lo if mins["__actions"] is None else np.minimum(mins["__actions"], lo)
                    maxs["__actions"] = hi if maxs["__actions"] is None else np.maximum(maxs["__actions"], hi)
        if (i + 1) % 10 == 0:
            print(f"#  scanned {i + 1}/{len(paths)}")

    block = {"obs": {}}
    for k in args.keys:
        if mins[k] is None:
            continue
        block["obs"][k] = {
            "min": [round(float(x), 5) for x in np.atleast_1d(mins[k])],
            "max": [round(float(x), 5) for x in np.atleast_1d(maxs[k])],
        }
    if args.include_actions and mins["__actions"] is not None:
        block["actions"] = {
            "min": [round(float(x), 5) for x in np.atleast_1d(mins["__actions"])],
            "max": [round(float(x), 5) for x in np.atleast_1d(maxs["__actions"])],
        }

    print("# paste into meta.obs_normalization:")
    print(yaml.safe_dump(block, default_flow_style=None, sort_keys=False))


if __name__ == "__main__":
    main()
