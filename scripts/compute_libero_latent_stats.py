"""Read latent.hdf5 (output of process_sdvae_data.py run_libero) and print
per-key min/max stats ready to paste into latent_img.yaml.

Usage:
  python scripts/compute_libero_latent_stats.py --latent path/to/latent.hdf5
"""
import argparse

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", required=True, help="path to latent.hdf5")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="recompute min/max by reading every demo (default: read attrs)",
    )
    args = parser.parse_args()

    with h5py.File(args.latent, "r") as f:
        data_grp = f["data"]
        attrs = dict(data_grp.attrs)
        print(f"# {args.latent}: {attrs.get('total', '?')} demos")

        keys = set()
        for demo_id in data_grp.keys():
            for k in data_grp[demo_id]["latent"].keys():
                keys.add(k)
            break  # all demos share the same latent keys

        if args.rescan:
            mins = {k: float("inf") for k in keys}
            maxs = {k: float("-inf") for k in keys}
            n = len(data_grp.keys())
            for i, demo_id in enumerate(data_grp.keys()):
                for k in keys:
                    arr = data_grp[demo_id]["latent"][k][:]
                    mins[k] = min(mins[k], float(arr.min()))
                    maxs[k] = max(maxs[k], float(arr.max()))
                if (i + 1) % 50 == 0:
                    print(f"# scanned {i + 1}/{n}")
        else:
            mins, maxs = {}, {}
            for k in keys:
                mins[k] = float(attrs.get(f"min_z_{k}", attrs["min_z"]))
                maxs[k] = float(attrs.get(f"max_z_{k}", attrs["max_z"]))

    print("# paste under data.meta.obs_normalization.obs:")
    for k in sorted(keys):
        lo, hi = mins[k], maxs[k]
        # symmetric pad of 5% to avoid clipping at training time
        pad = 0.05 * (hi - lo)
        print(f"      latent_{k}:")
        print(f"        min: {lo - pad:.4f}")
        print(f"        max: {hi + pad:.4f}")


if __name__ == "__main__":
    main()
