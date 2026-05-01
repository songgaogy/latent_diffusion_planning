#!/usr/bin/env python3
"""Expand each MP4 under a directory into a folder of per-frame comparison PNGs.

Eval videos (e.g. LIBERO) store each frame as CHW/HWC RGB with ground-truth camera
on the left and decoded plan on the right. This script splits at mid-width and
rewrites a single PNG per frame: optional separator + labels in filename only.

Usage:
  ./scripts/mp4_to_compare_frame_dirs.py experiments/libero_long/ldp_long01/video
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


def add_vertical_separator(left: np.ndarray, right: np.ndarray, sep_w: int) -> np.ndarray:
    """Concatenate HxWx3 uint8 arrays with a white vertical bar."""
    if sep_w <= 0:
        return np.concatenate([left, right], axis=1)
    h = left.shape[0]
    bar = np.full((h, sep_w, 3), 255, dtype=np.uint8)
    return np.concatenate([left, bar, right], axis=1)


def split_compare_frame(frame: np.ndarray, sep_w: int) -> np.ndarray:
    """frame: HWC uint8. Left = GT, right = plan (eval convention)."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB, got shape {frame.shape}")
    w = frame.shape[1]
    mid = w // 2
    left = frame[:, :mid, :].copy()
    right = frame[:, mid:, :].copy()
    return add_vertical_separator(left, right, sep_w)


def process_mp4(mp4_path: Path, sep_w: int, dry_run: bool) -> int:
    stem = mp4_path.stem
    out_dir = mp4_path.with_suffix("")  # same parent, name without .mp4
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} exists and is not a directory")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for frame in iio.imiter(mp4_path, plugin="FFMPEG"):
        arr = np.asarray(frame)
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        compare = split_compare_frame(arr, sep_w)
        out_path = out_dir / f"frame_{n:06d}.png"
        if not dry_run:
            Image.fromarray(compare).save(out_path)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description="MP4 → folder of GT|plan comparison PNGs")
    p.add_argument(
        "video_dir",
        type=Path,
        help="Directory containing .mp4 files (e.g. experiments/.../video)",
    )
    p.add_argument(
        "--separator-width",
        type=int,
        default=0,
        help="White pixels between left (GT) and right (plan); 0 to only split.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files.",
    )
    args = p.parse_args()
    video_dir = args.video_dir.expanduser().resolve()
    if not video_dir.is_dir():
        raise SystemExit(f"Not a directory: {video_dir}")

    mp4s = sorted(video_dir.glob("*.mp4"))
    if not mp4s:
        raise SystemExit(f"No .mp4 files in {video_dir}")

    for mp4_path in mp4s:
        n = process_mp4(mp4_path, args.separator_width, args.dry_run)
        print(f"{mp4_path.name} -> {mp4_path.stem}/ ({n} frames)")


if __name__ == "__main__":
    main()
