#!/usr/bin/env bash
# Encode all libero_10 raw images (both cameras) into latents using a trained
# SD-VAE checkpoint. Writes latent.hdf5 under
# experiments/${experiment_folder}/${experiment_name}/.
#
# Required:
#   restore_snapshot_path=PATH_TO_VAE_CKPT  (full absolute path to a numbered
#                                            ckpt directory, e.g.
#                                            experiments/libero_vae/vae_all256/ckpt/100000.ckpt)
#
# Override examples:
#   bash scripts/preprocess_libero.bash \
#       experiment_folder=libero_long experiment_name=preproc01 \
#       restore_snapshot_path=$(pwd)/experiments/libero_vae/vae_all256/ckpt/100000.ckpt
#
# After this finishes, run:
#   /home/dodo/miniconda3/envs/ldp/bin/python scripts/compute_libero_latent_stats.py \
#       --latent experiments/<folder>/<name>/latent.hdf5
# and paste the printed min/max values into data/cfg/libero_long/latent_img.yaml.

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_NAME=${WANDB_NAME:-songgao-personal}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PY=${PY:-/home/dodo/miniconda3/envs/ldp/bin/python}

set -x
"$PY" process_sdvae_data.py \
    data=cfg/libero_long/img \
    horizon=1 \
    obs_horizon=1 \
    shard=64 \
    "$@"
