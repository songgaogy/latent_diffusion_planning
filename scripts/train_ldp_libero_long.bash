#!/usr/bin/env bash
# Train the LDP planner + IDM on libero_10 latents.
#
# Required overrides:
#   agent.vae_pretrain_path=PATH_TO_VAE_CKPT  - the SD-VAE checkpoint trained
#       on libero_all (e.g. experiments/libero_vae/vae_all256/ckpt/100000.ckpt)
#   data.train_latent_path=PATH_TO_LATENT_HDF5 - latent.hdf5 produced by
#       scripts/preprocess_libero.bash
#
# Also paste the latent min/max values printed by
# scripts/compute_libero_latent_stats.py into
# data/cfg/libero_long/latent_img.yaml before running.
#
# Override examples:
#   bash scripts/train_ldp_libero_long.bash \
#       experiment_name=run01 \
#       agent.vae_pretrain_path=$(pwd)/experiments/libero_vae/vae_all256/ckpt/100000.ckpt \
#       data.train_latent_path=$(pwd)/experiments/libero_long/preproc01/latent.hdf5

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_NAME=${WANDB_NAME:-songgao-personal}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

PY=${PY:-/home/dodo/miniconda3/envs/ldp/bin/python}

set -x
"$PY" train_bc.py \
    --config-name train_libero_long \
    "$@"
