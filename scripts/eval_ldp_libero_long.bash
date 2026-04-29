#!/usr/bin/env bash
# Evaluate trained LDP checkpoints on libero_10 (one rollout per init state per
# task; n_eval_episodes split across n_eval_processes libero subprocesses).
#
# Required overrides:
#   experiment_folder=...
#   experiment_name=...
#   agent.vae_pretrain_path=PATH_TO_VAE_CKPT
#   data.train_latent_path=PATH_TO_LATENT_HDF5

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

PY=${PY:-/home/dodo/miniconda3/envs/ldp/bin/python}

set -x
"$PY" eval_bc.py \
    --config-name eval_libero_long \
    "$@"
