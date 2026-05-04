#!/usr/bin/env bash
# Train the SD-VAE on all 5 LIBERO suites (~5500 demos) at 64x64.
#
# Override examples:
#   bash scripts/train_vae_libero.bash experiment_name=run01 batch_size=8
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_vae_libero.bash batch_size=8 n_workers=2

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_NAME=${WANDB_NAME:-songgao-personal}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HYDRA_FULL_ERROR=1

source scripts/env_helpers.bash
PY="$(resolve_ldp_python)"

set -x
"$PY" train_vae.py \
    --config-name train_vae_libero \
    "$@"
