#!/usr/bin/env bash
# Visualize decoded LDP planner outputs on libero_long training windows.
#
# Useful overrides:
#   CHECKPOINT=450000 NUM_SAMPLES=16 START_INDEX=100 STRIDE=10 bash scripts/visualize_ldp_libero_long.bash
#   OUT_DIR=/tmp/ldp_viz BATCH_SIZE=4 SEED=7 bash scripts/visualize_ldp_libero_long.bash
#   bash scripts/visualize_ldp_libero_long.bash --checkpoint 400000 --num-samples 2

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_NAME=${WANDB_NAME:-songgao-personal}
export HYDRA_FULL_ERROR=1

source scripts/env_helpers.bash
PY="$(resolve_ldp_python)"

EXPERIMENT_DIR=${EXPERIMENT_DIR:-experiments/libero_long/ldp_goal_cond_v3}
CHECKPOINT=${CHECKPOINT:-latest}        # either "latest" or a numeric step
NUM_SAMPLES=${NUM_SAMPLES:-8}
START_INDEX=${START_INDEX:-0}
STRIDE=${STRIDE:-1}
SEED=${SEED:-1}

ARGS=(
    --experiment-dir "$EXPERIMENT_DIR"
    --checkpoint "$CHECKPOINT"
    --num-samples "$NUM_SAMPLES"
    --start-index "$START_INDEX"
    --stride "$STRIDE"
    --seed "$SEED"
)

if [[ -n "${BATCH_SIZE:-}" ]]; then
    ARGS+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "${OUT_DIR:-}" ]]; then
    ARGS+=(--output-dir "$OUT_DIR")
fi

set -x
"$PY" utils/visualize_planner_outputs.py "${ARGS[@]}" "$@"
