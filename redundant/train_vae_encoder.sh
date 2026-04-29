export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
cd "$(dirname "$0")/.."
source scripts/env_helpers.bash
PY="$(resolve_ldp_python)"

"$PY" train_vae.py \
    experiment_folder=VAE_FOLDER \
    experiment_name=VAE_NAME \
    data=cfg/rm_lift/mixed_img
