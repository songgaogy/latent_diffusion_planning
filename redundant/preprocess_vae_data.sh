export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
cd "$(dirname "$0")/.."
source scripts/env_helpers.bash
PY="$(resolve_ldp_python)"

"$PY" process_sdvae_data.py \
    experiment_folder=VAE_FOLDER \
    experiment_name=VAE_NAME \
    data=cfg/rm_lift/img \
    data.train_path=DATA_PATH \
    restore_snapshot_path=PATH_TO_VAE_CKPT
