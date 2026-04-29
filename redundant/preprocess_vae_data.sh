export CUDA_VISIBLE_DEVICES=0,1
cd /home/dodo/Documents/latent_diffusion_planning

python process_sdvae_data.py 
    experiment_folder=VAE_FOLDER \
    experiment_name=VAE_NAME \
    data=cfg/rm_lift/img \
    data.train_path=DATA_PATH \
    restore_snapshot_path=PATH_TO_VAE_CKPT