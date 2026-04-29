export CUDA_VISIBLE_DEVICES=0,1
cd /home/dodo/Documents/latent_diffusion_planning

python train_vae.py \
    experiment_folder=VAE_FOLDER \
    experiment_name=VAE_NAME \
    data=cfg/rm_lift/mixed_img
