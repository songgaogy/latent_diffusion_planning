export CUDA_VISIBLE_DEVICES=0,1
cd /home/dodo/Documents/latent_diffusion_planning

python train_bc.py \
    experiment_folder=FOLDER \
    experiment_name=NAME \
    data=cfg/rm_lift/img \
    n_grad_steps=1000 \
    save_every_step=1000 \
    warmup_steps=10
