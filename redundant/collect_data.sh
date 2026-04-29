export CUDA_VISIBLE_DEVICES=0,1
cd /home/dodo/Documents/latent_diffusion_planning

python collect_data.py \
    experiment_folder=FOLDER \
    experiment_name=NAME \
    folder_tag=1 \
    eval_tag=test \
    ckpt=1000 \
    n_eval_episodes=500 \
    save_path=PATH_TO_SAVE \
    data.env_params.env_kwargs.lowdim_obs=[robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object]