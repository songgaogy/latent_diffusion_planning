# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Official implementation of *Latent Diffusion Planning* (Xie et al., 2025, arXiv:2504.16925). LDP trains imitation policies in a learned latent image space using two diffusion heads (planner + IDM) on top of a frozen Stable-Diffusion–style VAE. Targets Robomimic (`rm_lift`/`rm_can`/`rm_square`) and ALOHA-sim (`aloha_cube`) tasks.

## Environment & Execution

- Conda env name is `ldp`. Always run Python via `/home/dodo/miniconda3/envs/ldp/bin/python` (or activate `ldp` first). Do not assume the system `python`.
- Logging: prefer WandB. User's wandb settings: `WANDB_NAME=songgao-personal`, always export `WANDB_MODE=offline`. The codebase defaults to `use_wandb: false`; flip the flag in CLI/yaml if logging is desired. Note: `train_bc.py:248` has a literal `YOUR_ENTITY` placeholder — replace before enabling wandb.
- GPU selection: scripts in `scripts/` set `CUDA_VISIBLE_DEVICES=0,1`. JAX picks up all visible devices and shards batches positionally (`train_bc.py:74`); `batch_size % n_devices == 0` is asserted.
- For commands with long CLI parameters, add a `.bash` runner under `scripts/` and tell the user how to invoke it (per AGENTS.md convention; existing examples: `redundant/train_vae_encoder.sh`, `redundant/train_bc.sh`, `redundant/preprocess_vae_data.sh`, `scripts/train_mixed_bc.sh` — the last is mis-named: it actually runs `collect_data.py`).

## Common Commands

All training/eval entry points are Hydra apps; configs live next to the script (e.g. `train_bc.yaml`) and are composed with `defaults:` lists. Override anything from the CLI with dotted keys, and pick a top-level config via `-cn <name>`.

Pipeline (in order):

1. **Train VAE** — `python train_vae.py experiment_folder=… experiment_name=… data=cfg/rm_lift/mixed_img`
2. **Pre-encode images to latents** (avoids VAE inference in the dataloader) — `python process_sdvae_data.py experiment_folder=… experiment_name=… data=cfg/<task>/img data.train_path=… restore_snapshot_path=PATH_TO_VAE_CKPT`
3. **Train LDP agent** — `python train_bc.py agent=ldp_agent data=cfg/<task>/latent_img agent.vae_pretrain_path=… data.meta.obs_normalization.obs.latent_agentview_image.{min,max}=±10 horizon=9 action_horizon=4 -cn train_mixed_bc_<task>`
4. **Train IDM with mixed (expert + suboptimal) data** — same flags but use `train_mixed_bc.py`.
5. **Collect suboptimal rollouts** for mixed training — first run `train_bc.py` with small `n_grad_steps`, then `python collect_data.py … ckpt=1000 n_eval_episodes=500 save_path=…`.
6. **Eval** — `python eval_bc.py` (config inherits from `train_bc.yaml`; outputs to `experiments/${experiment_folder}_eval${folder_tag}/${experiment_name}_${eval_tag}`).

When using LDP with pre-encoded latents, the `data.meta.obs_normalization.obs.latent_<key>.{min,max}` bounds **must** match the actual latent stats — values out of range silently break inputs.

## Code Architecture

Hydra is the spine: every module (agent, data, model, network) is instantiated via `hydra.utils.instantiate(cfg.<x>)` from a `_target_:` key. To understand any component, start at the relevant yaml and follow `_target_` to the class.

- **Agents** (`agent/`): Flax `PyTreeNode`s with a `create()` classmethod and a JIT'd `update()`. Variants:
  - `dp_agent` — vanilla diffusion policy in pixel/lowdim space.
  - `dp_repr_agent` — diffusion policy on representation features.
  - `ldp_agent` — the LDP method: a **planner** (1D U-Net `ConditionalUnet1D`) over latent observations + an **IDM** (MLP-ResNet) that produces actions given two consecutive latents. Holds a frozen `FlaxAutoencoderKL` (`vae_module`/`vae_params`) used for `vae_encode`/`vae_decode` when raw images are present; if data is pre-encoded the VAE is bypassed.
  - `ldp_hier_agent` — hierarchical variant.
  Each agent yaml exposes `use_planner`/`use_idm`, alphas, diffusion step counts, and the obs_horizon/pred_horizon/action_horizon triple. The training loop drives `agent.update(batch, rng, step)` and (during eval) `agent.sample_action(batch, rng)` plus optional `agent.sample(...)`/`agent.sample_plan_stats(...)`.

- **Networks** (`networks/`): `diffusion_nets_v2.ConditionalUnet1D` (planner backbone), `mlp_diffusion_nets.MLPResNet` (IDM head), `diffusion.FourierFeatures` (time embed), `mlp_nets.MLP`, `resnet_v1` (image encoder).

- **Data** (`data/`): one module per (env, batching) combo — `robomimic_data` / `robomimic_latent_data` / `robomimic_mixed_data` / `robomimic_mixed_latent_data` and the matching `alohasim_*`. The `latent_*` variants load pre-encoded VAE latents from step 2; the `mixed_*` variants merge expert + suboptimal datasets. Configs in `data/cfg/<task>/{img,latent_img,mixed_img,mixed_latent_img}.yaml` carry `meta.obs_normalization`, `lowdim_obs`, `rgb_obs`, `env_params`, and `shape_meta`. Each data module exposes `train_dataloader()` / `eval_dataloader()` plus `name`, `shape_meta`, `env_params`, `batch_size`.

- **Envs** (`envs/`): `robosuite_env.py` (Robomimic) and `alohasim_env.py` / `alohasim_ee_env.py` (ALOHA). Eval routes through `utils.rm_env_utils.run_robomimic_eval` for `data.name.startswith("rm")` and `utils.aloha_env_utils.run_aloha_eval` for `"aloha" in data.name` (`train_bc.py:170`).

- **Model** (`model/stable_vae_model.py`): wraps Stable-Diffusion VAE for `train_vae.py` and `process_sdvae_data.py`.

- **Utilities** (`utils/`): `flax_utils.TrainStateEMA` is used everywhere (each agent state has both `params` and `ema_params`); `data_utils` provides `normalize_obs`/`unnormalize_obs`/`postprocess_batch`; `logger.Logger` handles TB+WandB; `py_utils.Every` is the ubiquitous step-trigger helper.

- **Checkpointing**: `orbax.PyTreeCheckpointer` writes to `<work_dir>/ckpt/<step>.ckpt` (each ckpt is a dict bundling `data`, resolved `cfg`, and the agent params via `agent.get_params()`). `Workspace.load_snapshot` (`train_bc.py:210`) selectively restores keys matching `<prefix>_params` into the corresponding `<prefix>_state`, with special-cased handling for `encoder_params` (shared vs per-camera) and EMA keys skipped. `cfg.restore_keys` (list) filters which prefixes to load — useful for loading just the planner or just the IDM.

- **Working directory**: Hydra rewrites cwd to `experiments/${experiment_folder}/${experiment_name}` (or `…_eval${folder_tag}/...` for eval). All ckpts/videos/logs are relative to that, so absolute paths must be used for `restore_snapshot_path` etc.

## Conventions Inherited from `AGENTS.md`

- **Minimal-diff edits.** Don't rename existing variables/functions or reformat unrelated code. Don't refactor or upgrade deps without being asked.
- **Plan before acting on non-trivial changes** (new files/modules, multi-file edits, new classes/algorithms, structural refactors). For obvious one-liner fixes or running commands, just do it.
- **Don't invent APIs.** PyTorch/JAX/Hydra/robosuite signatures must be verified against source or docs before use.
- **Code comments in English, short and precise.** Lengthy debugging or architectural explanations to the user can be in Chinese (user is a native Chinese speaker); status updates and code stay English.
- **Configurable seeds and exposed hyperparameters** for any new experimental feature; route through Hydra yaml rather than hard-coding.
