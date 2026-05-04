"""Diagnose LDP planner training failure on libero_long.

Loads a checkpoint, runs:
  (1) plan_loss vs t (50 buckets) — does the U-Net fail uniformly or at specific t?
  (2) per-frame plan_mse on training data — naive baselines (zero / constant /
      identity) compared to the actual sampler output.
  (3) per-camera output statistics — is the model collapsing to constant?
"""
import os, sys, glob
sys.path.insert(0, '/home/gy/Documents/latent_diffusion_planning')
os.chdir('/home/gy/Documents/latent_diffusion_planning')

import numpy as np
import jax, jax.numpy as jnp
import yaml
import hydra
from omegaconf import OmegaConf, open_dict
OmegaConf.register_new_resolver("eval", eval, replace=True)
from pathlib import Path

# Compose the train config exactly as train_bc.py does
hydra.core.global_hydra.GlobalHydra.instance().clear()
with hydra.initialize(version_base=None, config_path="."):
    cfg = hydra.compose(config_name="train_libero_long", overrides=[
        f"agent.vae_pretrain_path={os.path.abspath('experiments/libero_vae/vae_all64/ckpt/200000.ckpt')}",
        f"data.train_latent_path={os.path.abspath('experiments/libero_long/preproc01/latent.hdf5')}",
        "n_workers=0",
        "batch_size=8",
    ])
print(OmegaConf.to_yaml(cfg.agent, resolve=False)[:200])

data = hydra.utils.instantiate(cfg.data)
loader = data.train_dataloader()
it = iter(loader)
batch = next(it)
batch = jax.tree.map(lambda t: t.numpy(), batch)
print("batch keys:", list(batch.keys()), "obs:", list(batch["obs"].keys()), "goal:", list(batch.get("goal_obs", {})))
print("latent_agentview shape:", batch["obs"]["latent_agentview_rgb"].shape)
print("actions shape:", batch["actions"].shape)

# Build agent same as train_bc.init_agent
agent_class = hydra.utils.get_class(cfg.agent._target_)
OmegaConf.resolve(cfg.agent)
with open_dict(cfg.agent):
    cfg.agent.pop("_target_")
rng = jax.random.PRNGKey(0)
agent = agent_class.create(rng, batch, data.shape_meta, **cfg.agent)
print("agent created.")

# Restore planner / idm params from the 250k ckpt
import orbax.checkpoint as ocp
from flax.training import orbax_utils
ckpt_path = os.path.abspath("experiments/libero_long/ldp_goal_cond_v0/ckpt/250000.ckpt")
ck = ocp.PyTreeCheckpointer()
target = {"planner_params": agent.planner_state.params, "idm_params": agent.idm_state.params}
restore_args = orbax_utils.restore_args_from_target(target)
restored = ck.restore(ckpt_path, item=target, restore_args=restore_args,
                      transforms={}, transforms_default_to_original=True)
print("ckpt top-level keys:", list(restored.keys()))
new_planner_state = agent.planner_state.replace(params=restored["planner_params"], ema_params=restored["planner_params"])
new_idm_state = agent.idm_state.replace(params=restored["idm_params"], ema_params=restored["idm_params"])
agent = agent.replace(planner_state=new_planner_state, idm_state=new_idm_state)
print("ckpt loaded.")

from utils.data_utils import postprocess_batch

# (1) plan_loss vs t bucket
@jax.jit
def planner_pred_at_t(params, batch_norm, t_val, key):
    obs_emb = agent.get_obs_cond(batch_norm["obs"])
    goal_cond = agent.get_goal_cond(batch_norm)
    next_obs_emb = obs_emb[:, agent.config["obs_horizon"]:]
    noise = jax.random.normal(key, shape=next_obs_emb.shape)
    t = jnp.full((next_obs_emb.shape[0],), t_val, dtype=jnp.int32)
    noisy = agent.planner_noise_scheduler.add_noise(agent.planner_noise_state, next_obs_emb, noise, t)
    obs_cond = obs_emb[:, :agent.config["obs_horizon"]].reshape(obs_emb.shape[0], -1)
    pred = agent.planner_state.apply_fn({"params": params}, noisy, t, obs_cond, goal_img_cond=goal_cond)
    return pred, noise, next_obs_emb, noisy

# Collect 8 batches for stable stats
batches = [batch]
for _ in range(7):
    b = next(it)
    b = jax.tree.map(lambda t: t.numpy(), b)
    batches.append(b)

T = 100
ts = list(range(0, T, 5))  # 20 buckets
loss_by_t = {}
key = jax.random.PRNGKey(1)
for t_val in ts:
    losses = []
    for b in batches:
        bn = postprocess_batch(b, agent.obs_normalization)
        key, sk = jax.random.split(key)
        pred, noise, _, _ = planner_pred_at_t(agent.planner_state.params, bn, t_val, sk)
        losses.append(float(jnp.mean((pred - noise) ** 2)))
    loss_by_t[t_val] = float(np.mean(losses))
print("\nplan_loss by t (squaredcos cap_v2 schedule, T=100):")
for t_val, l in loss_by_t.items():
    print(f"  t={t_val:3d}  loss={l:.4f}")

# (2) Per-frame MSE: how does the planner's full sampler output compare to GT?
# Use sample_viz to get the planner output
key, sk = jax.random.split(key)
b = batches[0]
out_actions, viz_metrics = agent.sample_viz(b, sk)
# Compute plan vs GT obs_emb in normalized space
# Reproduce what sample_viz_step does
bn = postprocess_batch(b, agent.obs_normalization)
gt_obs_emb = agent.get_obs_cond(bn["obs"])  # (B, 9, 520)
print("\ngt obs_emb stats: mean=%.4f std=%.4f min=%.4f max=%.4f" %
      (float(gt_obs_emb.mean()), float(gt_obs_emb.std()), float(gt_obs_emb.min()), float(gt_obs_emb.max())))

# Replicate sampler internals to get noisy_next_obs (the planner output)
# sample_viz_step does:
@jax.jit
def get_planner_output(b_norm, key):
    obs_emb = agent.get_obs_cond(b_norm["obs"])
    goal_cond = agent.get_goal_cond(b_norm)
    obs_cond = obs_emb[:, :agent.config["obs_horizon"]].reshape(obs_emb.shape[0], -1)
    key, nk = jax.random.split(key)
    noisy = jax.random.normal(nk, (obs_emb.shape[0], agent.config["pred_horizon"], agent.config["obs_dim"]))
    n = agent.config["planner_n_diffusion_steps"]
    def loop(i, args):
        cur, k = args
        sk, k = jax.random.split(k)
        kk = n - 1 - i
        nz = agent.planner_state.apply_fn({"params": agent.planner_state.params}, cur, kk, obs_cond, goal_img_cond=goal_cond)
        cur = agent.planner_noise_scheduler.step(agent.planner_noise_state, nz, kk, cur, sk).prev_sample
        return cur, k
    sk, key = jax.random.split(key)
    final, _ = jax.lax.fori_loop(0, n, loop, (noisy, sk))
    return final, obs_emb

planner_out, gt_obs_emb_v = get_planner_output(bn, key)
gt_future = gt_obs_emb_v[:, agent.config["obs_horizon"]:]  # (B, 8, 520)
print("\nplanner sampler output stats: mean=%.4f std=%.4f min=%.4f max=%.4f" %
      (float(planner_out.mean()), float(planner_out.std()), float(planner_out.min()), float(planner_out.max())))
print("gt_future stats:           mean=%.4f std=%.4f" % (float(gt_future.mean()), float(gt_future.std())))

print("\nper-frame plan_mse (sampler vs GT, vs naive baselines):")
for f in range(8):
    mse_planner = float(jnp.mean((planner_out[:, f] - gt_future[:, f]) ** 2))
    mse_zero = float(jnp.mean(gt_future[:, f] ** 2))
    # naive identity: predict current frame
    cur = gt_obs_emb_v[:, agent.config["obs_horizon"] - 1]
    mse_identity = float(jnp.mean((cur - gt_future[:, f]) ** 2))
    print(f"  frame {f+1}: planner_mse={mse_planner:.4f}  zero_mse={mse_zero:.4f}  identity_mse={mse_identity:.4f}")

# (3) Output collapse check: variance of planner output across batch elements per dim
po_var_per_dim = float(jnp.var(planner_out, axis=0).mean())
gt_var_per_dim = float(jnp.var(gt_future, axis=0).mean())
print(f"\nper-dim variance across batch:  planner_out={po_var_per_dim:.4f}  gt={gt_var_per_dim:.4f}")
