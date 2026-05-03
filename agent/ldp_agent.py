from einops import rearrange
from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import hydra
from omegaconf import OmegaConf, open_dict
import optax
import orbax
import orbax.checkpoint as ckpt 
from pathlib import Path
import yaml

from diffusers import FlaxAutoencoderKL
from diffusers.schedulers.scheduling_ddpm_flax import FlaxDDPMScheduler, FlaxDDPMSchedulerOutput
from flax.core import FrozenDict
from flax.training import orbax_utils
from networks.mlp_diffusion_nets import MLPDiffusion
from optax._src import linear_algebra
import utils.flax_utils as flax_utils
from utils.flax_utils import nonpytree_field
from utils.data_utils import postprocess_batch, postprocess_batch_obs, normalize_obs, unnormalize_obs

import numpy as np

def _patch_flax_trace_level_for_jax_06():
    import flax.core.tracers as flax_tracers

    def trace_level(main):
        if main:
            return getattr(
                main,
                "level",
                getattr(getattr(main, "main", None), "level", float("-inf")),
            )
        return float("-inf")

    if not hasattr(jax.core.find_top_trace(()), "level"):
        flax_tracers.trace_level = trace_level

_patch_flax_trace_level_for_jax_06()

def _patch_diffusers_ddpm_step_for_jax_06():
    if getattr(FlaxDDPMScheduler.step, "_ldp_jax06_patch", False):
        return

    def step(self, state, model_output, timestep, sample, key=None, return_dict=True):
        t = timestep
        if key is None:
            key = jax.random.PRNGKey(0)

        if model_output.shape[1] == sample.shape[1] * 2 and self.config.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = jnp.split(model_output, sample.shape[1], axis=1)
        else:
            predicted_variance = None

        alpha_prod_t = state.common.alphas_cumprod[t]
        alpha_prod_t_prev = jnp.where(t > 0, state.common.alphas_cumprod[t - 1], jnp.array(1.0, dtype=self.dtype))
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            raise ValueError(
                f"prediction_type {self.config.prediction_type} must be epsilon, sample, or v_prediction"
            )

        if self.config.clip_sample:
            pred_original_sample = jnp.clip(pred_original_sample, -1, 1)

        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * state.common.betas[t]) / beta_prod_t
        current_sample_coeff = state.common.alphas[t] ** 0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        def random_variance():
            split_key = jax.random.split(key, num=1)[0]
            noise = jax.random.normal(split_key, shape=model_output.shape, dtype=self.dtype)
            return self._get_variance(state, t, predicted_variance=predicted_variance) ** 0.5 * noise

        variance = jnp.where(t > 0, random_variance(), jnp.zeros(model_output.shape, dtype=self.dtype))
        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample, state)
        return FlaxDDPMSchedulerOutput(prev_sample=pred_prev_sample, state=state)

    step._ldp_jax06_patch = True
    FlaxDDPMScheduler.step = step

_patch_diffusers_ddpm_step_for_jax_06()

class LDPAgent(flax.struct.PyTreeNode):
    planner_state : flax_utils.TrainStateEMA
    idm_state: flax_utils.TrainStateEMA
    vae_module: Any = nonpytree_field()
    vae_params: dict[str, FrozenDict]
    obs_normalization: dict[str, Any]
    use_planner: bool
    use_idm: bool
    alpha_planner: float
    alpha_idm: float
    lr_schedule: Any = nonpytree_field()
    planner_noise_scheduler: Any = nonpytree_field()
    planner_noise_state: Any = nonpytree_field() 
    idm_noise_scheduler: Any = nonpytree_field()
    idm_noise_state: Any = nonpytree_field()
    idm_noise_state: Any = nonpytree_field()
    config: dict = nonpytree_field() # check create function for definition

    @jax.jit
    def vae_encode(self, batch):
        new_batch = dict()
        for key in batch.keys():
            if not f'latent_{key}' in self.config['rgb_obs']:
                new_batch[key] = batch[key]
                continue
            init_obs = batch[key]

            B, H = init_obs.shape[:2]
            init_obs = init_obs.reshape(-1, *init_obs.shape[-3:])
            if key not in self.obs_normalization["obs"]:
                init_obs = init_obs / 255.0
                init_obs = (init_obs - 0.5) / 0.5
            # init_obs transpose from BHWC to BCHW
            init_obs_resize = jnp.transpose(init_obs, (0, 3, 1, 2))
            z = self.vae_module.apply({"params": self.vae_params}, init_obs_resize, method=self.vae_module.encode)['latent_dist'].mean
            feats = z.reshape(B, H, -1)
            feats = jax.lax.stop_gradient(feats)
            feats = normalize_obs({f'latent_{key}': feats}, self.obs_normalization['obs'])[f'latent_{key}']
            new_batch[f'latent_{key}'] = feats 
        return new_batch

    @jax.jit
    def vae_decode_full(self, full_latent):
        """Decode an *untruncated* per-camera latent.

        Used by sample_viz_step to render the start frame of the plan video
        from the full (8,8,4) latent we observe at inference, while predicted
        future frames still go through the truncated `vae_decode` path
        (planner only outputs `vae_feature_dim` dims). Without this the
        right-hand video panel shows a partly-zero-padded latent and the
        decoder hallucinates the missing rows.

        Accepts shape (B, H, 8, 8, 4) or (B, H, 256); both are reshaped to
        (B*H, 8, 8, 4) for the decoder.
        """
        B, H = full_latent.shape[:2]
        z = full_latent.reshape(B * H, 8, 8, 4)
        key = self.config['rgb_obs'][0]
        z = unnormalize_obs({key: z}, self.obs_normalization['obs'])[key]
        reconstruct = self.vae_module.apply({"params": self.vae_params}, z, method=self.vae_module.decode).sample
        reconstruct = reconstruct.reshape(B, H, *reconstruct.shape[1:])
        return reconstruct

    @staticmethod
    def _spatial_project_latent(feat, vae_feature_dim, data_name):
        """Project a per-camera SD-VAE latent into the planner's
        ``vae_feature_dim`` slot.

        Two strategies:
          * libero + vae_feature_dim==64: 2x2 spatial avg-pool on the 8x8
            grid (keep all 4 channels). Pooled latent has 4*4*4=64 dims and
            every cell still represents a region of the original image, so
            the planner sees the table / objects / gripper instead of just
            the top-2-rows wallpaper strip the legacy "first 64 flat dims"
            truncation kept.
          * otherwise: legacy ``[..., :vae_feature_dim]`` flat truncation,
            so rm_lift / aloha runs are unaffected.

        Accepts ``feat`` of shape (..., 256) (post-vae_encode flat) or
        (..., 8, 8, 4) (pre-encoded latent.hdf5). Returns (..., vfd).
        """
        is_spatial = (
            feat.ndim >= 3
            and feat.shape[-3] == 8
            and feat.shape[-2] == 8
            and feat.shape[-1] == 4
        )
        if is_spatial:
            spatial = feat
        else:
            # flat (..., 256) -> (..., 8, 8, 4)
            spatial = feat.reshape(*feat.shape[:-1], 8, 8, 4)
        leading = spatial.shape[:-3]
        if "libero" in (data_name or "") and vae_feature_dim == 64:
            # reshape (8, 8, 4) -> (4, 2, 4, 2, 4) and avg over the two
            # length-2 axes (the within-block H/W axes).
            pooled = spatial.reshape(*leading, 4, 2, 4, 2, 4).mean(axis=(-2, -4))
            return pooled.reshape(*leading, vae_feature_dim)
        # legacy truncation
        return spatial.reshape(*leading, -1)[..., :vae_feature_dim]

    @jax.jit
    def vae_decode(self, feats):
        B, H = feats.shape[:2]
        if self.config['vae_feature_dim'] == 16:
            feats = feats[:, :, :16]
            z = feats.reshape(B * H, 2, 2, 4)
        elif self.config['vae_feature_dim'] == 32:
            feats = feats[:, :, :32]
            z = feats.reshape(B * H, 2, 2, 8) 
        elif self.config['vae_feature_dim'] == 36:
            feats = feats[:, :, :36]
            z = feats.reshape(B * H, 3, 3, 4)
        elif self.config['vae_feature_dim'] == 64:
            feats = feats[:, :, :64]
            if "libero" in self.config["data_name"]:
                # planner output is the 2x2 spatial-avg-pooled libero latent
                # ((4,4,4) shape, mirroring _spatial_project_latent). Nearest-
                # upsample 4x4 -> 8x8 to feed SD-VAE decoder, which expects
                # 8x8x4 for 256x256 reconstructions.
                z = feats.reshape(B * H, 4, 4, 4)
                z = jnp.repeat(jnp.repeat(z, 2, axis=1), 2, axis=2)
            else:
                z = feats.reshape(B * H, 4, 4, 4)
        elif self.config['vae_feature_dim'] == 256:
            # 256x256 SD-VAE input -> 32x32x4 latent; take first 256 flat dims
            # reshaped as 8x8x4 for the (learned, upsampling) decoder.
            feats = feats[:, :, :256]
            z = feats.reshape(B * H, 8, 8, 4)
        elif self.config['vae_feature_dim'] == 1024:
            feats = feats[:, :, :1024]
            z = feats.reshape(B * H, 16, 16, 4)
        elif self.config['vae_feature_dim'] == 4096:
            # full 256x256 SD-VAE latent (no truncation)
            z = feats[:, :, :4096].reshape(B * H, 32, 32, 4)
        key = self.config['rgb_obs'][0]
        z = unnormalize_obs({key: z}, self.obs_normalization['obs'])[key]
        reconstruct = self.vae_module.apply({"params": self.vae_params}, z, method=self.vae_module.decode).sample
        reconstruct = reconstruct.reshape(B, H, *reconstruct.shape[1:]) # B, H, 3, W, W
        return reconstruct


    @jax.jit
    def get_obs_cond(self, batch):
        lowdim_obs_cond = jnp.concatenate([batch[key] for key in self.config['lowdim_obs']], axis=-1).astype(jnp.float32)
        B, H = lowdim_obs_cond.shape[:2]
        lowdim_obs_cond = lowdim_obs_cond.reshape(-1, *lowdim_obs_cond.shape[2:])
        # Per-camera image latent -> vae_feature_dim, via _spatial_project_latent
        # (2x2 avg-pool for libero+64, legacy flat truncation otherwise).
        per_cam = [
            self._spatial_project_latent(
                batch[key],
                self.config['vae_feature_dim'],
                self.config.get('data_name'),
            )
            for key in self.config['rgb_obs']
        ]
        image_features = jnp.concatenate(per_cam, axis=-1)
        lowdim_obs_cond = lowdim_obs_cond.reshape(B, H, -1)
        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)
        return obs_cond

    @classmethod
    def _get_obs_cond(cls, batch, rgb_obs, lowdim_obs, obs_horizon, init_enc_rng=None, vae_feature_dim=None, data_name=None):
        lowdim_obs_cond = jnp.concatenate([batch[key] for key in lowdim_obs], axis=-1).astype(jnp.float32)
        B, H = lowdim_obs_cond.shape[:2]
        lowdim_obs_cond = lowdim_obs_cond.reshape(-1, *lowdim_obs_cond.shape[2:])

        if vae_feature_dim is not None:
            per_cam = [
                cls._spatial_project_latent(batch[key], vae_feature_dim, data_name)
                for key in rgb_obs
            ]
        else:
            per_cam = [batch[key].reshape(B, H, -1) for key in rgb_obs]
        image_features = jnp.concatenate(per_cam, axis=-1)

        lowdim_obs_cond = lowdim_obs_cond.reshape(B, H, -1)
        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)
        return obs_cond

    @staticmethod
    def _get_goal_cond(goal_obs, goal_rgb_obs, vae_feature_dim=None, data_name=None):
        if goal_obs is None or len(goal_rgb_obs) == 0:
            return None
        per_cam = []
        for key in goal_rgb_obs:
            v = goal_obs[key]
            if vae_feature_dim is not None and key.startswith("latent_"):
                v = LDPAgent._spatial_project_latent(v, vae_feature_dim, data_name)
            else:
                v = v.reshape(v.shape[0], -1)
            per_cam.append(v)
        return jnp.concatenate(per_cam, axis=-1)

    def get_goal_cond(self, batch):
        if not self.config['use_goal_cond'] or 'goal_obs' not in batch:
            return None
        return self._get_goal_cond(
            batch['goal_obs'],
            self.config['goal_rgb_obs'],
            self.config['vae_feature_dim'],
            self.config.get('data_name'),
        )

    def plan_loss(self, params, rng, obs_emb, obs_horizon, goal_img_cond=None):
        # noising
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (obs_emb.shape[0],), 0, self.config['planner_n_diffusion_steps'])
        next_obs_emb = obs_emb[:, obs_horizon:]
        noise = jax.random.normal(noise_rng, shape=next_obs_emb.shape)
        noisy_next_obs_emb = self.planner_noise_scheduler.add_noise(self.planner_noise_state, next_obs_emb, noise, t)

        obs_cond = obs_emb[:, :obs_horizon, ...] # tbh I'm not sure this is correct
        obs_cond = obs_cond.reshape(obs_emb.shape[0], -1)
        pred_noise = self.planner_state.apply_fn({"params": params}, noisy_next_obs_emb, t, obs_cond, goal_img_cond=goal_img_cond)
        loss = jnp.mean((pred_noise - noise) ** 2)
        metrics = dict()
        return loss, metrics

    def idm_loss(self, params, rng, obs_emb, actions, obs_horizon):
        s_sprime = rearrange(jnp.concatenate((obs_emb[:, obs_horizon-1:-1, :], obs_emb[:, obs_horizon:, :]), axis=-1), 'B H D -> (B H) D')
        actions = jnp.reshape(actions[:, :-1], (-1, *actions.shape[2:]))

        # noising
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (actions.shape[0], 1), 0, self.config['idm_n_diffusion_steps'])
        noise = jax.random.normal(noise_rng, shape=actions.shape)
        noisy_actions = self.idm_noise_scheduler.add_noise(self.idm_noise_state, actions, noise, t)
        pred_noise = self.idm_state.apply_fn({"params": params}, s_sprime, noisy_actions, t)
        loss = jnp.mean((pred_noise - noise) ** 2)
        return loss

    def loss(self, params, batch, rng, use_planner, use_idm, obs_horizon):
        obs_emb = self.get_obs_cond(batch['obs'])
        goal_img_cond = self.get_goal_cond(batch)
        action = batch['actions']

        if use_planner:
            rng, plan_rng = jax.random.split(rng)
            plan_loss, plan_metrics = self.plan_loss(params['planner'], plan_rng, obs_emb, obs_horizon, goal_img_cond)
            plan_loss = self.alpha_planner * plan_loss
        else:
            plan_loss = 0
            plan_metrics = dict()

        if use_idm:
            rng, idm_rng = jax.random.split(rng)
            idm_loss = self.idm_loss(params['idm'], idm_rng, obs_emb, action, obs_horizon)
            idm_loss = self.alpha_idm * idm_loss
        else:
            idm_loss = 0

        loss = plan_loss + idm_loss

        metrics = dict(plan_loss=plan_loss, idm_loss=idm_loss, loss=loss)
        metrics.update(plan_metrics)

        metrics['emb_min'] = jnp.min(obs_emb)
        metrics['emb_max'] = jnp.max(obs_emb)
        metrics['emb_mean'] = jnp.mean(obs_emb)
        metrics['emb_std'] = jnp.std(obs_emb)
        if goal_img_cond is not None:
            metrics['goal_cond_min'] = jnp.min(goal_img_cond)
            metrics['goal_cond_max'] = jnp.max(goal_img_cond)
            metrics['goal_cond_std'] = jnp.std(goal_img_cond)

        metrics['action_min'] = jnp.min(action)
        metrics['action_max'] = jnp.max(action)

        # debugging
        debug_metrics = dict()
        for key in batch['obs']:
            debug_metrics[f"{key}_min"] = jnp.min(batch['obs'][key])#.item()
            debug_metrics[f"{key}_max"] = jnp.max(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_mean"] = jnp.mean(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_std"] = jnp.std(batch['obs'][key])#.item()
        metrics.update(debug_metrics)

        return loss, metrics

    def loss_mixed(self, params, batch, mixed_batch, rng, use_planner, use_idm, obs_horizon):
        obs_emb = self.get_obs_cond(batch['obs'])
        goal_img_cond = self.get_goal_cond(batch)
        action = batch['actions']
        mixed_obs_emb = self.get_obs_cond(mixed_batch['obs'])
        mixed_action = mixed_batch['actions']

        if use_planner:
            rng, plan_rng = jax.random.split(rng)
            plan_loss, plan_metrics = self.plan_loss(params['planner'], plan_rng, obs_emb, obs_horizon, goal_img_cond)
            plan_loss = self.alpha_planner * plan_loss
        else:
            plan_loss = 0
            plan_metrics = dict()

        if use_idm:
            rng, idm_rng = jax.random.split(rng)
            idm_loss = self.idm_loss(params['idm'], idm_rng, mixed_obs_emb, mixed_action, obs_horizon)
            idm_loss = self.alpha_idm * idm_loss
        else:
            idm_loss = 0

        loss = plan_loss + idm_loss

        metrics = dict(plan_loss=plan_loss, idm_loss=idm_loss, loss=loss)
        metrics.update(plan_metrics)

        metrics['emb_min'] = jnp.min(obs_emb)
        metrics['emb_max'] = jnp.max(obs_emb)
        metrics['emb_mean'] = jnp.mean(obs_emb)
        metrics['emb_std'] = jnp.std(obs_emb)
        if goal_img_cond is not None:
            metrics['goal_cond_min'] = jnp.min(goal_img_cond)
            metrics['goal_cond_max'] = jnp.max(goal_img_cond)
            metrics['goal_cond_std'] = jnp.std(goal_img_cond)

        metrics['action_min'] = jnp.min(action)
        metrics['action_max'] = jnp.max(action)

        # debugging
        debug_metrics = dict()
        for key in batch['obs']:
            debug_metrics[f"{key}_min"] = jnp.min(batch['obs'][key])#.item()
            debug_metrics[f"{key}_max"] = jnp.max(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_mean"] = jnp.mean(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_std"] = jnp.std(batch['obs'][key])#.item()
        metrics.update(debug_metrics)

        return loss, metrics

    def update(self, batch, rng, step):
        use_planner = bool(self.use_planner) and step % self.config['update_planner_every'] == 0
        use_idm = bool(self.use_idm) and step % self.config['update_idm_every'] == 0
        use_idm = use_idm and step >= self.config['update_idm_after']
        update_planner = self.config['update_planner_until'] < 0 or step < self.config['update_planner_until']
        update_planner = update_planner and step >= self.config['update_planner_after']
        use_planner = use_planner and update_planner
        return self.update_step(batch, rng, use_planner, use_idm, 
                self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('use_planner', 'use_idm', 'obs_horizon'))
    def update_step(self, batch, rng, use_planner, use_idm, obs_horizon):
        batch = postprocess_batch(batch, self.obs_normalization)
        rng, g_rng = jax.random.split(rng)
        
        # get params
        combined_params = dict()
        if use_planner:
            combined_params['planner'] = self.planner_state.params
        if use_idm:
            combined_params['idm'] = self.idm_state.params
        
        # Compute loss and gradients
        grads, metrics = jax.grad(self.loss, has_aux=True)(combined_params, batch, g_rng, use_planner, use_idm, obs_horizon)
        g_norm = linear_algebra.global_norm(grads)
        metrics['g_norm'] = g_norm
        if use_planner:
            planner_grads = grads['planner']
            new_planner_state = self.planner_state.apply_gradients(grads=planner_grads)
            metrics["planner_lr"] = self.lr_schedule(self.planner_state.step)
            metrics["planner_step"] = self.planner_state.step
        else:
            new_planner_state = self.planner_state
            metrics["planner_lr"] = 0
            metrics["planner_step"] = 0
            metrics["noise_diff"] = 0
        if use_idm:
            idm_grads = grads['idm']
            new_idm_state = self.idm_state.apply_gradients(grads=idm_grads)
            metrics["idm_lr"] = self.lr_schedule(self.idm_state.step)
            metrics["idm_step"] = self.idm_state.step
        else:
            new_idm_state = self.idm_state
            metrics["idm_lr"] = 0
            metrics["idm_step"] = 0

        
        return self.replace(planner_state=new_planner_state, 
                            idm_state=new_idm_state), metrics

    def update_mixed(self, batch, mixed_batch, rng, step):
        use_planner = bool(self.use_planner) and step % self.config['update_planner_every'] == 0
        use_idm = bool(self.use_idm) and step % self.config['update_idm_every'] == 0
        use_idm = use_idm and step >= self.config['update_idm_after']
        update_planner = self.config['update_planner_until'] < 0 or step < self.config['update_planner_until']
        update_planner = update_planner and step >= self.config['update_planner_after']
        use_planner = use_planner and update_planner
        return self.update_mixed_step(batch, mixed_batch, rng, use_planner, use_idm, 
                self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('use_planner', 'use_idm', 'obs_horizon'))
    def update_mixed_step(self, batch, mixed_batch, rng, use_planner, use_idm, obs_horizon):
        batch = postprocess_batch(batch, self.obs_normalization)
        mixed_batch = postprocess_batch(mixed_batch, self.obs_normalization)
        rng, g_rng = jax.random.split(rng)
        
        # get params
        combined_params = dict()
        if use_planner:
            combined_params['planner'] = self.planner_state.params
        if use_idm:
            combined_params['idm'] = self.idm_state.params
        
        # Compute loss and gradients
        grads, metrics = jax.grad(self.loss_mixed, has_aux=True)(combined_params, batch, mixed_batch, g_rng, use_planner, use_idm, obs_horizon)
        g_norm = linear_algebra.global_norm(grads)
        metrics['g_norm'] = g_norm
        if use_planner:
            planner_grads = grads['planner']
            new_planner_state = self.planner_state.apply_gradients(grads=planner_grads)
            metrics["planner_lr"] = self.lr_schedule(self.planner_state.step)
            metrics["planner_step"] = self.planner_state.step
        else:
            new_planner_state = self.planner_state
            metrics["planner_lr"] = 0
            metrics["planner_step"] = 0
            metrics["noise_diff"] = 0
        if use_idm:
            idm_grads = grads['idm']
            new_idm_state = self.idm_state.apply_gradients(grads=idm_grads)
            metrics["idm_lr"] = self.lr_schedule(self.idm_state.step)
            metrics["idm_step"] = self.idm_state.step
        else:
            new_idm_state = self.idm_state
            metrics["idm_lr"] = 0
            metrics["idm_step"] = 0

        return self.replace(planner_state=new_planner_state, 
                            idm_state=new_idm_state), metrics
    def get_metrics(self, batch, rng):

        return self.get_metrics_step(batch, rng, 
                bool(self.use_planner), bool(self.use_idm), 
                self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('use_planner', 'use_idm', 'obs_horizon'))
    def get_metrics_step(self, batch, rng, use_planner, use_idm, obs_horizon):
        batch = postprocess_batch(batch, self.obs_normalization)
        rng, g_rng = jax.random.split(rng)

        # get params
        combined_params = dict()
        if use_planner:
            combined_params['planner'] = self.planner_state.params
        if use_idm:
            combined_params['idm'] = self.idm_state.params
        
        # Compute loss and gradients
        loss, metrics = self.loss(combined_params, batch, g_rng, use_planner, use_idm, obs_horizon)  
        return metrics

    def sample_action_from_plan(self, batch, next_plan, eval_rng):
        if 'actions' in batch.keys():
            batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert len(batch.keys()) == 1
            batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)
        batch['obs'] = self.vae_encode(batch['obs'])
        return self.sample_action_from_plan_step(batch, next_plan, eval_rng, self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('obs_horizon'))
    def sample_action_from_plan_step(self, batch, next_plan, eval_rng, obs_horizon):
        # Planner
        for k, v in batch['obs'].items():
            B = v.shape[0]
            break

        start_plan = self.get_obs_cond(batch['obs'])

        # IDM
        s_sprime = rearrange(jnp.concatenate((start_plan, next_plan), axis=-1), 'B H D -> (B H) D')
        transition_emb = jnp.concatenate((s_sprime[:, :self.config['obs_dim']], s_sprime[:, self.config['obs_dim']:]), axis=1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (transition_emb.shape[0], self.config['action_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['idm_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.idm_state.apply_fn({"params": self.idm_state.params}, transition_emb, noisy_action, k)
            noisy_action = self.idm_noise_scheduler.step(self.idm_noise_state, noise_pred, k, noisy_action, s_rng).prev_sample

            return noisy_action, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))
        action = rearrange(noisy_action, '(B H) D -> B H D', B=B)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action

    def sample_action(self, batch, eval_rng):
        if 'actions' in batch.keys():
            batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert len(batch.keys()) == 1
            batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)
        batch['obs'] = self.vae_encode(batch['obs'])
        return self.sample_action_step(batch, eval_rng, self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('obs_horizon'))
    def sample_action_step(self, batch, eval_rng, obs_horizon):
        # Planner
        for k, v in batch['obs'].items():
            B = v.shape[0]
            break

        plan = self.get_obs_cond(batch['obs'])

        # IDM
        s_sprime = rearrange(jnp.concatenate((plan[:, :-1, :], plan[:, 1:, :]), axis=-1), 'B H D -> (B H) D')
        transition_emb = jnp.concatenate((s_sprime[:, :self.config['obs_dim']], s_sprime[:, self.config['obs_dim']:]), axis=1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (transition_emb.shape[0], self.config['action_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['idm_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.idm_state.apply_fn({"params": self.idm_state.params}, transition_emb, noisy_action, k)
            noisy_action = self.idm_noise_scheduler.step(self.idm_noise_state, noise_pred, k, noisy_action, s_rng).prev_sample

            return noisy_action, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))
        action = rearrange(noisy_action, '(B H) D -> B H D', B=B)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action

    def sample(self, batch, eval_rng):
        return self.sample_viz(batch, eval_rng)

    def sample_viz(self, batch, eval_rng):
        eval_rng = jnp.asarray(np.asarray(jax.device_get(eval_rng)).reshape(-1, 2)[0], dtype=jnp.uint32)
        if 'actions' in batch.keys():
            batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert "obs" in batch.keys()
            goal_obs = batch.get("goal_obs", None)
            obs = batch["obs"]
            raw_rgb_obs = [k for k in obs if f"latent_{k}" in self.config["rgb_obs"]]
            if raw_rgb_obs:
                norm_obs = {
                    k: v for k, v in obs.items()
                    if k in self.obs_normalization["obs"]
                }
                if norm_obs:
                    norm_obs = normalize_obs(norm_obs, self.obs_normalization["obs"])
                for k in raw_rgb_obs:
                    norm_obs[k] = obs[k]
                batch = {"obs": norm_obs}
            else:
                batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)
            if goal_obs is not None:
                batch["goal_obs"] = normalize_obs(goal_obs, self.obs_normalization["obs"])

        batch['obs'] = self.vae_encode(batch['obs'])
        action, metrics = self.sample_viz_step(batch, eval_rng, self.config['obs_horizon'])

        # from training batch. not inference
        if metrics['obs_emb'].shape[1] > self.config['obs_horizon']:
            metrics['plan_mse'] = jnp.mean((metrics['noisy_next_obs'] - metrics['obs_emb'][:, self.config['obs_horizon']:, :]) ** 2)
        metrics.pop('obs_emb')
        metrics.pop('noisy_next_obs')
        return action, metrics

    @partial(jax.jit, static_argnames=('obs_horizon'))
    def sample_viz_step(self, batch, eval_rng, obs_horizon):
        # Planner
        for k, v in batch['obs'].items():
            B = v.shape[0]
            break

        obs_emb = self.get_obs_cond(batch['obs'])
        goal_img_cond = self.get_goal_cond(batch)
        obs_cond = obs_emb[:, :obs_horizon, ...].reshape(obs_emb.shape[0], -1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_next_obs = jax.random.normal(noise_rng, (B, self.config['pred_horizon'], self.config['obs_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['planner_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_next_obs, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.planner_state.apply_fn({"params": self.planner_state.params}, noisy_next_obs, k, obs_cond, goal_img_cond=goal_img_cond)
            noisy_next_obs = self.planner_noise_scheduler.step(self.planner_noise_state, noise_pred, k, noisy_next_obs, s_rng).prev_sample

            return noisy_next_obs, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_next_obs, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_next_obs, s_rng))

        start = 0
        end = start + self.config['action_horizon']
        plan = noisy_next_obs[:, start:end, :] # (B, T, D)
        start_state = obs_emb[:, obs_horizon-1:obs_horizon, :] # during inference, only pass obs_horizon imgs

        # Visualization split: start frame from full untruncated latent (so
        # the left-most panel of the video is a clean reconstruction); the
        # predicted action_horizon frames go through the truncated decode
        # path because the planner only emits vae_feature_dim per cam.
        rgb_viz_key = self.config['rgb_obs'][0]
        start_full = batch['obs'][rgb_viz_key][:, obs_horizon - 1:obs_horizon]
        plan_viz_start = self.vae_decode_full(start_full)
        plan_viz_pred = self.vae_decode(plan)
        plan_viz = jnp.concatenate((plan_viz_start, plan_viz_pred), axis=1)

        # Combined latent sequence kept for the IDM (uses truncated dims).
        plan = jnp.concatenate((start_state, plan), axis=1)

        # IDM
        s_sprime = rearrange(jnp.concatenate((plan[:, :-1, :], plan[:, 1:, :]), axis=-1), 'B H D -> (B H) D')
        transition_emb = jnp.concatenate((s_sprime[:, :self.config['obs_dim']], s_sprime[:, self.config['obs_dim']:]), axis=1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (transition_emb.shape[0], self.config['action_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['idm_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.idm_state.apply_fn({"params": self.idm_state.params}, transition_emb, noisy_action, k)
            noisy_action = self.idm_noise_scheduler.step(self.idm_noise_state, noise_pred, k, noisy_action, s_rng).prev_sample

            return noisy_action, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))
        action = rearrange(noisy_action, '(B H) D -> B H D', B=B)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action, dict(plan_viz=plan_viz, noisy_next_obs=noisy_next_obs, obs_emb=obs_emb, plan=plan)

    def get_params(self):
        params = dict()
        if self.use_planner:
            params['planner_params'] = self.planner_state.params
        if self.use_idm:
            params['idm_params'] = self.idm_state.params
        return params

    @classmethod
    def create(
        cls, rng, batch, shape_meta,
        # Hydra Config
        name, planner, idm_net, preprocess_time, cond_encoder, 
        vae_pretrain_path, vae_feature_dim,
        use_planner, use_idm,
        lowdim_obs, rgb_obs, goal_rgb_obs, use_goal_cond, obs_normalization, data_name,
        obs_horizon, pred_horizon, action_horizon, 
        planner_n_diffusion_steps, idm_n_diffusion_steps,
        alpha_planner, alpha_idm,
        lr, end_lr, idm_lr, idm_end_lr, 
        warmup_steps, decay_steps,
        update_planner_every, update_idm_every, update_idm_after,
        update_planner_until, update_planner_after,
        grad_clip,
    ):
        # process data info
        lowdim_obs_dim = 0
        for key in lowdim_obs:
            lowdim_obs_dim += int(np.prod(shape_meta['all_shapes'][key]))
        resnet_feature_dim = vae_feature_dim
        vision_feature_dim = resnet_feature_dim * len(rgb_obs) # ResNet18 has output dim of 512
        goal_rgb_obs = list(goal_rgb_obs)
        goal_dim = 0
        if use_goal_cond:
            for key in goal_rgb_obs:
                if key.startswith("latent_"):
                    goal_dim += vae_feature_dim
                else:
                    goal_dim += int(np.prod(shape_meta['all_shapes'][key]))
        obs_dim = lowdim_obs_dim + vision_feature_dim
        action_dim = shape_meta['ac_dim']

        # load_vae
        if "ckpt" in vae_pretrain_path:
            model_cfg_path = Path(vae_pretrain_path) / '../../.hydra/config.yaml'
            with open(model_cfg_path, 'r') as f:
                model_cfg_path = OmegaConf.create(yaml.safe_load(f))
            vae_module = hydra.utils.instantiate(model_cfg_path.model.vae)
            target_params = vae_module.init(
                rng,
                jnp.zeros((2, 3, 64, 64)),
            )['params']
            target = {'vae_params': target_params}
            restore_args = orbax_utils.restore_args_from_target(target)
            ckpter = orbax.checkpoint.PyTreeCheckpointer()
            raw_restored = ckpter.restore(
                vae_pretrain_path,
                item=target,
                restore_args=restore_args,
                transforms={},
                transforms_default_to_original=True,
            )
            vae_params = raw_restored['vae_params']
        else:
            vae_module, vae_params = FlaxAutoencoderKL.from_pretrained(vae_pretrain_path)
            vae_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices('gpu')[0]), vae_params)
        print(f"vae number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(vae_params)):e}")

        # get init batch
        rng, init_enc_rng = jax.random.split(rng)
        init_time = jnp.zeros((1,), dtype=jnp.int32)
        init_action = batch['actions']

        # create encoder
        obs_emb = LDPAgent._get_obs_cond(batch['obs'], rgb_obs, lowdim_obs, obs_horizon, init_enc_rng, vae_feature_dim, data_name)
        goal_img_cond = None
        if use_goal_cond:
            if "goal_obs" not in batch:
                raise ValueError("use_goal_cond=True requires batch['goal_obs']")
            goal_img_cond = LDPAgent._get_goal_cond(batch['goal_obs'], goal_rgb_obs, vae_feature_dim, data_name)

        # create planner
        if use_planner:
            rng, init_rng = jax.random.split(rng)
            with open_dict(planner):
                planner.input_dim = obs_dim # important! model obs not action
                planner.global_cond_dim = obs_dim + goal_dim
                planner._convert_ = 'all'
            planner = hydra.utils.instantiate(planner)
            obs_cond = obs_emb[:, :obs_horizon, ...]
            obs_cond = obs_cond.reshape(obs_emb.shape[0], -1)
            planner_params = planner.init(init_rng, obs_emb[:, obs_horizon:], init_time, obs_cond, goal_img_cond=goal_img_cond)["params"]
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(planner_params))
            print(f"planner number of parameters: {param_count:e}")

            # create train state
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=end_lr,
                peak_value=lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
            tx = optax.adam(lr_schedule)
            planner_state = flax_utils.TrainStateEMA.create(
                apply_fn=planner.apply,
                params=planner_params,
                tx=tx,
            )
        else:
            planner_state = None

        # create idm
        if use_idm:
            rng, init_idm_rng = jax.random.split(rng)
            with open_dict(idm_net):
                idm_net.out_dim = action_dim
                idm_net._convert_ = 'all'
            idm_net_cls = lambda: hydra.utils.instantiate(idm_net)
            preprocess_time_cls = lambda: hydra.utils.instantiate(preprocess_time)
            cond_encoder_cls = lambda: hydra.utils.instantiate(cond_encoder)

            idm = MLPDiffusion(cond_encoder_cls, idm_net_cls, preprocess_time_cls)

            # maybe should turn this into a separate jit util function
            # it's possible that with obs_horizon, action / s_sprime could be off??
            s_sprime = rearrange(jnp.concatenate((obs_emb[:, obs_horizon-1:-1, :], obs_emb[:, obs_horizon:, :]), axis=-1), 'B H D -> (B H) D')
            transition_emb = jnp.concatenate((s_sprime[:, :obs_dim], s_sprime[:, obs_dim:]), axis=1)
            init_action = init_action[:, :-1, :]
            init_action = init_action.reshape(-1, *init_action.shape[2:])
            idm_params = idm.init(init_idm_rng, transition_emb, init_action, init_time)["params"]
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(idm_params))
            print(f"IDM number of parameters: {param_count:e}")
            
            # create train state
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=idm_end_lr,
                peak_value=idm_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=idm_end_lr,
            )
            tx = optax.adam(lr_schedule)
            idm_state = flax_utils.TrainStateEMA.create(
                apply_fn=idm.apply,
                params=idm_params,
                tx=tx,
            )
        else:
            idm_state = None

        
        # create noise scheduler
        planner_noise_scheduler = FlaxDDPMScheduler(
                            num_train_timesteps=planner_n_diffusion_steps,
                            beta_schedule='squaredcos_cap_v2',
                            clip_sample=True,
                            prediction_type='epsilon'
                            )
        planner_noise_state = planner_noise_scheduler.create_state()
        idm_noise_scheduler = FlaxDDPMScheduler(
                            num_train_timesteps=idm_n_diffusion_steps,
                            beta_schedule='squaredcos_cap_v2',
                            clip_sample=True,
                            prediction_type='epsilon'
                            )
        idm_noise_state = idm_noise_scheduler.create_state()

        # create config with additional variables
        config = flax.core.FrozenDict(dict(
                    planner_n_diffusion_steps=planner_n_diffusion_steps,
                    idm_n_diffusion_steps=idm_n_diffusion_steps,
                    lowdim_obs=lowdim_obs, rgb_obs=rgb_obs, obs_horizon=obs_horizon,
                    name=name, action_dim=shape_meta['ac_dim'],
                    pred_horizon=pred_horizon, action_horizon=action_horizon,
                    obs_dim=obs_dim, goal_dim=goal_dim,
                    use_goal_cond=use_goal_cond, goal_rgb_obs=goal_rgb_obs,
                    update_planner_every=update_planner_every, update_idm_every=update_idm_every, 
                    update_planner_until=update_planner_until,
                    update_planner_after=update_planner_after, 
                    update_idm_after=update_idm_after, 
                    vae_feature_dim=vae_feature_dim, data_name=data_name
                    ))
        obs_normalization = flax_utils.cfg_to_jnp(obs_normalization)

        return cls(planner_state, idm_state, vae_module, vae_params,
                 obs_normalization, use_planner, use_idm, 
                 alpha_planner, alpha_idm, lr_schedule,
                 planner_noise_scheduler, planner_noise_state, idm_noise_scheduler, idm_noise_state, 
                 config)
