"""Smoke test for the multi-camera obs-cond path in agent/ldp_agent.py.

Without spinning up the full agent + VAE, we just call ``LDPAgent._get_obs_cond``
(a classmethod) on a synthetic batch with two latent cameras and assert the
returned shape preserves temporal alignment.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.ldp_agent import LDPAgent  # noqa: E402
from networks.diffusion_nets_v2 import ConditionalUnet1D  # noqa: E402


def test_two_camera_latent_shapes_rank3():
    B, H = 4, 3
    latent_dim = 16
    lowdim_total = 6 + 2  # ee_states + gripper_states
    batch = {
        "latent_agentview_rgb": jnp.zeros((B, H, latent_dim), dtype=jnp.float32),
        "latent_eye_in_hand_rgb": jnp.zeros((B, H, latent_dim), dtype=jnp.float32),
        "ee_states": jnp.zeros((B, H, 6), dtype=jnp.float32),
        "gripper_states": jnp.zeros((B, H, 2), dtype=jnp.float32),
    }
    out = LDPAgent._get_obs_cond(
        batch=batch,
        rgb_obs=["latent_agentview_rgb", "latent_eye_in_hand_rgb"],
        lowdim_obs=["ee_states", "gripper_states"],
        obs_horizon=H,
    )
    assert out.shape == (B, H, 2 * latent_dim + lowdim_total), out.shape


def test_two_camera_latent_shapes_rank5():
    """Pre-encoded latents from disk: shape (B, H, latent_H, latent_W, latent_C)."""
    B, H = 4, 3
    latent_H, latent_W, latent_C = 2, 2, 4
    flat = latent_H * latent_W * latent_C
    batch = {
        "latent_agentview_rgb": jnp.zeros(
            (B, H, latent_H, latent_W, latent_C), dtype=jnp.float32
        ),
        "latent_eye_in_hand_rgb": jnp.zeros(
            (B, H, latent_H, latent_W, latent_C), dtype=jnp.float32
        ),
        "ee_states": jnp.zeros((B, H, 6), dtype=jnp.float32),
        "gripper_states": jnp.zeros((B, H, 2), dtype=jnp.float32),
    }
    out = LDPAgent._get_obs_cond(
        batch=batch,
        rgb_obs=["latent_agentview_rgb", "latent_eye_in_hand_rgb"],
        lowdim_obs=["ee_states", "gripper_states"],
        obs_horizon=H,
    )
    assert out.shape == (B, H, 2 * flat + 8), out.shape


def test_temporal_alignment_preserved():
    """Camera-1 frame t should land in the same H slot as camera-2 frame t.

    Encode this by putting distinct sentinel values per (camera, time) and
    checking the output slot for time t contains BOTH cam1[t] and cam2[t]
    sentinels (and only those).
    """
    B, H, D = 1, 4, 5
    lowdim_total = 3
    cam1 = np.tile(np.arange(H)[None, :, None] * 1.0, (B, 1, D)).astype(np.float32)
    cam2 = np.tile(np.arange(H)[None, :, None] * 1.0 + 100.0, (B, 1, D)).astype(np.float32)
    lowdim = np.zeros((B, H, lowdim_total), dtype=np.float32)
    batch = {
        "latent_agentview_rgb": jnp.asarray(cam1),
        "latent_eye_in_hand_rgb": jnp.asarray(cam2),
        "ee_states": jnp.asarray(lowdim[:, :, :2]),
        "gripper_states": jnp.asarray(lowdim[:, :, 2:]),
    }
    out = np.asarray(
        LDPAgent._get_obs_cond(
            batch=batch,
            rgb_obs=["latent_agentview_rgb", "latent_eye_in_hand_rgb"],
            lowdim_obs=["ee_states", "gripper_states"],
            obs_horizon=H,
        )
    )
    # Per H slot t: first D entries == t (cam1), next D == t + 100 (cam2)
    for t in range(H):
        cam1_slice = out[0, t, :D]
        cam2_slice = out[0, t, D : 2 * D]
        assert np.allclose(cam1_slice, t), f"t={t}: cam1 slice {cam1_slice}"
        assert np.allclose(cam2_slice, t + 100), f"t={t}: cam2 slice {cam2_slice}"


def test_ldp_agent_uses_direct_latent_flattening():
    # cheap sanity: the LIBERO spatial projection path was removed.
    src = (REPO_ROOT / "agent" / "ldp_agent.py").read_text()
    assert "_spatial_project_latent" not in src
    assert "vae_feature_dim'] == 16" in src


def test_goal_cond_two_camera_shapes():
    B = 4
    D = 16
    goal_obs = {
        "latent_agentview_rgb": jnp.zeros((B, 2, 2, 4), dtype=jnp.float32),
        "latent_eye_in_hand_rgb": jnp.zeros((B, D), dtype=jnp.float32),
    }
    out = LDPAgent._get_goal_cond(
        goal_obs,
        ["latent_agentview_rgb", "latent_eye_in_hand_rgb"],
    )
    assert out.shape == (B, 2 * D), out.shape


def test_conditional_unet_goal_cond_forward():
    B, T, D = 2, 4, 16
    model = ConditionalUnet1D(
        input_dim=D,
        global_cond_dim=24,
        down_dims=(32, 64),
        downsample=True,
    )
    sample = jnp.zeros((B, T, D), dtype=jnp.float32)
    timestep = jnp.zeros((B,), dtype=jnp.int32)
    global_cond = jnp.zeros((B, 8), dtype=jnp.float32)
    goal_img_cond = jnp.zeros((B, 16), dtype=jnp.float32)
    params = model.init(
        jnp.asarray([0, 1], dtype=jnp.uint32),
        sample,
        timestep,
        global_cond,
        goal_img_cond=goal_img_cond,
    )
    out = model.apply(params, sample, timestep, global_cond, goal_img_cond=goal_img_cond)
    assert out.shape == sample.shape, out.shape


if __name__ == "__main__":
    fns = [
        test_two_camera_latent_shapes_rank3,
        test_two_camera_latent_shapes_rank5,
        test_temporal_alignment_preserved,
        test_ldp_agent_uses_direct_latent_flattening,
        test_goal_cond_two_camera_shapes,
        test_conditional_unet_goal_cond_forward,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK  {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback

            traceback.print_exc()
            print(f" FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(0 if failed == 0 else 1)
