"""Visualize LDP planner outputs on training data.

This script loads a saved LDP experiment, restores a checkpoint, samples the
planner on deterministic training windows, decodes GT and predicted latents
with the VAE, and saves paired comparison images.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hydra
import jax
import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode GT and planner latent outputs for an LDP checkpoint."
    )
    parser.add_argument(
        "--experiment-dir",
        default="experiments/libero_long/ldp_goal_cond_v3",
        help="Experiment directory containing .hydra/config.yaml and ckpt/.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="'latest', a numeric step such as 500000, or a checkpoint path.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_checkpoint(experiment_dir: Path, checkpoint: str) -> tuple[Path, str]:
    if checkpoint == "latest":
        ckpts = sorted(
            experiment_dir.glob("ckpt/*.ckpt"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found under {experiment_dir / 'ckpt'}")
        ckpt_path = ckpts[-1]
    elif checkpoint.isdigit():
        ckpt_path = experiment_dir / "ckpt" / f"{checkpoint}.ckpt"
    else:
        ckpt_path = Path(checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (REPO_ROOT / ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    return ckpt_path, ckpt_path.stem


def load_experiment_cfg(experiment_dir: Path, batch_size: int):
    cfg_path = experiment_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        cfg.batch_size = batch_size
        cfg.n_workers = 0
        cfg.use_wandb = False
        cfg.use_tb = False
        cfg.data.batch_size = batch_size
        cfg.data.n_workers = 0
        cfg.data.cache_all_in_ram = False
    OmegaConf.resolve(cfg)
    return cfg


def stack_numpy_tree(samples):
    elem = samples[0]
    if isinstance(elem, dict):
        return {key: stack_numpy_tree([sample[key] for sample in samples]) for key in elem}
    return np.ascontiguousarray(np.stack(samples, axis=0))


def load_agent(cfg, data, init_batch, ckpt_path: Path):
    agent_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent, resolve=True))
    agent_target = agent_cfg.pop("_target_")
    agent_class = hydra.utils.get_class(agent_target)
    rng = jax.random.PRNGKey(int(cfg.seed))
    agent = agent_class.create(rng, init_batch, data.shape_meta, **agent_cfg)

    target = {}
    if agent.use_planner:
        target["planner_params"] = agent.planner_state.params
    if agent.use_idm:
        target["idm_params"] = agent.idm_state.params
    if not target:
        raise RuntimeError("Checkpoint restore target is empty; planner/IDM are disabled.")

    ckpter = ocp.PyTreeCheckpointer()
    restore_args = orbax_utils.restore_args_from_target(target)
    restored = ckpter.restore(
        ckpt_path,
        item=target,
        restore_args=restore_args,
        transforms={},
        transforms_default_to_original=True,
    )
    if "planner_params" in restored:
        agent = agent.replace(
            planner_state=agent.planner_state.replace(
                params=restored["planner_params"],
                ema_params=restored["planner_params"],
            )
        )
    if "idm_params" in restored:
        agent = agent.replace(
            idm_state=agent.idm_state.replace(
                params=restored["idm_params"],
                ema_params=restored["idm_params"],
            )
        )
    return agent


def to_uint8_nhwc(decoded):
    arr = np.array(jax.device_get(decoded))
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 5 and arr.shape[2] == 3:
        arr = arr.transpose(0, 1, 3, 4, 2)
    return arr


def make_pair_image(gt_frames, plan_frames, title: str):
    if gt_frames.shape != plan_frames.shape:
        raise ValueError(f"GT/planner frame shape mismatch: {gt_frames.shape} vs {plan_frames.shape}")

    n_frames, height, width, channels = gt_frames.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames, got shape {gt_frames.shape}")

    label_h = 24
    title_h = 28
    canvas = np.full((title_h + 2 * (label_h + height), n_frames * width, 3), 255, dtype=np.uint8)
    canvas[title_h + label_h : title_h + label_h + height] = np.concatenate(list(gt_frames), axis=1)
    y2 = title_h + 2 * label_h + height
    canvas[y2 : y2 + height] = np.concatenate(list(plan_frames), axis=1)

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), title, fill=(0, 0, 0))
    for idx in range(n_frames):
        label = "t0" if idx == 0 else f"t+{idx}"
        draw.text((idx * width + 8, title_h + 4), f"gt {label}", fill=(0, 0, 0))
        draw.text((idx * width + 8, y2 - label_h + 4), f"planner {label}", fill=(0, 0, 0))
    return image


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    os.chdir(REPO_ROOT)
    experiment_dir = Path(args.experiment_dir).expanduser()
    if not experiment_dir.is_absolute():
        experiment_dir = (REPO_ROOT / experiment_dir).resolve()

    ckpt_path, ckpt_name = resolve_checkpoint(experiment_dir, args.checkpoint)
    batch_size = args.batch_size or args.num_samples
    cfg = load_experiment_cfg(experiment_dir, batch_size)

    data = hydra.utils.instantiate(cfg.data)
    dataset = data.train_dataset
    max_index = args.start_index + (args.num_samples - 1) * args.stride
    if max_index >= dataset.total_n_sequences:
        raise ValueError(
            f"Requested max index {max_index}, but dataset has {dataset.total_n_sequences} sequences."
        )
    samples = [dataset.get_item(args.start_index + i * args.stride) for i in range(args.num_samples)]
    batch = stack_numpy_tree(samples)

    agent = load_agent(cfg, data, batch, ckpt_path)
    rng = jax.random.PRNGKey(args.seed)
    _, metrics = agent.sample_viz(batch, rng)

    from utils.data_utils import postprocess_batch

    batch_norm = postprocess_batch(batch, agent.obs_normalization)
    obs_horizon = int(agent.config["obs_horizon"])
    action_horizon = int(agent.config["action_horizon"])
    rgb_viz_key = agent.config["rgb_obs"][0]
    gt_latent = batch_norm["obs"][rgb_viz_key][
        :, obs_horizon - 1 : obs_horizon + action_horizon
    ]
    gt_viz = agent.vae_decode_full(gt_latent)

    gt_uint8 = to_uint8_nhwc(gt_viz)
    plan_uint8 = to_uint8_nhwc(metrics["plan_viz"])

    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "planner_viz" / ckpt_name
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_samples):
        sample_index = args.start_index + i * args.stride
        title = f"{experiment_dir.name} ckpt={ckpt_name} sample_index={sample_index}"
        image = make_pair_image(gt_uint8[i], plan_uint8[i], title)
        image.save(output_dir / f"sample_{i:03d}_idx_{sample_index}.png")

    print(f"Saved {args.num_samples} paired planner visualizations to {output_dir}")


if __name__ == "__main__":
    main()
