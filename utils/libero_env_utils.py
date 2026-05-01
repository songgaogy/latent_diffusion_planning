"""Multi-process eval orchestrator for the LIBERO benchmark.

Mirrors utils/rm_env_utils.run_robomimic_eval's queue-based pattern, but each
worker owns a LiberoRemoteEnv (a subprocess in the libero conda env) instead
of a local RobosuiteEnv.

Workers (the EvalProc analogs) themselves run as torch.multiprocessing.Process
in the ldp env; they do not import jax/flax. Inference happens centrally on
the parent process using `policy.sample_viz` / `policy.sample`.
"""
from __future__ import annotations

import os
import time
import traceback
from collections import deque
from pathlib import Path
from typing import List

import h5py
import numpy as np
import psutil
import torch.multiprocessing as mp

import jax
from data.libero_data import _resolve_paths


def _single_prng_key(key):
    return jax.numpy.asarray(np.asarray(jax.device_get(key)).reshape(-1, 2)[0], dtype=jax.numpy.uint32)


def _sorted_demo_keys(h5_group):
    keys = list(h5_group.keys())
    keys.sort(key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
    return keys


def _build_goal_bank(env_params):
    latent_path = env_params.get("goal_latent_path")
    goal_demo_paths = env_params.get("goal_demo_paths")
    goal_rgb_obs = list(env_params.get("goal_rgb_obs", []))
    if not latent_path or not goal_demo_paths or not goal_rgb_obs:
        return None

    demo_paths = _resolve_paths(goal_demo_paths)
    if not demo_paths:
        raise ValueError(f"goal_demo_paths resolved to empty: {goal_demo_paths}")

    goal_bank = {}
    with h5py.File(os.path.expanduser(latent_path), "r") as latent_f:
        latent_data = latent_f["data"]
        for task_id, path in enumerate(demo_paths):
            goals = []
            with h5py.File(path, "r", swmr=True, libver="latest") as raw_f:
                for demo_key in _sorted_demo_keys(raw_f["data"]):
                    demo_id = f"{Path(path).stem}__{demo_key}"
                    if demo_id not in latent_data:
                        continue
                    goal = {}
                    for key in goal_rgb_obs:
                        latent_key = key[len("latent_"):] if key.startswith("latent_") else key
                        goal[key] = np.asarray(latent_data[demo_id]["latent"][latent_key][-1], dtype=np.float32)
                    goals.append(goal)
            if goals:
                goal_bank[task_id] = goals
    if not goal_bank:
        raise ValueError(f"No goal latents found in {latent_path}")
    return goal_bank


def _select_goal_obs(goal_bank, task_ids, seeds):
    goal_obs = {}
    for task_id, seed in zip(task_ids, seeds):
        goals = goal_bank[int(task_id)]
        goal = goals[int(np.random.default_rng(int(seed)).integers(len(goals)))]
        for key, value in goal.items():
            goal_obs.setdefault(key, []).append(value)
    return {key: np.asarray(value, dtype=np.float32) for key, value in goal_obs.items()}


class LiberoEvalProc:
    """Per-rollout worker: spawns one LiberoRemoteEnv, runs episodes for a list
    of seeds, talks to the parent via two queues."""

    def __init__(
        self,
        seeds: List[int],
        process_id: int,
        env_kwargs: dict,
        obs_horizon: int,
        rgb_viz: str,
        terminal_queue: mp.Queue,
        mp_context=None,
    ):
        self.seeds = seeds
        self.process_id = process_id
        self.env_kwargs = dict(env_kwargs)
        self.obs_horizon = obs_horizon
        self.rgb_viz = rgb_viz
        self.terminal_queue = terminal_queue
        mp_context = mp_context or mp
        self.send_queue = mp_context.Queue()
        self.recv_queue = mp_context.Queue()

    def start(self):
        try:
            from envs.libero_remote_env import LiberoRemoteEnv

            # accept either a single task_id or a list of task_ids; cycle through
            task_ids = self.env_kwargs.get("task_ids")
            if task_ids is None:
                task_ids = [self.env_kwargs.get("task_id", 0)]
            task_ids = list(task_ids)

            results = {}
            for s_idx, seed in enumerate(self.seeds):
                # rotate task selection across seeds so a worker covers
                # multiple tasks rather than spawning one env per task
                task_id = int(task_ids[s_idx % len(task_ids)])
                # one env per (worker, task). Cache the env across consecutive
                # rollouts on the same task to avoid subprocess startup cost.
                if (
                    not hasattr(self, "_env")
                    or self._env_task_id != task_id
                    or self._env.task_id != task_id
                ):
                    if hasattr(self, "_env"):
                        try:
                            self._env.close()
                        except Exception:
                            pass
                    env_kwargs = {
                        k: v
                        for k, v in self.env_kwargs.items()
                        if k not in ("task_id", "task_ids", "lowdim_obs", "rgb_obs")
                    }
                    env_kwargs["task_id"] = task_id
                    env_kwargs["worker_id"] = self.process_id
                    self._env = LiberoRemoteEnv(**env_kwargs)
                    self._env_task_id = task_id

                env = self._env
                np.random.seed(seed)

                # use the benchmark's per-task init states, indexed by seed
                init_idx = seed % max(1, env.n_init_states)
                ob_dict = env.set_init_state(init_idx)

                obs_deque = deque([ob_dict] * self.obs_horizon, maxlen=self.obs_horizon)
                debug_obs = []
                env_steps = 0
                total_reward = 0.0
                success = False

                while True:
                    self.send_queue.put((self.process_id, obs_deque, task_id, seed))
                    out = self.recv_queue.get()
                    if len(out) == 1:
                        action = out[0]
                        plan_viz = None
                    else:
                        action, plan_viz = out

                    rew = 0.0
                    done = False
                    for idx, ac in enumerate(action):
                        ob_dict, r_tmp, done, info = env.step(np.asarray(ac))
                        env_steps += 1
                        obs_deque.append(ob_dict)
                        if plan_viz is not None:
                            gt = ob_dict[self.rgb_viz].transpose((2, 0, 1))
                            debug_obs.append(np.concatenate([gt, plan_viz[idx]], axis=-1))
                        else:
                            debug_obs.append(ob_dict[self.rgb_viz])
                        rew += float(r_tmp)
                        success = success or bool(info.get("success", False))
                        if done or success:
                            break

                    total_reward += rew
                    if done or success:
                        # signal main loop that this worker is mid-rollout->done
                        self.send_queue.put((self.process_id, [dict(reset=True)]))
                        break

                results[seed] = dict(
                    success=float(success),
                    reward=total_reward,
                    horizon=env_steps,
                    task_id=task_id,
                    debug_obs=debug_obs,
                )

            try:
                self._env.close()
            except Exception:
                pass
            self.terminal_queue.put((self.process_id, results))
        except Exception:
            self.terminal_queue.put((self.process_id, "error", traceback.format_exc()))


def run_libero_eval(
    env_params: dict,
    policy,
    policy_name: str,
    n_rollout: int,
    n_proc: int,
    seed: int,
    eval_rng,
    verbose: bool = True,
):
    """Drop-in replacement for run_robomimic_eval, dispatched on the libero
    branch from train_bc.py / eval_bc.py."""
    assert n_rollout % n_proc == 0, f"n_rollout={n_rollout} not divisible by n_proc={n_proc}"
    rollouts_per_proc = n_rollout // n_proc
    eval_rng = _single_prng_key(eval_rng)

    obs_horizon = env_params["obs_horizon"]
    rgb_viz = env_params["rgb_viz"]
    env_kwargs = dict(env_params["env_kwargs"])
    rgb_obs = list(env_kwargs["rgb_obs"])
    lowdim_obs = list(env_kwargs.get("lowdim_obs", []))
    use_goal_cond = bool(getattr(policy, "config", {}).get("use_goal_cond", False))
    goal_bank = _build_goal_bank(env_params) if use_goal_cond else None

    mp_context = mp.get_context("spawn")
    terminal_queue = mp_context.Queue()
    procs = []
    for i in range(n_proc):
        seeds = list(range(seed + i * rollouts_per_proc, seed + (i + 1) * rollouts_per_proc))
        procs.append(
            LiberoEvalProc(
                seeds=seeds,
                process_id=i,
                env_kwargs=env_kwargs,
                obs_horizon=obs_horizon,
                rgb_viz=rgb_viz,
                terminal_queue=terminal_queue,
                mp_context=mp_context,
            )
        )

    put_queues = {i: p.recv_queue for i, p in enumerate(procs)}
    get_queues = {i: p.send_queue for i, p in enumerate(procs)}

    processes = {i: mp_context.Process(target=p.start) for i, p in enumerate(procs)}
    for _, pp in processes.items():
        pp.start()

    t0 = time.time()
    results = dict()

    while len(processes) > 0:
        # drain finished workers
        while not terminal_queue.empty():
            out = terminal_queue.get()
            if len(out) == 3:
                _, _, tb = out
                print(f"libero worker died:\n{tb}")
                # tear down siblings
                for k, pp in processes.items():
                    pp.terminate()
                raise RuntimeError("libero eval worker crashed")
            term_idx, proc_results = out
            results.update(proc_results)
            processes[term_idx].join()
            processes.pop(term_idx)
            get_queues.pop(term_idx)
            put_queues.pop(term_idx)

        # batch obs requests across workers, run a single jax inference
        idxs = []
        task_ids = []
        rollout_seeds = []
        obs_dict = {}
        for _, q in get_queues.items():
            if q.empty():
                continue
            data = q.get()
            obs_deque = data[1]
            if isinstance(obs_deque, list) and len(obs_deque) and isinstance(obs_deque[0], dict) and obs_deque[0].get("reset", False):
                continue
            task_ids.append(data[2] if len(data) > 2 else 0)
            rollout_seeds.append(data[3] if len(data) > 3 else seed)
            for k in obs_deque[0].keys():
                v = np.stack([x[k] for x in obs_deque])
                obs_dict.setdefault(k, []).append(v)
            idxs.append(data[0])

        if not idxs:
            time.sleep(0.001)
            continue

        # stack across the worker batch dim
        for k in list(obs_dict.keys()):
            obs_dict[k] = np.array(obs_dict[k], dtype=np.float32)

        # mirror rm_env_utils: agent reads dataset-namespaced rgb keys (strip
        # `latent_` prefix on the env side; the agent will re-encode raw rgb
        # via vae_encode if it expects latents).
        agent_rgb = [k.replace("latent_", "") if k.startswith("latent_") else k for k in rgb_obs]
        agent_obs_dims = agent_rgb + lowdim_obs
        if "optimal" in agent_obs_dims:
            sample = obs_dict[lowdim_obs[0]]
            obs_dict["optimal"] = np.ones((sample.shape[0], 1, 1), dtype=sample.dtype)
        # subset
        missing = [k for k in agent_obs_dims if k not in obs_dict]
        if missing:
            raise RuntimeError(
                f"env produced obs missing keys {missing}; got {sorted(obs_dict)}"
            )
        obs_dict = {k: obs_dict[k] for k in agent_obs_dims}
        batch = dict(obs=obs_dict)
        if goal_bank is not None:
            missing_goal_tasks = sorted(set(task_ids) - set(goal_bank))
            if missing_goal_tasks:
                raise RuntimeError(f"goal bank missing task ids {missing_goal_tasks}")
            batch["goal_obs"] = _select_goal_obs(goal_bank, task_ids, rollout_seeds)

        eval_rng = _single_prng_key(eval_rng)
        s_rng, eval_rng = jax.random.split(eval_rng)
        s_rng = _single_prng_key(s_rng)
        visualize_plan = policy.config["name"] in ("ldp_agent", "ldp_hier_agent")
        if visualize_plan:
            batch_action, plan_dict = policy.sample_viz(batch, s_rng)
            plan_viz = plan_dict["plan_viz"]
            plan_viz = (np.clip((np.array(plan_viz) + 1) / 2, 0, 1) * 255).astype(np.uint8)
            batch_action = np.array(jax.device_get(batch_action))
            for j, idx in enumerate(idxs):
                put_queues[idx].put((batch_action[j], plan_viz[j]))
        else:
            batch_action, _ = policy.sample(batch, s_rng)
            batch_action = np.array(jax.device_get(batch_action))
            for j, idx in enumerate(idxs):
                put_queues[idx].put((batch_action[j],))

    # aggregate
    rollout_logs = {}
    videos = []
    per_task = {}
    for r in results.values():
        for k, v in r.items():
            if k == "debug_obs":
                videos.append(v)
            elif k == "task_id":
                continue
            else:
                rollout_logs.setdefault(k, []).append(v)
        per_task.setdefault(r["task_id"], []).append(r["success"])

    rollout_logs = {k: float(np.mean(v)) for k, v in rollout_logs.items()}
    for tid, succs in per_task.items():
        rollout_logs[f"success_task_{tid}"] = float(np.mean(succs))
    rollout_logs["total_time"] = time.time() - t0
    rollout_logs["RAM_GB"] = float(psutil.Process(os.getpid()).memory_info().rss / 1e9)
    if verbose:
        print(f"[run_libero_eval] {n_rollout} rollouts in {rollout_logs['total_time']:.1f}s")
        print(rollout_logs)
    return rollout_logs, videos
