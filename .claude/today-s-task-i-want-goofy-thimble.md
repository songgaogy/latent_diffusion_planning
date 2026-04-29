# Plan: Latent Diffusion Planning on LIBERO Long

## Context
Apply LDP to the LIBERO benchmark. Train the VAE on **all five LIBERO suites** (libero_spatial + libero_object + libero_goal + libero_10 + libero_90, ~130 task hdf5s, ~5500 demos), then train the diffusion **planner + IDM only on libero_10** (the long-horizon suite, 10 tasks, ~410 demos). Evaluate on the 10 libero_10 tasks via a real LIBERO env.

User's locked decisions:
- VAE input: **256×256** (matches the data already on disk under `/home/dodo/data1/libero/256/`).
- Cameras: **both** `agentview_rgb` + `eye_in_hand_rgb` encoded independently and concatenated as latent obs.
- Eval: **client-server bridge** between `ldp` and `libero` conda envs.
- No mixed/suboptimal data path.

## Key facts (verified by inspection)

### LIBERO data on disk
`/home/dodo/data1/libero/256/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/*demo.hdf5`, symlinked at `data/libero`. 10/10/10/10/90 task files, ~40–45 demos/task. Schema (consistent across suites):
```
data/demo_N/
  obs/  agentview_rgb (T,256,256,3) uint8
        eye_in_hand_rgb (T,256,256,3) uint8
        ee_pos (T,3) f64, ee_ori (T,3) f64 Euler, ee_states (T,6) f64
        gripper_states (T,2) f64, joint_states (T,7) f64
  actions (T,7) f64 in [-1,1]
  rewards, dones, robot_states, states
```
**No `next_obs`, no `object` key, no `env_args` attrs on the `data` group, demo `num_samples` attr presence to be verified at coding time** (fall back to `actions.shape[0]`).

### LDP contracts the libero modules must satisfy
- `cfg.data` returns an object with `train_dataloader()`, `eval_dataloader()`, `train_dataset`, `name`, `shape_meta`, `env_params`, `meta`, `batch_size`. Batch is `{obs:{key:[T,*shape]}, actions:[T,ac_dim]}`. `train_bc.py:170-190` dispatches eval by `data.name` (currently `rm…`/`aloha…`).
- `dataset.env_meta` is consumed only inside the rm/eval/collect dispatch branches and only when `data.name.startswith("rm")` (`train_bc.py:171-173`, `eval_bc.py:170-171`, `train_mixed_bc.py:178-179`, `collect_data.py:80-81`). The libero branch can ignore it; **drop env_meta from the libero loader entirely**.
- `process_sdvae_data.py:67` already hardcodes resize to 256² — no-op for native-256 data. Currently has `run_rm` (uses `next_obs`) and `run_aloha` (no `next_obs`) branches; we add a `run_libero` branch.
- `agent/ldp_agent.py:69-80` only handles `vae_feature_dim ∈ {16,32,36,64}`; needs new cases for 256² inputs (SD-VAE produces 32×32×4 = 4096 floats per camera).

### Bugs in existing code that block multi-camera latents
- **`agent/ldp_agent.py:get_obs_cond` line 93** and **`_get_obs_cond` line 106** concatenate per-camera latents on `axis=1` (the time/horizon axis), then reshape to `(B,H,-1)`. With two latent cameras this fails because the result has shape `(B, 2H, D)` not `(B, H, 2D)`. **Must change `axis` to `-1` on the latent path** (or unify by always concatenating on the feature axis after reshape). Untested in upstream because rm_lift and aloha_cube only ever use one RGB camera.

### LIBERO env contract (server side)
`OffScreenRenderEnv(bddl_file_name, camera_heights=256, camera_widths=256, camera_names=["agentview","robot0_eye_in_hand"], …)`; `reset()`/`step(action)` return obs dict with **runtime keys** `agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object` — different names from the dataset and includes a quaternion not Euler. Init states via `benchmark.get_task_init_states(task_id)` then `env.set_init_state(...)` (do NOT use the demo's first state — biases success).

### Conda compat
`libero` package's `__init__.py` blocks on interactive `input()` if `~/.libero/config.yaml` is missing — confirmed unimportable in `ldp` env. Eval must run via subprocess in the `libero` env.

## Design

### 1. New data modules
- **`data/libero_data.py`** — mirror `data/alohasim_data.py` (no-next_obs path). Differences:
  - Constructor accepts `train_glob` / `eval_glob` (string glob → list of hdf5 paths). Builds a registry `self.demos: list[(file_path, demo_key, length)]` with **globally unique IDs** like `f"{task_stem}__{demo_key}"`. Open hdf5 files lazily, cache by path.
  - `obs_keys = list(meta.lowdim_obs) + list(meta.rgb_obs)`.
  - **Lowdim choice: `[ee_states, gripper_states]`** (8-d total; matches HDF5 schema verbatim; eval-side adapter constructs `ee_states = concat(eef_pos, quat2euler(eef_quat))` once).
  - Expose `iter_demo_obs(rgb_keys) -> generator[(demo_id, {key: ndarray})]` for `process_sdvae_data` to consume without poking `hdf5_file` directly.
  - No `env_meta`. Constructor reads `env_params` from yaml and stores it.
- **`data/libero_latent_data.py`** — mirror `data/robomimic_latent_data.py` but uses the libero registry; reads from a single `latent.hdf5` written by `process_sdvae_data.py` and from raw hdf5s for non-rgb obs.

### 2. Hydra data configs
- **`data/cfg/libero_all/img.yaml`** — VAE training. Glob: `data/libero/*/*demo.hdf5` (5 suites). `meta.rgb_obs:[agentview_rgb, eye_in_hand_rgb]`, `meta.lowdim_obs:[ee_states, gripper_states]`. `shape_meta.all_shapes` declares 256² imgs and lowdim shapes plus a `optimal:[1]` entry copied from rm_lift convention (defensive). `env_params` empty (no env eval during VAE).
- **`data/cfg/libero_long/img.yaml`** — planner/IDM raw-image training. Glob: `data/libero/libero_10/*demo.hdf5`. Same shape_meta. `env_params.env_kwargs`: `task_suite=libero_10`, `task_ids=range(10)`, `camera_heights=256`, `camera_widths=256`, `camera_names=[agentview, robot0_eye_in_hand]`, `controller=OSC_POSE`, `horizon=600`. `env_params.rgb_viz=agentview_rgb`.
- **`data/cfg/libero_long/latent_img.yaml`** — `meta.rgb_obs:[latent_agentview_rgb, latent_eye_in_hand_rgb]`. Adds `train_latent_path: ${...}/latent.hdf5`. `obs_normalization.obs.latent_*.{min,max}` to be filled in **after** running `process_sdvae_data.py` (a tiny stats helper reads `data.attrs[min_z, max_z]` from latent.hdf5 — do NOT inherit rm_lift's hardcoded `[-10, 10]`).

### 3. Top-level training configs
- **`train_vae_libero.yaml`** — defaults `data: cfg/libero_all/img`, `model: stable_vae_model`, `horizon=1`, batch/lr matching `train_vae.yaml`.
- **`train_libero_long.yaml`** — defaults `agent: ldp_agent`, `data: cfg/libero_long/latent_img`. `horizon=9, action_horizon=4` (mirroring rm_lift). `agent.vae_feature_dim=256` per camera (8×8×4 spatial slice; total cond ≈ 2×256 + 8 lowdim ≈ 520 — well within planner U-Net `down_dims:[256,512,1024]`). Add `agent.vae_pretrain_path` flag.
- **`eval_libero_long.yaml`** — extends `train_libero_long.yaml` like `eval_bc.yaml` extends `train_bc.yaml`.

### 4. Agent changes (`agent/ldp_agent.py`)
- **Fix multi-camera latent concat** at `get_obs_cond:93` and `_get_obs_cond:106`: switch `jnp.concatenate(..., axis=1)` to `axis=-1` on the latent path so two cameras stack along feature dim, not time. Unit-test with a synthetic batch.
- **Extend `vae_decode`** with cases `vae_feature_dim==256 → reshape (B*H, 8, 8, 4)` and `vae_feature_dim==1024 → reshape (B*H, 16, 16, 4)`. (Default 256; 1024 is a fallback if plan_viz looks too coarse and we have memory.) Decoder uses `obs_normalization[rgb_obs[0]]` only for plan_viz — visualizing just the agentview reconstruction is acceptable.

### 5. `process_sdvae_data.py` — new `run_libero` branch
- Selected by `"libero" in cfg.data.name`. Iterates `data.iter_demo_obs(rgb_keys)`, encodes each `(demo_id, key, frames)`, writes `latent.hdf5` with groups keyed by the **unique demo_id** (no name collisions across the 10 task files). Writes global `min_z, max_z` per `rgb_key` to root attrs.
- Don't share with `run_rm` (which assumes `next_obs/{key}` exists).

### 6. Client-server eval bridge
Transport: **subprocess in `/home/dodo/miniconda3/envs/libero/bin/python` per worker, length-prefixed pickle frames over stdin/stdout, stderr drained by a dedicated thread to a per-worker log file** (`bufsize=0`, explicit `flush()` after each frame). One subprocess per parallel rollout (matches `n_eval_processes`).

- **`envs/libero_proto.py`** — shared module (importable from both envs) with `read_frame(stream) / write_frame(stream, obj)` using `struct.pack(">I", len) + pickle.dumps(obj)`.
- **`envs/libero_obs_adapter.py`** — pure-python: rename `agentview_image→agentview_rgb`, `robot0_eye_in_hand_image→eye_in_hand_rgb`, build `ee_states = concat(robot0_eef_pos, R.from_quat(robot0_eef_quat).as_euler('xyz'))` (verify convention against a sample dataset trajectory at coding time), pass `gripper_states ← robot0_gripper_qpos`. Server-side so the schema mismatch lives in one file. Unit-testable without MuJoCo.
- **`envs/libero_eval_server.py`** (runs in libero env) — instantiates one `OffScreenRenderEnv` from CLI args (bddl_file, cameras, resolution); main loop reads `(cmd, payload)` frames, dispatches to `reset / set_init_state(idx) / step(action) / check_success / close`, returns adapted obs dict.
- **`envs/libero_remote_env.py`** (runs in ldp env) — `LiberoRemoteEnv(bddl_file, ...)` spawns the server subprocess and exposes `reset()`, `set_init_state(idx)`, `step(action)`, `is_success()`, `close()`. Mirrors the small surface used by `run_*_eval`. Includes a stderr-drain thread.
- **`utils/libero_env_utils.py`** — `run_libero_eval(env_params, agent, agent_name, n_rollout, n_proc, seed, rng)` mirrors `utils/rm_env_utils.py:run_robomimic_eval`'s queue-based worker pattern but spawns `LiberoRemoteEnv` instead of `RobosuiteEnv`. Pulls per-task BDDL paths and init states from a small `LiberoTaskRegistry` helper that wraps `libero.libero.benchmark.get_benchmark_dict()["libero_10"]()`. Init state per rollout: `init_states[seed % n_init_states]` (NOT the demo's first state).

### 7. Eval dispatch additions
- `train_bc.py:170-190` — add `elif "libero" in self.data.name: env_metrics, videos = libero_env_utils.run_libero_eval(...)`. Skip the `env_meta` update line.
- `eval_bc.py` — same dispatch.

### 8. Stats helpers (one-time scripts)
- **`scripts/compute_libero_obs_norm.py`** — sweeps all 130 hdf5 files, prints `obs_normalization` block (min/max per `ee_states/gripper_states/joint_states/...`) for paste into the data yamls.
- **`scripts/compute_libero_latent_stats.py`** — opens `latent.hdf5`, prints global `min_z/max_z` per latent key for paste into `latent_img.yaml`.

### 9. Bash entry scripts
`scripts/train_vae_libero.bash`, `scripts/preprocess_libero.bash`, `scripts/train_ldp_libero_long.bash`, `scripts/eval_ldp_libero_long.bash`. Each sets `WANDB_MODE=offline`, `CUDA_VISIBLE_DEVICES`, and prints the exact `python …` command for reproducibility.

### 10. Sanity test
- **`test/test_libero_data.py`** — loads one demo from one libero_10 hdf5, calls `iter_demo_obs`, asserts shapes match `shape_meta.all_shapes`. Catches the multi-file indexing class of bugs early.

## Files to create

| Path | Purpose |
|---|---|
| `data/libero_data.py` | Multi-file glob loader + demo registry + `iter_demo_obs` |
| `data/libero_latent_data.py` | Latent variant for planner/IDM training |
| `data/cfg/libero_all/img.yaml` | VAE training data (5 suites) |
| `data/cfg/libero_long/img.yaml` | Planner raw-image data (libero_10) |
| `data/cfg/libero_long/latent_img.yaml` | Planner latent data (libero_10) |
| `train_vae_libero.yaml` | Top-level VAE training config |
| `train_libero_long.yaml` | Top-level LDP training config |
| `eval_libero_long.yaml` | Top-level eval config |
| `envs/libero_proto.py` | Length-prefixed pickle frame helpers |
| `envs/libero_obs_adapter.py` | Runtime → dataset key/quat→euler bridge |
| `envs/libero_eval_server.py` | Server worker (libero env) |
| `envs/libero_remote_env.py` | Client wrapper (ldp env) |
| `utils/libero_env_utils.py` | `run_libero_eval`, `LiberoTaskRegistry` |
| `scripts/compute_libero_obs_norm.py` | One-time obs stats |
| `scripts/compute_libero_latent_stats.py` | Latent min/max extractor |
| `scripts/train_vae_libero.bash` | Bash entry |
| `scripts/preprocess_libero.bash` | Bash entry |
| `scripts/train_ldp_libero_long.bash` | Bash entry |
| `scripts/eval_ldp_libero_long.bash` | Bash entry |
| `test/test_libero_data.py` | Smoke test |

## Files to modify

| Path | Change |
|---|---|
| `agent/ldp_agent.py` | Fix multi-cam concat axis at `get_obs_cond:93` and `_get_obs_cond:106` (axis=1 → axis=-1 on latent path). Extend `vae_decode:69-80` with cases for `vae_feature_dim ∈ {256, 1024}`. |
| `process_sdvae_data.py` | Add `run_libero` branch (no-next_obs, multi-file via `data.iter_demo_obs`). |
| `train_bc.py` | Add `elif "libero" in self.data.name:` dispatch (skip `env_meta` update). |
| `eval_bc.py` | Same dispatch addition. |

## Existing code to reuse (do NOT reimplement)
- `data/alohasim_data.py` — template for no-next_obs HDF5 loader (`weld_demos` style).
- `utils/rm_env_utils.py:EvalProc` and `run_robomimic_eval` queue-based worker pattern — copy structure for `run_libero_eval`, swap `Process` → subprocess+pipes.
- `utils.flax_utils.TrainStateEMA`, `utils.data_utils.{normalize_obs, unnormalize_obs, postprocess_batch_obs}` — agent already uses these; nothing libero-specific to add.
- `process_sdvae_data.py:run_aloha` (lines 124-186) — template for `run_libero`.
- `model/stable_vae_model.py` + `agent/ldp_agent.py:vae_encode/vae_decode` — VAE infrastructure unchanged; only `vae_decode` reshape table is extended.

## Execution order

1. Write `LiberoData` + obs-norm script; produce stats; populate `cfg/libero_all/img.yaml`. Run `test_libero_data.py`.
2. Train VAE: `bash scripts/train_vae_libero.bash`. Confirm reconstructions visually.
3. Patch `process_sdvae_data.py` (`run_libero`); run `bash scripts/preprocess_libero.bash` to produce `latent.hdf5` for libero_10. Run latent-stats script; populate `cfg/libero_long/latent_img.yaml`.
4. Patch `agent/ldp_agent.py` (concat-axis fix + new vae_feature_dim cases). Spot-check with a tiny sanity batch (B=2, two cams) before launching real training.
5. Build the eval bridge end-to-end before training: spawn `LiberoRemoteEnv`, run `env.reset(); env.step(zeros)` once, dump first-frame image, compare against a training demo first-frame from the same init state (mean abs pixel diff < ~5).
6. Train LDP: `bash scripts/train_ldp_libero_long.bash` (start with `vae_feature_dim=256`).
7. Eval: `bash scripts/eval_ldp_libero_long.bash` runs all checkpoints over the 10 libero_10 tasks.

## Verification (end-to-end)

- **Data smoke**: `python -m pytest test/test_libero_data.py` passes; obs shapes match yaml.
- **VAE**: training loss < 0.05 on held-out images after ~100k steps; eyeballing recon quality on tensorboard.
- **Latent preprocessing**: `latent.hdf5` size ≈ #demos × avg_T × 4096 floats × 2 cams × 4 bytes; `min_z/max_z` attrs populated.
- **Multi-camera concat fix**: tiny unit test — call `_get_obs_cond` on a fake 2-cam latent batch, assert output shape `(B, T, 2*vae_feature_dim + lowdim)`.
- **Bridge**: `python envs/libero_remote_env.py --smoke` instantiates one env, reset, step, prints obs shapes; round-trip latency < 50ms/step.
- **Init-state sanity**: run one rollout with zeros action, render first frame, diff against `obs[demo_0]/agentview_rgb[0]` from same task — should be near-identical (after init_state load).
- **End-to-end**: `train_bc.py` reaches first eval cycle without crashing; eval reports per-task success rates from `run_libero_eval`. Use `n_eval_episodes=20, n_eval_processes=5` for the first smoke run.

## Risks and mitigations
- **Quaternion convention drift** between dataset Euler and `R.from_quat(...).as_euler('xyz')`: pin convention by sampling 10 frames where `ee_pos` is stationary, comparing computed Euler against stored `ee_ori`; choose `xyz/zyx` whichever matches. Bake into `libero_obs_adapter.py` with a comment.
- **Camera intrinsics mismatch** between training data renders and live env renders: render the first frame of demo 0 of task 0 from the running env at the exact init state and pixel-diff. If diff is large, check `OffScreenRenderEnv` defaults and adjust.
- **`vae_feature_dim=256` too coarse**: fallback path is `1024` (16×16×4) — both cases added in vae_decode. Visualize plan_viz to decide.
- **Subprocess hangs** if pickle stream gets stderr-corrupted: enforced with separate stderr drain thread and `bufsize=0`.
