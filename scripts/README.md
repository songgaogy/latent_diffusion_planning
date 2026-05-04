## Libero Long: Training / Preprocessing / Evaluation Pipeline

1. Train VAE (5 suites, ~5500 demos)
```bash
bash scripts/train_vae_libero.bash
```

2. Preprocess `libero_10` -> `latent.hdf5` (libero_long refers to libero_10)
```bash
bash scripts/preprocess_libero.bash \
  experiment_folder=libero_long \
  experiment_name=preproc01 \
  restore_snapshot_path=$(pwd)/experiments/libero_vae/vae_all64/ckpt/275000.ckpt
```

3. Generate/write `latent stats` into the latent YAML
```bash
source scripts/env_helpers.bash
PY="$(resolve_ldp_python)"
"$PY" scripts/compute_libero_latent_stats.py \
  --latent experiments/libero_long/preproc64/latent.hdf5
```

4. Train LDP
```bash
bash scripts/train_ldp_libero_long.bash \
  agent.vae_pretrain_path=$(pwd)/experiments/libero_vae/vae_all64/ckpt/275000.ckpt \
  data.train_latent_path=$(pwd)/experiments/libero_long/preproc64/latent.hdf5 \
  experiment_name="ldp_goal_cond_64_v0"
```

5. Evaluation
```bash
bash scripts/eval_ldp_libero_long.bash \
  experiment_folder=libero_long \
  experiment_name=ldp_long01 \
  agent.vae_pretrain_path=$(pwd)/experiments/libero_vae/vae_all64/ckpt/200000.ckpt \
  data.train_latent_path=$(pwd)/experiments/libero_long/preproc01/latent.hdf5
```
