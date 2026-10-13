# Kimi K3 pretraining

`kimi_k3_pretrain_config` is **early, perf not optimized**. It uses the native
K3 provider with TP4/PP4/EP128 on 512 GB200 GPUs, BF16, full activation
recomputation, and distributed Adam. The uneven pipeline has 24/23/23/23 layers;
CP and virtual pipeline parallelism are disabled.

Launch with the standard recipe runner inside your allocation:

```bash
export KIMI_K3_HF_PATH="$KIMI_SNAPSHOT"
export FLA_CACHE_MODE=always NCCL_PXN_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
uv run python scripts/training/run_recipe.py \
  --recipe kimi_k3_pretrain_config \
  --pretrained_checkpoint "$KIMI_SNAPSHOT" \
  --seq_length 8192 --max_steps 10 \
  dataset.blend="[['$RP2_TRAIN_PREFIX'],null]" dataset.split=100,0,0 \
  dataset.path_to_cache="$DATA_CACHE" \
  validation.eval_iters=0 \
  checkpoint.save=null checkpoint.load=null
```

Provide an existing indexed RP2 `.bin`/`.idx` pair tokenized with the K3
snapshot tokenizer. The example consumes only the training prefix; held-out
RP2 documents should be prepared separately. The default recipe's dataset
placeholder must be replaced for real-data training. Configure tokenizer,
cache, temporary, logger, and checkpoint paths for your allocation. For an
offline snapshot set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` as well.

The first functional attempt uses 8192 tokens. If it OOMs, preserve the error
and memory logs before retrying `--seq_length 4096`; the initial validation
plan does not go below 4096. Keep the external one-hour limit for this trial.
The 100-step warmup and 10000-step cosine horizon remain unchanged when
`--max_steps` is reduced. Standard scheduler initialization starts at zero
learning rate and advances after each update.

The recipe retains two controls from prior diagnostics: `NCCL_PXN_DISABLE=1`
for an observed expert all-to-all transport failure, and fixed KDA/gated-norm
forward tuning configurations for otherwise identical BF16 replicas. The
pre-wrap hook installs these through the investigated FLA 0.5.2/Triton APIs,
preserves kernel arithmetic, and verifies selected configurations after each
forward. These controls do not repair underlying runtime libraries.
`FLA_CACHE_MODE=always` adds file reads and is diagnostic overhead. Backward
kernels retain the installed autotuner.

The full 8K standard-runner training attempt is pending. Inspect actual
iterations, loss, gradient norm, skipped/NaN counts, and memory logs;
scheduler completion alone is insufficient. Successful short execution does
not establish independent full-model loss agreement or long-run convergence.
