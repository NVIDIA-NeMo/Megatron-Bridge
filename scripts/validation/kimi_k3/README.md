# Kimi K3 checkpoint-initialized training trial

This bounded driver exercises the original 93-layer Kimi K3 checkpoint on real
RP2 tokens: checkpoint import, DDP gradient buffers, full activation
recomputation, backward, and distributed Adam. It records phase memory, loss,
gradient norm, and sampled parameter updates for every rank.

This is an experimental functional recipe. Full-model training at 8K is being
validated; a completed process alone does not establish loss correctness or
long-run stability. Independent full-model loss comparison remains outstanding.

## Configuration

| Setting | Value |
| --- | --- |
| GPUs | 512 GB200, four per node |
| TP / PP / EP / ETP / CP / DP | 4 / 4 / 128 / 1 / 1 / 32 |
| Pipeline layers | 24 / 23 / 23 / 23 |
| Precision | BF16 model, FP32 gradient reduction |
| Microbatch / global batch | 1 / 128; four microbatches per DP rank |
| Sequence length | 8192; retry 4096 only after diagnosed OOM |
| Recomputation | Full, uniform, one layer per recomputation block |
| Optimizer | Distributed Adam, betas 0.9/0.95, epsilon 1e-8 |
| Weight decay / gradient clipping | 0.1 / 1.0 |
| Learning rate | Peak 1e-5, 100-step linear warmup, 10000-step cosine horizon |
| Trial | Ten updates; a 3000-second check before each step |

Keep an external one-hour job limit: a long individual step can exceed the
internal time check. No virtual pipeline or context-parallel KDA is requested.
Shortening the trial does not compress its learning-rate schedule. The first
update uses learning rate 1e-7.

## Inputs and launch

Use an existing local Kimi K3 snapshot containing its configuration, tokenizer,
remote model code, and checkpoint shards. The driver uses the installed Bridge
importer, including MXFP4 dequantization. It does not download weights or write
a full-model checkpoint.

Prepare RP2 with that snapshot's tokenizer, append EOS between documents, and
store token IDs as a contiguous **int32** binary stream at `RP2_PREFIX.bin`.
The training stream must exclude held-out documents. This driver reads the
stream directly; it does not accept arbitrary indexed-dataset integer dtypes.
At the default settings ten steps require at least 10,485,761 tokens. Samples
are deterministic, disjoint in prediction targets across DP ranks and steps,
and use contiguous next-token labels across document boundaries.

Set the following inside an existing allocation, with one process per GPU
and the usual `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`
environment variables. Replace the three input/output variables with your
existing shared-storage paths:

```bash
export NCCL_PXN_DISABLE=1 FLA_CACHE_MODE=always CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run python scripts/validation/kimi_k3/train.py \
  --snapshot "$KIMI_SNAPSHOT" \
  --data-prefix "$RP2_PREFIX" \
  --output "$TRIAL_OUTPUT" \
  --seq-length 8192 --steps 10
```

Submit all 512 processes with your scheduler. Keep source, caches, temporary
files, and per-rank logs in your approved job storage. If the 8K attempt runs
out of memory, retain its error and phase-memory records before trying
`--seq-length 4096`. This trial does not support sequences below 4096.

## Runtime controls

The investigated runtime uses FLA 0.5.2, Triton configuration caching, and NCCL
2.30.7. Pin the same runtime when comparing results. The driver checks the
expected tuning API and fails if it changes.

`NCCL_PXN_DISABLE=1` is an environment-specific transport mitigation: a prior
expert all-to-all check returned stale rows with the default path and passed
exact comparisons with PXN disabled. This does not fix the underlying library
or driver issue and carries no performance claim.

The seven KDA forward configurations and gated-normalization configuration
are held fixed to prevent previously observed BF16 differences between
otherwise identical replicas. These controls retain the installed kernel
arithmetic and pointer heuristics. The driver checks the configurations
actually selected after each step. Unspecified backward kernels use the
installed autotuner. `FLA_CACHE_MODE=always` rereads configuration files during
each controlled kernel call and adds diagnostic filesystem overhead; this
trial is not a throughput benchmark. The stored configurations were checked at short sequence
length; 8K backward and optimizer memory remain part of this trial.

## Interpreting results

Each `training-rank-N.json` records phase peaks and completed steps. Inspect all
512 records and require 128 distinct nodes, matching topology and step counts,
finite checkpoint parameters, finite gradients, a finite positive global
gradient norm, finite updated parameters, and at least one changed parameter
sample globally. Inspect the loss trajectory separately for instability.

`passed` means at least one update satisfied these execution checks;
`target_steps_completed` separately records whether all requested steps ran.
Neither field means independent loss agreement, convergence, or production
readiness. An OOM is saved as `oom: true` when PyTorch raises its OOM exception;
otherwise retain the rank traceback and scheduler termination reason. A killed
process can leave only its last progress record.
