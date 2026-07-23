# MiniMax-M2 Examples

This directory contains example scripts for [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2), a large sparse MoE model with 456B total parameters (45.9B active), 256 experts, and FP8 quantization.

> **M2.5 / M2.7 compatibility:** [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) and [MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) share the same architecture (`MiniMaxM2ForCausalLM`) and work with the same bridge. Replace the model ID in the scripts below (e.g. `MiniMaxAI/MiniMax-M2.5`).

## Hardware Requirements

MiniMax-M2 requires **at least 2 nodes (16 GPUs)** for inference and conversion. The model cannot fit on a single 8-GPU node because:

- TEGroupedMLP workspace is proportional to `num_experts / EP`; with EP=8 on 1 node, workspace alone OOMs.
- TP does **not** reduce expert memory — use EP instead.
- Minimum recommended config: `TP=1, EP=16, PP=1` (2 nodes × 8 GPUs).

## Checkpoint Conversion

[slurm_conversion.sh](slurm_conversion.sh) uses `convert.sh roundtrip` to
submit a fixed `TP=2`, `PP=1`, `EP=8` config and verify HF ↔ Megatron
round-trip conversion. Run the wrapper from a Slurm login node; it waits for
the job by default. Validation happens in memory rather than producing another
checkpoint.

### Setup

```bash
export CONTAINER_IMAGE=/path/to/container.sqsh
export SLURM_ACCOUNT=your_account
export SLURM_PARTITION=batch
export CONTAINER_MOUNTS=/shared:/shared
# Optional: export HF_TOKEN and HF_HOME before launching.
```

The current checkout is mounted automatically at `/opt/Megatron-Bridge` and
must be on storage visible from the compute nodes. Add other comma-separated
host-to-container mounts through `CONTAINER_MOUNTS`.

### Submit

```bash
bash examples/models/minimax/minimax_m2/slurm_conversion.sh
```

Cluster-specific `srun` options are not enabled by default. Forward any that
your cluster requires, for example:

```bash
bash examples/models/minimax/minimax_m2/slurm_conversion.sh \
    --srun-arg=--mpi=pmix
```

### Expected output

The round-trip launcher prints a parameter comparison table with successful
matches marked ✅ and exits nonzero if any converted weight differs.

## Inference

[slurm_inference.sh](slurm_inference.sh) runs text generation on the full FP8 checkpoint with `TP=1, EP=16`.

### Setup

Edit the variables at the top of `slurm_inference.sh`:

```bash
CONTAINER_IMAGE="/path/to/container.sqsh"
export HF_TOKEN="hf_your_token_here"
```

### Submit

```bash
sbatch examples/models/minimax/minimax_m2/slurm_inference.sh
```

### Expected output

```
======== GENERATED TEXT OUTPUT ========
Prompt: What is 2+2?
Generated: What is 2+2? The answer is 4.
=======================================
```
