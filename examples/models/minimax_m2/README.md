# MiniMax-M2 Examples

This directory contains example scripts for [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2), a large sparse MoE model with 456B total parameters (45.9B active), 256 experts, and FP8 quantization.

## Hardware Requirements

MiniMax-M2 requires **at least 2 nodes (16 GPUs)** for inference and conversion. The model cannot fit on a single 8-GPU node because:

- TEGroupedMLP workspace is proportional to `num_experts / EP`; with EP=8 on 1 node, workspace alone OOMs.
- TP does **not** reduce expert memory — use EP instead.
- Minimum recommended config: `TP=1, EP=16, PP=1` (2 nodes × 8 GPUs).

## Checkpoint Conversion

[slurm_conversion.sh](slurm_conversion.sh) sweeps multiple TP/PP/EP configs to verify HF ↔ Megatron round-trip conversion.

### Setup

Edit the variables at the top of `slurm_conversion.sh`:

```bash
CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,/path/to/project:/opt/Megatron-Bridge"
# Optional:
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/path/to/shared/HF_HOME"
```

### Submit

```bash
sbatch examples/models/minimax_m2/slurm_conversion.sh
```

### Expected output (per config)

```
MiniMax-M2 Round-Trip Conversion Sweep
Job: <JOB_ID> | Nodes: 2
Parallelism configs: 2,1,8 1,2,8 2,2,4
======================================
Config 1/3: TP=2, PP=1, EP=8
...
All parameters match: True ✅
[OK] Config 1: TP=2, PP=1, EP=8 passed
...
All 3 configs passed
======================================
```

## Inference

[slurm_inference.sh](slurm_inference.sh) runs text generation on the full FP8 checkpoint with `TP=1, EP=16`.

### Setup

Edit the variables at the top of `slurm_inference.sh`:

```bash
CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,/path/to/project:/opt/Megatron-Bridge"
export HF_TOKEN="hf_your_token_here"
```

### Submit

```bash
sbatch examples/models/minimax_m2/slurm_inference.sh
```

### Expected output

```
======== GENERATED TEXT OUTPUT ========
Prompt: What is artificial intelligence?
Generated: What is artificial intelligence? Artificial intelligence (AI) is the simulation of...
=======================================
```

Verified with prompt `"What is 2+2?"` → generates `"The answer is 4."`.
