# MiMo-V2-Flash Examples

This directory contains example scripts for [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) (Xiaomi), a 309B-total / ~15B-active sparse MoE model with hybrid attention (full + sliding window), fine-grained MoE (256 experts, top-8), asymmetric attention head dimensions (`head_dim=192` for Q/K, `v_head_dim=128` for V), dual-base RoPE, Multi-Token Prediction (MTP), and FP8 block-wise quantization.

The HF checkpoint depends on custom modeling code, so all commands below pass `--trust-remote-code`.

## Hardware Requirements

MiMo-V2-Flash requires **at least 2 nodes (16 GPUs)** for inference and conversion. Although the source checkpoint is FP8, conversion dequantizes its weights to BF16 before loading them into Megatron. The 47 MoE layers contain about 302.8B expert parameters, or about 605.6 GB in BF16, so the parallel layout must shard the target expert weights with enough headroom for the rest of the model and communication workspaces.

- With `EP=8, ETP=1`, each rank holds about 75.7 GB of BF16 expert parameters before dense parameters and communication workspaces, leaving insufficient memory headroom.
- Dense TP does **not** reduce expert memory when `ETP=1`. Set expert tensor parallelism (`--etp`) to shard each expert's matrices when using TP, or combine PP and EP so each rank holds fewer MoE layers/experts.
- Context parallelism is **not** supported (TE backends refuse CP + learnable softmax on SWA layers).
- TP size must be ≤ `min(num_key_value_heads, swa_num_key_value_heads)`.

The conversion sweep uses layouts that keep dequantized expert parameters near 37.85 GB per GPU:

| TP | PP | EP | ETP |
|----|----|----|-----|
| 2  | 1  | 8  | 2   |
| 1  | 2  | 8  | 1   |
| 2  | 2  | 4  | 2   |

## Checkpoint Conversion

[slurm_conversion.sh](slurm_conversion.sh) sweeps multiple TP/PP/EP/ETP configs to verify HF ↔ Megatron round-trip conversion.

### Setup

Edit the variables at the top of `slurm_conversion.sh`:

```bash
CONTAINER_IMAGE="/path/to/container.sqsh"
# Optional:
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/path/to/shared/HF_HOME"
```

### Submit

```bash
sbatch examples/models/mimo_v2_flash/slurm_conversion.sh
```

### Expected output

The slurm wrapper prints a header per config and an `[OK]` line on success.
The underlying conversion script (`hf_megatron_roundtrip_multi_gpu.py`)
prints a parameter-by-parameter comparison table with ✅ / ❌ in the
"Matches Original" column, and raises a `ValueError("Weight mismatch
detected")` on any mismatch (which the wrapper turns into an `ERROR`
line and a non-zero exit). A successful run ends with:

```
[OK] Config 3: TP=2, PP=2, EP=4, ETP=2 passed

======================================
All 3 configs passed
======================================
```

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
sbatch examples/models/mimo_v2_flash/slurm_inference.sh
```

### Expected output

`hf_to_megatron_generate_text.py` ends with a rank-0 print block of the form:

```
======== GENERATED TEXT OUTPUT ========
Prompt: What is artificial intelligence?
Generated: <model's continuation of the prompt>
=======================================
```
