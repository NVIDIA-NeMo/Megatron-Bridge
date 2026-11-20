# MiMo-V2-Flash Examples

This directory contains example scripts for [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) (Xiaomi), a 309B-total / ~15B-active sparse MoE model with hybrid attention (full + sliding window), fine-grained MoE (256 experts, top-8), asymmetric attention head dimensions (`head_dim=192` for Q/K, `v_head_dim=128` for V), dual-base RoPE, Multi-Token Prediction (MTP), and FP8 block-wise quantization.

The HF checkpoint depends on custom modeling code, so all commands below pass `--trust-remote-code`.

## Hardware Requirements

MiMo-V2-Flash requires **at least 2 nodes (16 GPUs)** for inference and conversion. Although the source checkpoint is FP8, conversion dequantizes its weights to BF16 before loading them into Megatron. The 47 MoE layers contain about 302.8B expert parameters, or about 605.6 GB in BF16, so the parallel layout must shard the target expert weights with enough headroom for the rest of the model and communication workspaces.

- With `EP=8, ETP=1`, each rank holds about 75.7 GB of BF16 expert parameters before dense parameters and communication workspaces, leaving insufficient memory headroom.
- Dense TP does **not** reduce expert memory when `ETP=1`. Set expert tensor parallelism (`--etp`) to shard each expert's matrices when using TP, or combine PP and EP so each rank holds fewer MoE layers/experts.
- Context parallelism is **not** supported (TE backends refuse CP + learnable softmax on SWA layers).
- TP size must be ≤ `min(num_key_value_heads, swa_num_key_value_heads)`.

The conversion example uses a layout that keeps dequantized expert parameters near 37.85 GB per GPU:

| TP | PP | EP | ETP |
|----|----|----|-----|
| 2  | 1  | 8  | 2   |

## Checkpoint Conversion

[slurm_conversion.sh](slurm_conversion.sh) uses `convert.sh roundtrip` to
submit the recommended `TP=2`, `PP=1`, `EP=8`, `ETP=2` config and verify HF ↔
Megatron round-trip conversion. Run the wrapper from a Slurm login node; it
waits for the job by default. Validation happens in memory rather than producing
another checkpoint.

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
bash examples/models/mimo_v2_flash/slurm_conversion.sh
```

Cluster-specific `srun` options are not enabled by default. Forward any that
your cluster requires, for example:

```bash
bash examples/models/mimo_v2_flash/slurm_conversion.sh \
    --srun-arg=--mpi=pmix
```

### Expected output

The public round-trip launcher prints a parameter-by-parameter comparison table
with ✅ / ❌ in the "Matches Original" column, and raises a
`ValueError("Weight mismatch detected")` on any mismatch. The wrapper exits
immediately when the synchronous job fails.

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

## MiMo-V2.6-Flash

[MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (`MiMoV2ForCausalLM`, `model_type: mimo_v2`) has the same language model as MiMo-V2-Flash, MTP layers included, and is converted by `MiMoV26Bridge`, which reuses the MiMo-V2-Flash bridge. Only the checkpoint format differs:

- `self_attn.qkv_proj` is one FP8 tensor stored as `num_key_value_heads` slices of `[Q | K | V]`, with 128x128 block scales that restart at each slice.
- Routed experts are packed MXFP4 (E2M1 values with one E8M0 `weight_scale` per 32 values).
- The dense and MTP MLPs are FP8 with 128x128 block scales, as in MiMo-V2-Flash.
- The vision, audio and speech-embedding weights are not converted.

Import dequantizes these weights to BF16. Export writes the source format back: weights that are MXFP4 or FP8 in the source checkpoint are requantized with its scale layout, `qkv_proj` is rebuilt slice by slice, the vision, audio and speech-embedding tensors are copied unchanged from the source checkpoint, and the source `quantization_config` is kept. Export therefore needs the source checkpoint (`--hf-model`), and `--export-weight-dtype` is not supported.

### Hardware Requirements

The routed experts take about 605 GB in BF16. One node of 8 GPUs with at least 141 GB each (for example B200) fits the model with `TP=1, PP=1, EP=8, ETP=1`. With 80 GB GPUs, use 2 nodes and shard the experts further, as for MiMo-V2-Flash above. The MiMo-V2-Flash restrictions apply: no context parallelism, and TP at most `min(num_key_value_heads, swa_num_key_value_heads)` = 4.

### Checkpoint Conversion

Import the Hugging Face checkpoint into a Megatron checkpoint:

```bash
./scripts/conversion/convert.sh import \
  --executor slurm --device gpu \
  --nodes 1 --gpus-per-node 8 \
  --account ACCOUNT --partition PARTITION --time 02:00:00 \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /workspace \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --env HF_TOKEN \
  --hf-model XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --megatron-path /workspace/models/mimo-v2.6-flash \
  --tp 1 --pp 1 --ep 8 --etp 1 \
  --trust-remote-code
```

Export it back to the Hugging Face source format:

```bash
./scripts/conversion/convert.sh export \
  --executor slurm --device gpu \
  --nodes 1 --gpus-per-node 8 \
  --account ACCOUNT --partition PARTITION --time 02:00:00 \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /workspace \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --env HF_TOKEN \
  --hf-model XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --megatron-path /workspace/models/mimo-v2.6-flash/iter_0000000 \
  --hf-path /workspace/models/mimo-v2.6-flash-hf \
  --tp 1 --pp 1 --ep 8 --etp 1 \
  --no-distributed-save \
  --trust-remote-code
```

Forward any `srun` options your cluster requires with `--srun-arg=ARG`, as described in the [conversion launcher README](../../../scripts/conversion/README.md).

### Expected output

The export writes every tensor of the source checkpoint into the same shard layout and fails if any is missing, since strict export is the default. Compared with the source checkpoint:

- BF16 and FP32 weights are identical.
- MXFP4 expert weights decode to identical values with identical scales. A zero may be stored as +0 where the source stores -0.
- FP8 weights keep their FP8 codes. Their scales are recomputed from the BF16 weights, so decoded values can differ by up to about 0.4% of a tensor's largest value.

The exported checkpoint keeps the source `config.json` fields, including `quantization_config`, so it loads in the same inference engines as the original checkpoint.
