# GLM-5 / GLM-5.1 / GLM-5.2 Examples

Scripts for the GLM-5 family: [GLM-5](https://huggingface.co/zai-org/GLM-5) (`zai-org/GLM-5`), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) (`zai-org/GLM-5.1`), and [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) (`zai-org/GLM-5.2`). These are large sparse MoE models with Multi-Latent Attention (MLA) and Dynamic Sparse Attention (DSA).

All three variants share the `GlmMoeDsaForCausalLM` architecture and use the same `GLM5Bridge`. Set `MODEL_NAME` to select the checkpoint in the Slurm scripts.

| Property | Value |
|---|---|
| HF model IDs | `zai-org/GLM-5`, `zai-org/GLM-5.1`, `zai-org/GLM-5.2` |
| Architecture | MoE + MLA + DSA (`GlmMoeDsaForCausalLM`) |
| Layers | 78 logical blocks; 156 physical Hybrid layers (first 3 MLPs dense, rest MoE) |
| Routed experts | 256, top-8 per token |
| Shared experts | 1 per MoE layer |
| Total params | ~750-800B (BF16, variant-dependent) |
| Active params | ~60B per token |

**Requirements:** `transformers >= 5.2.0`, `fast-hadamard-transform` (CUDA extension, required by DSA)

The bridge represents each logical block as a `D-` dense pair or `DE` MoE pair in `HybridModel`. GLM-5.2 IndexShare is derived from `indexer_types`; when PP is enabled, stage boundaries are automatically moved to full-indexer DSA layers so shared indexer output never crosses stages. BF16 checkpoint conversion is supported; FP8 checkpoints and MTP execution are outside the current scope.

## Hardware Requirements

Full-model conversion and inference in BF16 requires **at least 8 nodes (64 GPUs × 80 GB)**. Key constraints:

- EP must divide 256 (number of routed experts). Valid: 1, 2, 4, 8, 16, 32, 64, 128, 256.
- TP does **not** reduce expert memory — increase EP instead.
- Minimum recommended: `TP=2, EP=32, PP=1` (64 GPUs, 8 nodes).
- `TP=1, EP=64` works for conversion but may cause empty-dispatch issues during autoregressive inference with single-token batches. Use `TP >= 2` for inference.

## Inference (Megatron)

[slurm_inference.sh](slurm_inference.sh) loads the HF checkpoint, converts to Megatron in-memory, and runs greedy text generation with `TP=2, EP=32` across 64 GPUs.

```bash
sbatch examples/models/glm/glm5/slurm_inference.sh
```

### Expected output

```
======== GENERATED TEXT OUTPUT ========
Prompt: What is artificial intelligence?
Generated: What is artificial intelligence? Artificial intelligence (AI) is a field of
computer Science and Engineering that deals with the creation of intelligent
machines, which are used in different areas such...
=======================================
```

## Checkpoint Conversion (Round-Trip)

[slurm_conversion.sh](slurm_conversion.sh) runs HF → Megatron → HF round-trip conversion and verifies weight fidelity. Saves the exported HF checkpoint to `OUTPUT_DIR`.

```bash
sbatch examples/models/glm/glm5/slurm_conversion.sh
```

Default config (8 nodes, 64 GPUs): `TP=2, EP=32`.

> **Note:** The round-trip verification step (comparing ~63K weight tensors on rank 0)
> may hit Lustre I/O contention at this model scale. The HF→Megatron conversion
> itself is validated by the successful inference above.

## Script Configuration

Both scripts resolve the HF model from the local cache to avoid `snapshot_download` race conditions with 64 concurrent processes. Set these environment variables before submitting:

| Variable | Description |
|---|---|
| `CONTAINER_IMAGE` | Path to Singularity/SquashFS container image |
| `BRIDGE_PATH` | Megatron-Bridge checkout on shared storage (bind-mounted as `/opt/Megatron-Bridge`) |
| `HF_HOME` | HuggingFace cache directory containing the selected `zai-org/GLM-5` family checkpoint |
| `HF_TOKEN` | HuggingFace access token (for gated model access) |
| `MODEL_NAME` | HF model name without the `zai-org/` prefix; defaults to `GLM-5`. Supported values include `GLM-5`, `GLM-5.1`, and `GLM-5.2`. |
| `OUTPUT_DIR` | Conversion output directory (conversion script only) |
