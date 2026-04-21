# GLM-5 Examples

Scripts for [GLM-5](https://huggingface.co/zai-org/GLM-5) (`zai-org/GLM-5`), a large sparse MoE model with Multi-Latent Attention (MLA) and Dynamic Sparse Attention (DSA).

| Property | Value |
|---|---|
| HF model ID | `zai-org/GLM-5` |
| Architecture | MoE + MLA + DSA |
| Layers | 78 transformer (first 3 dense, rest MoE) |
| Routed experts | 256, top-8 per token |
| Shared experts | 1 per MoE layer |
| Total params | ~800B+ (BF16) |
| Active params | ~60B per token |

**Requirements:** `transformers >= 5.2.0`, `fast-hadamard-transform` (CUDA extension, required by DSA)

## Hardware Requirements

GLM-5 requires **at least 8 nodes (64 GPUs × 80 GB)** for full-model conversion and inference in BF16. Key constraints:

- EP must divide 256 (number of routed experts). Valid: 1, 2, 4, 8, 16, 32, 64, 128, 256.
- TP does **not** reduce expert memory — increase EP instead.
- Minimum recommended: `TP=2, EP=32, PP=1` (64 GPUs, 8 nodes).
- `TP=1, EP=64` works for conversion but may cause empty-dispatch issues during autoregressive inference with single-token batches. Use `TP >= 2` for inference.

### Pre-requisites

Install `fast-hadamard-transform` (required by the DSA attention variant) into the project venv from a GPU node:

```bash
pip install --target=.venv/lib/python3.12/site-packages --no-deps --no-build-isolation \
    git+https://github.com/Dao-AILab/fast-hadamard-transform.git
```

The PyPI source distribution is incomplete; install from the git repo.

## Inference (Megatron)

[slurm_inference.sh](slurm_inference.sh) loads the HF checkpoint, converts to Megatron in-memory, and runs greedy text generation with `TP=2, EP=32` across 64 GPUs.

```bash
sbatch examples/models/glm5/slurm_inference.sh
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
sbatch examples/models/glm5/slurm_conversion.sh
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
| `HF_HOME` | HuggingFace cache directory (must contain the downloaded `zai-org/GLM-5` model) |
| `HF_TOKEN` | HuggingFace access token (for gated model access) |
| `OUTPUT_DIR` | Conversion output directory (conversion script only) |

## MCore Patches Required

GLM-5's DSA attention variant requires two patches to `megatron/core/models/gpt/experimental_attention_variant_module_specs.py`:

1. **DSA dispatch:** Add `elif config.experimental_attention_variant == "dsa"` to `get_experimental_attention_variant_module_spec` to call `get_dsa_module_spec_for_backend`.
2. **MLA metainfo:** Add `metainfo={"fuse_input_layernorm": False}` to the `MLASelfAttention` `ModuleSpec` in `get_dsa_module_spec_for_backend`.
