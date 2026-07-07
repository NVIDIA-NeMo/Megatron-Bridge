# GLM-5, GLM-5.1, and GLM-5.2

[GLM-5](https://huggingface.co/zai-org/GLM-5), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), and [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) are large sparse MoE language models with Multi-Latent Attention and Dynamic Sparse Attention. Megatron Bridge supports all three checkpoints through the shared `GLM5Bridge`.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| GLM-5 | `zai-org/GLM-5` | MoE + MLA + DSA architecture |
| GLM-5.1 | `zai-org/GLM-5.1` | Same architecture and mapping shape as GLM-5 |
| GLM-5.2 | `zai-org/GLM-5.2` | Adds DSA IndexShare between selected logical layers |

## Architecture Notes

- `GlmMoeDsaForCausalLM` architecture with 78 logical transformer blocks.
- Each logical block is represented by two `HybridModel` layers: `D-` for dense blocks and `DE` for MoE blocks. The standard 3-dense/75-MoE layout therefore has 156 physical layers.
- First 3 layers are dense; remaining layers use MoE.
- 256 routed experts with top-8 routing and one shared expert per MoE layer.
- Uses MLA plus DSA indexer parameters (`index_head_dim`, `index_n_heads`, `index_topk`).
- GLM-5.2 IndexShare is translated from the logical `indexer_types` schedule. Pipeline boundaries are placed only before DSA layers that compute their own indexer output.
- Requires `transformers >= 5.2.0`.
- DSA requires the `fast-hadamard-transform` CUDA extension.
- BF16 checkpoint conversion is supported. FP8 checkpoints and MTP execution are not currently supported.

## Examples

For conversion, inference, dependency notes, and hardware requirements, see the [GLM-5 examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/glm/glm5/README.md).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py)
- Examples: [`examples/models/glm/glm5`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/glm/glm5)
