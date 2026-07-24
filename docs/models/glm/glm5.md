# GLM-5, GLM-5.1, and GLM-5.2

[GLM-5](https://huggingface.co/zai-org/GLM-5), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), and [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) are large sparse MoE language models with Multi-Latent Attention and Dynamic Sparse Attention. Megatron Bridge supports these checkpoints through the shared `GLM5Bridge`.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| GLM-5 | `zai-org/GLM-5` | MoE + MLA + DSA architecture |
| GLM-5.1 | `zai-org/GLM-5.1` | Same architecture and mapping shape as GLM-5 |
| GLM-5.2 | `zai-org/GLM-5.2` | Adds IndexShare-style DSA index reuse settings |

## Architecture Notes

- `GlmMoeDsaForCausalLM` architecture with 78 transformer layers.
- First 3 layers are dense; remaining layers use MoE.
- 256 routed experts with top-8 routing and one shared expert per MoE layer.
- Uses MLA plus DSA indexer parameters (`index_head_dim`, `index_n_heads`, `index_topk`).
- Requires `transformers >= 5.2.0`.
- DSA requires the `fast-hadamard-transform` CUDA extension and MCore support for the DSA experimental attention variant.

## Examples

For conversion, inference, dependency notes, hardware requirements, and MCore patch requirements, see the [GLM-5 examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/glm/glm5/README.md).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py)
- Examples: [`examples/models/glm/glm5`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/glm/glm5)
