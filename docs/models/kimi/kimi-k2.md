# Kimi K2

[Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct) is a large sparse MoE language model from Moonshot AI. Megatron Bridge supports Kimi K2 through the `KimiK2Bridge`, which reuses the common MLA/MoE mapping pattern used by DeepSeek-style architectures.

## Supported Variants

Megatron Bridge supports Kimi K2 checkpoints with the `KimiK2ForCausalLM` architecture and `kimi_k2` model type, including:

- Kimi-K2-Instruct: https://huggingface.co/moonshotai/Kimi-K2-Instruct

## Architecture Notes

- Multi-Latent Attention with QK layernorm.
- Sparse MoE layers with grouped GEMM, all-to-all token dispatch, expert bias, and shared expert overlap.
- RMSNorm, gated MLPs, RoPE, and unshared input/output embeddings.
- Vocab size is padded to a multiple of 1280 for Megatron execution.

## Recipes

Kimi K2 recipes live in [`src/megatron/bridge/recipes/kimi/kimi_k2.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/kimi/kimi_k2.py).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/kimi/kimi_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_bridge.py)
