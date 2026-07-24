# Llama 2

> **Deprecation notice:** Llama 2 support is no longer actively maintained or
> tested against current upstream checkpoints and will be removed in Megatron
> Bridge 0.7.0. The shared Llama bridge remains supported for Llama 3 variants.

[Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf) is Meta's decoder-only transformer model family. Megatron Bridge supports Llama 2 through the shared Llama bridge for the Hugging Face `LlamaForCausalLM` architecture.

## Supported Variants

The repository includes a pretraining recipe for:

- Llama-2-7B: https://huggingface.co/meta-llama/Llama-2-7b-hf

The same bridge can auto-detect compatible Llama 2 causal language model checkpoints.

## Architecture Notes

- RMSNorm with SwiGLU feed-forward layers.
- RoPE positional embeddings.
- Grouped Query Attention support follows the Hugging Face configuration.
- Uses the same `LlamaBridge` implementation as later Llama 3 variants.

## Examples

Llama 2 uses the common conversion and generation entry points:

```bash
./scripts/conversion/convert.sh import \
  --hf-model meta-llama/Llama-2-7b-hf \
  --megatron-path /checkpoints/llama2_7b_megatron
```

```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path meta-llama/Llama-2-7b-hf \
  --megatron_model_path /checkpoints/llama2_7b_megatron \
  --prompt "What is artificial intelligence?"
```

## Recipes

- Llama 2 recipe: [`src/megatron/bridge/recipes/llama/llama2.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/llama/llama2.py)

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/llama/llama_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py)
