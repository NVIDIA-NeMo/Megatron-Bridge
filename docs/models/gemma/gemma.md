# Gemma

> **Deprecation notice:** Gemma 1 support (2B and 7B) is no longer actively
> maintained or tested against current upstream checkpoints and will be removed
> in Megatron Bridge 0.7.0.

[Gemma](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b) is Google's original lightweight open model family. Megatron Bridge supports Gemma causal language models through the `GemmaBridge` implementation for the Hugging Face `GemmaForCausalLM` architecture.

## Supported Variants

Megatron Bridge supports Hugging Face Gemma checkpoints that use the `gemma` model type, including:

- Gemma 2B: https://huggingface.co/google/gemma-2b
- Gemma 7B: https://huggingface.co/google/gemma-7b
- Gemma release collection: https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b

## Architecture Notes

- RMSNorm with zero-centered gamma.
- GeGLU-style gated MLPs.
- RoPE positional embeddings and flash attention backend.
- Shared input/output embedding weights.

## Examples

Gemma uses the common conversion and generation entry points:

```bash
./scripts/conversion/convert.sh import \
  --hf-model google/gemma-2b \
  --megatron-path /checkpoints/gemma_2b_megatron
```

```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path google/gemma-2b \
  --megatron_model_path /checkpoints/gemma_2b_megatron \
  --prompt "What is artificial intelligence?"
```

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/gemma/gemma_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/gemma/gemma_bridge.py)
- Conversion examples: [`examples/conversion`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/conversion)
