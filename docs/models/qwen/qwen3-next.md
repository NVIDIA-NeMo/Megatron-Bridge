# Qwen3-Next

[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) is a Qwen3 MoE variant with Gated-Delta Networks and Multi-Token Prediction support. Megatron Bridge supports it through the `Qwen3NextBridge`.

## Supported Variants

- Qwen3-Next-80B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-Next-80B-A3B-Thinking: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking

## Architecture Notes

- MoE model with 80B total parameters and 3B active parameters.
- Includes Gated-Delta Network layers, QK layernorm, Zero-Centered RMSNorm, and MTP.
- Recipes currently cover pretraining and full SFT; PEFT is not available for Qwen3-Next.

## Examples

The Qwen3-Next example entrypoint supports YAML and CLI overrides for full finetuning:

- [`examples/models/qwen/qwen3_next/finetune_qwen3_next_80b_a3b.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/qwen/qwen3_next/finetune_qwen3_next_80b_a3b.py)

## Recipes

- `qwen3_next_80b_a3b_pretrain_config`
- `qwen3_next_80b_a3b_sft_config`

See [`src/megatron/bridge/recipes/qwen/qwen3_next.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen/qwen3_next.py).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/qwen/qwen3_next_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen3_next_bridge.py)
- Family overview: [qwen.md](qwen.md)

