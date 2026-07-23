# Xiaomi-MiMo

Xiaomi-MiMo models use a Qwen2-style causal language backbone with Multi-Token Prediction support. Megatron Bridge supports MiMo causal language models through `MimoBridge`, which extends the Qwen2 bridge and adds MTP weight mappings.

## Supported Variants

Megatron Bridge supports Hugging Face checkpoints using the `MiMoForCausalLM` architecture and `mimo` model type.

## Architecture Notes

- Qwen2-style attention behavior with QKV bias enabled.
- Optional MTP layers are enabled from `num_nextn_predict_layers` in the Hugging Face config.
- The bridge maps MTP token/hidden layernorms, projection layers, attention weights, and gated MLP weights.
- Input projection halves are swapped during import/export to match Megatron and Hugging Face layouts.

## Examples

Checkpoint conversion and text-generation examples for MiMo causal language
models live under
[`examples/models/mimo`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/6a706b1c3afab4d21a6f5dd88aa6a75296d33fe2/examples/models/mimo/README.md).
Heterogeneous multimodal training examples live separately under
[`examples/megatron_mimo`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/megatron_mimo).

## Related Implementation

- MiMo bridge: [`src/megatron/bridge/models/mimo/mimo_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/mimo/mimo_bridge.py)
- Megatron-MiMo provider infrastructure: [`src/megatron/bridge/models/megatron_mimo`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/megatron_mimo)
