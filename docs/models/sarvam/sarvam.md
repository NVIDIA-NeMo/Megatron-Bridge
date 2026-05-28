# Sarvam

[Sarvam](https://huggingface.co/sarvamai/sarvam-30b) language models are MoE models from Sarvam AI. Megatron Bridge supports Sarvam dense-MLA and MoE variants through the Sarvam bridge implementations.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| Sarvam 30B | `sarvamai/sarvam-30b` | 30B total, 3B active, 128 experts top-6 |
| Sarvam 105B | `sarvamai/sarvam-105b` | 105B total, 10.3B active, 128 experts top-8 |

## Architecture Notes

- Sarvam MoE models use QKV layernorm and Grouped Query Attention.
- MoE layers map router weights, expert bias, shared experts, and per-expert gate/up/down projections.
- Examples use `--trust-remote-code` for Hugging Face loading.

## Examples

For checkpoint import/export, round-trip validation, and inference commands, see the [Sarvam examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/sarvam/README.md).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/sarvam`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/sarvam)
- Examples: [`examples/models/sarvam`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/sarvam)

