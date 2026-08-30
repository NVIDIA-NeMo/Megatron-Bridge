# Ling 3.0

[Ling 3.0 Tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) and [Ling 3.0 Flash](https://huggingface.co/inclusionAI/Ling-3.0-flash) are hybrid Mixture-of-Experts language models from inclusionAI. Megatron Bridge supports both public variants through the Bailing bridge with auto-detected Hugging Face configuration and bidirectional weight conversion.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| Ling 3.0 Tiny | `inclusionAI/Ling-3.0-tiny` | 24 logical blocks, 128 routed experts, low-rank-Q MLA |
| Ling 3.0 Flash | `inclusionAI/Ling-3.0-flash` | 42 logical blocks, 512 routed experts, direct-Q MLA and one MTP layer |

## Architecture Notes

- Hybrid decoder with KDA or MLA attention and dense or MoE feed-forward stages. Each Hugging Face logical block maps to two Megatron Core `HybridModel` positions.
- Sigmoid top-8 expert routing with shared experts and head-wise gated MLA output.
- Tiny uses low-rank-Q MLA, while Flash uses direct-Q MLA and includes MTP.
- Custom Hugging Face model code is required, so conversion commands use `--trust-remote-code`.

## Examples

For checkpoint import/export and round-trip validation, see the [Bailing examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/bailing/README.md).

## Training Recipe

The `ling_v3_tiny_pretrain_8gpu_h100_bf16_config` recipe provides a full-size, fresh-initialization, mock-data training smoke for Ling 3.0 Tiny. It enables one MTP depth for training-path coverage, independently of the public Tiny checkpoint, which has no MTP tensors. A training recipe is currently provided for Tiny only.

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/bailing`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/bailing)
- Tiny training recipe: [`src/megatron/bridge/recipes/bailing/h100/ling_v3_tiny.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/bailing/h100/ling_v3_tiny.py)
- Examples: [`examples/models/bailing`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/bailing)
