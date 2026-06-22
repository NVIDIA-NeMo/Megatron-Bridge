# Falcon H1

[Falcon H1](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct) is a hybrid language model family that combines Mamba, attention, and MLP blocks in each decoder layer. Megatron Bridge supports Falcon H1 through a custom provider and model implementation.

## Supported Variants

The public examples target the smallest instruction checkpoint:

- Falcon-H1-0.5B-Instruct: https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct

The bridge is registered for the `FalconH1ForCausalLM` architecture and can auto-detect compatible Falcon H1 checkpoints when their Hugging Face config uses the `falcon_h1` model type.

## Architecture Notes

- Uses a custom `FalconH1ModelProvider` and `FalconH1Model`.
- Each decoder block is represented as a parallel Mamba, attention, and MLP layer.
- The bridge maps Mamba input projections, convolution weights, state-space parameters, QKV projections, and gated MLP weights.
- Falcon H1 checkpoints use custom Hugging Face code; examples pass `--trust-remote-code`.

## Examples

For checkpoint conversion, tokenizer asset export, round-trip validation, and inference, see the [Falcon H1 examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/falcon_h1/README.md).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/falcon_h1`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/falcon_h1)
- Examples: [`examples/models/falcon_h1`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/falcon_h1)

