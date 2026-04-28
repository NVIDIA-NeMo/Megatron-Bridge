# Qwen 3.5 / 3.6

[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) is a family of vision-language models supporting multimodal understanding across text, images, and videos. Qwen3.5-VL includes both dense models and Mixture-of-Experts (MoE) variants for improved efficiency at scale.

[Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) shares the same architecture as Qwen3.5 VL MoE (`Qwen3_5MoeForConditionalGeneration`) and is supported through the same bridge with no code changes required.

Qwen 3.5/3.6 models feature a hybrid architecture combining GDN (Gated DeltaNet) layers with standard attention layers, SwiGLU activations, and RMSNorm. MoE variants use top-k routing with shared experts for better quality.

Qwen 3.5/3.6 models are supported via Megatron Bridge with auto-detected configuration and weight mapping.

```{important}
Please upgrade to `transformers` >= 5.2.0 in order to use the Qwen 3.5 models.
```

## Available Models

### Dense Models
- **Qwen3.5 0.8B** (`Qwen/Qwen3.5-0.8B`): 0.8B parameter vision-language model
  - Recommended: 1 node, 8 GPUs

- **Qwen3.5 2B** (`Qwen/Qwen3.5-2B`): 2B parameter vision-language model
  - Recommended: 1 node, 8 GPUs

- **Qwen3.5 4B** (`Qwen/Qwen3.5-4B`): 4B parameter vision-language model
  - Recommended: 1 node, 8 GPUs

- **Qwen3.5 9B** (`Qwen/Qwen3.5-9B`): 9B parameter vision-language model
  - Recommended: 1 node, 8 GPUs

- **Qwen3.5 27B** (`Qwen/Qwen3.5-27B`): 27B parameter vision-language model
  - Recommended: 2 nodes, 16 GPUs

### Mixture-of-Experts (MoE) Models
- **Qwen3.5 35B-A3B** (`Qwen/Qwen3.5-35B-A3B`): 35B total parameters, 3B activated per token
  - Recommended: 2 nodes, 16 GPUs

- **Qwen3.5 122B-A10B** (`Qwen/Qwen3.5-122B-A10B`): 122B total parameters, 10B activated per token
  - Recommended: 4 nodes, 32 GPUs

- **Qwen3.5 397B-A17B** (`Qwen/Qwen3.5-397B-A17B`): 397B total parameters, 17B activated per token
  - 512 experts with top-10 routing and shared experts
  - Recommended: 16 nodes, 128 GPUs

### Qwen3.6 (same bridge)
- **Qwen3.6 35B-A3B** (`Qwen/Qwen3.6-35B-A3B`): 35B total parameters, 3B activated per token
  - 256 experts with top-8 routing and shared experts
  - 40 layers: 10 groups × (3 GDN + 1 Attention)
  - Uses `Qwen3_5MoeForConditionalGeneration` architecture — auto-detected by `AutoBridge`
  - Recommended: 1 node, 8 GPUs (EP=8)

## Examples

For checkpoint conversion, inference, finetuning recipes, and step-by-step training guides, see the [Qwen 3.5 Examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/qwen35_vl/README.md).

## Hugging Face Model Cards

- Qwen3.5 0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Qwen3.5 2B: https://huggingface.co/Qwen/Qwen3.5-2B
- Qwen3.5 4B: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3.5 9B: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3.5 27B: https://huggingface.co/Qwen/Qwen3.5-27B
- Qwen3.5 35B-A3B (MoE): https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5 122B-A10B (MoE): https://huggingface.co/Qwen/Qwen3.5-122B-A10B
- Qwen3.5 397B-A17B (MoE): https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Qwen3.6 35B-A3B (MoE): https://huggingface.co/Qwen/Qwen3.6-35B-A3B

## Related Docs
- Related VLM: [Qwen3-VL](qwen3-vl.md)
- Related LLM: [Qwen](../llm/qwen.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
