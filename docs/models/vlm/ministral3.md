# Ministral 3

[Mistral AI's Ministral 3](https://huggingface.co/collections/mistralai/ministral-3) is a family of edge-optimized vision-language models designed for deployment across various hardware configurations. The Ministral 3 architecture combines a powerful language model with a vision encoder for multimodal understanding.

Ministral 3 models support multimodal tasks including image captioning, visual question answering, OCR, and general vision-language understanding. Despite their compact size, these models deliver strong performance for on-device and edge deployment scenarios.

Ministral family models are supported via the Bridge system with auto-detected configuration and weight mapping.

```{important}
Please upgrade to `transformers` v5 and upgrade `mistral-common` in order to use the Ministral 3 models.
```

## Available Models

### Vision-Language Models
- **Ministral 3 3B** (`mistralai/Ministral-3-3B-Base-2512`): 3.4B parameter vision-language model
  - 26 layers, 3072 hidden size
  - 32 attention heads, 8 query groups (GQA)
  - Vision encoder: ~0.4B parameters
  - Recommended: 1 node, 8 GPUs

- **Ministral 3 8B** (`mistralai/Ministral-3-8B-Base-2512`): 8.4B parameter vision-language model
  - 34 layers, 4096 hidden size
  - 32 attention heads, 8 query groups (GQA)
  - Vision encoder: ~0.4B parameters
  - Recommended: 1 node, 8 GPUs

- **Ministral 3 14B** (`mistralai/Ministral-3-14B-Base-2512`): ~14B parameter vision-language model
  - 40 layers, 5120 hidden size
  - 32 attention heads, 8 query groups (GQA)
  - Vision encoder: ~0.4B parameters
  - Recommended: 1 node, 8 GPUs

All models support extended context lengths up to 256K tokens using YaRN RoPE scaling.

## Model Architecture Features

Ministral 3 combines efficient language modeling with multimodal capabilities:

**Language Model Features:**
- **YaRN RoPE Scaling**: Advanced rope scaling for extended context lengths (up to 256K tokens)
- **Grouped Query Attention (GQA)**: Memory-efficient attention mechanism with 8 query groups
- **SwiGLU Activation**: Gated linear units with SiLU activation for improved performance
- **RMSNorm**: Layer normalization without mean centering for faster computation
- **Llama 4 Attention Scaling**: Position-dependent attention scaling for improved long-context handling

**Vision-Language Features:**
- **Vision Encoder**: Pre-trained vision encoder for robust visual understanding
- **Multimodal Projector**: Projects vision features to language model space
- **Flexible Image Handling**: Supports variable resolution images and multiple images per conversation

## Examples

For checkpoint conversion, inference, finetuning recipes, and step-by-step training guides, see the [Ministral 3 Examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/ministral3/README.md).

## Hugging Face Model Cards

- Ministral 3 3B Base: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
- Ministral 3 3B Instruct: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- Ministral 3 8B Base: https://huggingface.co/mistralai/Ministral-3-8B-Base-2512
- Ministral 3 8B Instruct: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
- Ministral 3 14B Base: https://huggingface.co/mistralai/Ministral-3-14B-Base-2512
- Ministral 3 14B Instruct: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

## Related Docs
- Related LLM: [Mistral](../llm/mistral.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

