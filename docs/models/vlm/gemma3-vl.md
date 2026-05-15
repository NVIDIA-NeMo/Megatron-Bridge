# Gemma 3 VL (Vision-Language)

[Google's Gemma 3 VL](https://huggingface.co/collections/google/gemma-3-release) is a family of vision-language models built on the same research and technology used to create Gemini models. The Gemma 3 VL architecture combines the text-generation capabilities of Gemma 3 with a SigLIP vision encoder for robust visual understanding.

Gemma 3 VL models support multimodal tasks including image captioning, visual question answering, OCR, and general vision-language understanding.

Gemma family models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

### Vision-Language Models
- **Gemma 3 VL 4B** (`google/gemma-3-4b-it`): 4B parameter vision-language model
  - 34 layers, 2560 hidden size
  - 16 attention heads, 4 query groups (GQA)
  - Vision encoder: SigLIP with 729M parameters
  - Recommended: 1 node, 8 GPUs
  
- **Gemma 3 VL 12B** (`google/gemma-3-12b-it`): 12B parameter vision-language model
  - 48 layers, 3840 hidden size
  - 24 attention heads, 8 query groups (GQA)
  - Vision encoder: SigLIP with 729M parameters
  - Recommended: 1 node, 8 GPUs
  
- **Gemma 3 VL 27B** (`google/gemma-3-27b-it`): 27B parameter vision-language model
  - 62 layers, 5376 hidden size
  - 32 attention heads, 16 query groups (GQA)
  - Vision encoder: SigLIP with 729M parameters
  - Recommended: 2 nodes, 16 GPUs

All models support a sequence length of 131,072 tokens and use hybrid attention patterns (sliding window + global).

## Model Architecture Features

Gemma 3 VL builds on the Gemma 3 architecture with additional multimodal capabilities:

**Language Model Features:**
- **Hybrid Attention Pattern**: Alternates between global and local sliding window attention for efficient long-context processing
- **GeGLU Activation**: Uses gated linear units with GELU activation for improved performance
- **RMSNorm**: Layer normalization without mean centering for faster computation
- **Rotary Embeddings**: Separate RoPE configurations for local and global attention layers

**Vision-Language Features:**
- **SigLIP Vision Encoder**: Pre-trained vision encoder with 729M parameters for robust visual understanding
- **Multimodal Integration**: Seamless integration of visual and textual information through learned projection layers
- **Flexible Image Handling**: Supports variable resolution images and multiple images per conversation

## Examples

For checkpoint conversion, inference, finetuning recipes, and step-by-step training guides, see the [Gemma 3 VL Examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/gemma3_vl/README.md).

## Hugging Face Model Cards

- Gemma 3 VL 4B: https://huggingface.co/google/gemma-3-4b-it
- Gemma 3 VL 12B: https://huggingface.co/google/gemma-3-12b-it
- Gemma 3 VL 27B: https://huggingface.co/google/gemma-3-27b-it

## Related Docs
- Text-Only Models: [Gemma 3](../llm/gemma3.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
