# GPT OSS

GPT OSS is a Mixture-of-Experts (MoE) language model family featuring two variants: **GPT OSS 20B** and **GPT OSS 120B**. These models are designed with advanced attention mechanisms and MoE architectures optimized for long-context understanding.

The GPT OSS models feature decoder-only architectures with routed expert layers, supporting context lengths up to 128K tokens through YaRN position embeddings. Both variants use grouped-query attention and specialized attention mechanisms including sliding window attention with learnable softmax.

GPT OSS models are supported via the Bridge system with specialized configurations for MoE optimizations and long-context training.

## Model Architecture

### GPT OSS 20B
- **Parameters**: 20B total
- **Layers**: 24 decoder layers
- **Experts**: 32 routed experts per layer with top-4 routing
- **Hidden size**: 2880
- **FFN hidden size**: 2880 (dense layers), 2880 (expert layers)
- **Attention heads**: 64 query heads, 8 key-value groups (GQA)
- **KV channels**: 64
- **Vocab size**: 201,088
- **Context Length**: 128K tokens (via YaRN)
- **Activation**: QuickGELU with gated linear units
- **Normalization**: RMSNorm

### GPT OSS 120B
- **Parameters**: 120B total
- **Layers**: 36 decoder layers
- **Experts**: 128 routed experts per layer with top-4 routing
- **Hidden size**: 2880
- **FFN hidden size**: 2880 (dense layers), 2880 (expert layers)
- **Attention heads**: 64 query heads, 8 key-value groups (GQA)
- **KV channels**: 64
- **Vocab size**: 201,088
- **Context Length**: 128K tokens (via YaRN)
- **Activation**: QuickGELU with gated linear units
- **Normalization**: RMSNorm

## Key Features

- **YaRN Position Embeddings**: Advanced rotary position embeddings with scaling factor 32.0 for long-context extension
- **Grouped-Query Attention (GQA)**: Efficient attention with 8 key-value groups
- **Sliding Window Attention**: Window size of 128 tokens with alternating full/windowed attention pattern
- **Learnable Softmax**: Novel softmax implementation with learnable offset parameters (sink attention)
- **QuickGELU Activation**: Fast approximate GELU with clamping at 7.0 for stability
- **MoE Routing**: Top-4 expert routing without load balancing loss
- **Grouped GEMM**: Optimized grouped matrix multiplications for expert computation
- **Bias in Linear Layers**: Linear layers include bias terms
- **Activation Clamping**: Output activations clamped to [-7.0, 7.0] for numerical stability

## Examples

For checkpoint conversion, inference, finetuning recipes, and step-by-step training guides, see the [GPT-OSS Examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/gpt_oss/README.md).

## API reference

- GPT OSS recipes: [bridge.recipes.gpt_oss](../../apidocs/bridge/bridge.recipes.gpt_oss.md)
- GPT OSS model provider: [bridge.models.gpt_oss.GPTOSSProvider](../../apidocs/bridge/bridge.models.gpt_oss.md)

## Hugging Face model cards

### GPT OSS 20B
- Base: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

### GPT OSS 120B
- Base: [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)

## Related docs

- Recipe usage and customization: [Recipe usage](../../recipe-usage.md)
- Training configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
- Attention optimizations: [Attention optimizations](../../training/attention-optimizations.md)
