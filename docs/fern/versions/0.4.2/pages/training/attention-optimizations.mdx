# Attention Optimizations

Megatron Bridge provides several attention optimizations to improve the efficiency and performance of transformer models. These optimizations include Flash Attention for memory efficiency, and Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) for computational efficiency.

## Flash Attention

### Overview

Flash attention is an algorithm designed to improve the efficiency of the attention mechanism in transformer models such as GPT and BERT. The attention mechanism has quadratic time and memory complexity in sequence length and can present significant runtime and memory challenges for longer sequences.

Compared to the standard, non-flash algorithm, flash attention applies two techniques to lower the memory requirement and improve compute efficiency:

1. **Tiling technique**: Decomposes the inputs based on the shared memory size and calculates the softmax one tile at a time. Instead of working on the entire query, key, and value tensors at once, it makes several passes at these tensors and then combines the results in a subsequent step.

2. **Recomputation technique**: Stores the softmax normalization factors (linear to sequence length), instead of the softmax results (quadratic to sequence length), and uses these normalization factors to recompute the attention scores. This saves the amount of data to write to global memory and reduces both the I/O traffic between global memory and shared memory.

Flash attention lowers the memory footprint and computational complexity from quadratic to linear, greatly extending the range of sequence length allowed in large language models.

### Configure Flash Attention

In Megatron Bridge, flash attention is configured through the `attention_backend` parameter in your model configuration. The framework supports multiple attention backends through Transformer Engine integration:

```python
from megatron.bridge.models import GPTModelProvider
from megatron.core.transformer.enums import AttnBackend

# Configure model with flash attention (default)
model_config = GPTModelProvider(
    attention_backend=AttnBackend.auto,  # Let TE choose the best backend (default)
    # ... other model parameters
)

# Or explicitly specify flash attention
model_config = GPTModelProvider(
    attention_backend=AttnBackend.flash_attn,  # Explicitly use flash attention
    # ... other model parameters
)
```

### Attention Backend Options

Megatron Bridge supports several attention backends through the `attention_backend` configuration:

- `AttnBackend.auto`: Automatically selects the best available backend (recommended)
- `AttnBackend.flash_attn`: Explicitly use Flash Attention implementation
- `AttnBackend.fused_attn`: Use cuDNN fused attention (when available)
- `AttnBackend.local`: Use local PyTorch implementation (for debugging)

### Environment Variable Control

For fine-grained control, you can still use environment variables to disable specific implementations:

```bash
# Disable flash attention
export NVTE_FLASH_ATTN=0

# Disable cuDNN flash attention  
export NVTE_FUSED_ATTN=0
```

However, the recommended approach is to use the `attention_backend` configuration parameter.

## Multi-query Attention (MQA) and Grouped-query Attention (GQA)

**Multi-query Attention (MQA)** and **Grouped-query Attention (GQA)** are modifications of the traditional multihead attention mechanism in Transformer models. These methods improve the efficiency and effectiveness of attention mechanisms.

### Overview

**Multi-query Attention (MQA)**

MQA treats all attention heads as a single group, reducing computational complexity and accelerating training times. It is beneficial when model scalability or limited computational resources are concerns.

**Grouped-query Attention (GQA)**

GQA groups the heads into clusters, each processing a subset of queries independently. This method balances the detailed focus of traditional multihead attention with the broad approach of MQA, enhancing nuanced input data processing.

These attention variants offer:

- **Reduced computational load**: Both methods decrease computation, beneficial for large models
- **Increased processing speed**: Simplifying attention leads to faster training and inference
- **Flexibility and adaptability**: Adjustments can be made based on task needs or hardware constraints

### Enable MQA and GQA

To use MQA or GQA in Megatron Bridge, adjust the `num_query_groups` parameter in your model configuration:

#### Multi-query Attention (MQA)
Set `num_query_groups` to 1 to treat all attention heads as a single group:

```python
from megatron.bridge.models import GPTModelProvider

model_config = GPTModelProvider(
    num_attention_heads=32,
    num_query_groups=1,  # Enables Multi-query Attention
    # ... other model parameters
)
```

#### Grouped-query Attention (GQA)
Set `num_query_groups` to a number that is a divisor of the total number of attention heads (more than one but less than the total heads):

```python
model_config = GPTModelProvider(
    num_attention_heads=32,
    num_query_groups=8,  # Enables Grouped-query Attention (4 heads per group)
    # ... other model parameters
)
```

#### Regular Multihead Attention
For regular attention, set this parameter to `None` or match it with the number of heads:

```python
model_config = GPTModelProvider(
    num_attention_heads=32,
    num_query_groups=None,  # Default setting for regular multihead attention
    # Or equivalently:
    # num_query_groups=32,  # One group per head
    # ... other model parameters
)
```

## Resources

- [Megatron Core Attention Implementation](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Transformer Engine Attention Mechanisms](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html)
