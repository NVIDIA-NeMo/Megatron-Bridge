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

## Dual Chunk Attention

Dual Chunk Attention (DCA) extends the effective context of RoPE-based GPT models by splitting a long sequence into fixed-size chunks. For each query chunk, DCA combines causal attention within the current chunk, attention to the immediately preceding chunk, and attention to all earlier chunks. The partial outputs are merged with log-sum-exp normalization so they are equivalent to a single softmax over the attended key partitions.

The Bridge implementation is based on the Megatron-LM DCA implementation proposed in [NVIDIA/Megatron-LM#4048](https://github.com/NVIDIA/Megatron-LM/pull/4048). The generic attention implementation lives under `megatron.bridge.models.transformer`; GPT-specific configuration and layer-spec wiring live under `megatron.bridge.models.gpt`.

### Configure DCA

The modern builder path uses `DualChunkGPTModelConfig` with an explicit `DualChunkTransformerConfig`:

```python
from megatron.bridge.models.gpt.dca import DualChunkGPTModelConfig
from megatron.bridge.models.transformer.dca import DualChunkTransformerConfig

transformer_config = DualChunkTransformerConfig(
    num_layers=2,
    hidden_size=1024,
    num_attention_heads=16,
    num_query_groups=8,
    transformer_impl="transformer_engine",
    apply_rope_fusion=False,
    dca_chunk_size=8192,
    dca_local_size=1024,
)

model_config = DualChunkGPTModelConfig(
    transformer=transformer_config,
    vocab_size=32000,
    seq_length=16384,
    position_embedding_type="rope",
)
```

The legacy provider path exposes the same configuration directly:

```python
from megatron.bridge.models.gpt.dca import DualChunkGPTModelProvider

model_config = DualChunkGPTModelProvider(
    num_layers=2,
    hidden_size=1024,
    num_attention_heads=16,
    num_query_groups=8,
    vocab_size=32000,
    seq_length=16384,
    position_embedding_type="rope",
    transformer_impl="transformer_engine",
    apply_rope_fusion=False,
    dca_chunk_size=8192,
    dca_local_size=1024,
)
```

The effective chunk length is `dca_chunk_size - dca_local_size`. FlashAttention is used for CUDA FP16/BF16 inputs when `flash-attn` is available; otherwise DCA uses an unfused PyTorch reference path. Because FlashAttention 2.x does not propagate gradients through its returned log-sum-exp tensor, long-sequence DCA uses an auxiliary FlashAttention pass to preserve the exact Q/K normalization gradients without materializing the attention score matrix. Long-sequence gradient-enabled execution with a head dimension of 256 falls back to the unfused path because the auxiliary pass needs one additional dimension. Selective core-attention recompute is supported.

For YARN, configure `yarn_rotary_scaling_factor`, `yarn_original_max_position_embeddings`, `yarn_beta_fast`, `yarn_beta_slow`, and `yarn_correction_range_round_to_int` on `DualChunkTransformerConfig`. The optional `yarn_mscale` and `yarn_mscale_all_dim` fields control the attention concentration factor.

Current DCA support is training-only. It requires RoPE or YARN on every decoder layer, `context_parallel_size=1`, backend-generated causal masks, and unpacked sequences. Inference/KV cache, MTP, sliding-window attention, CUDA graphs, fine-grained activation offloading, full Transformer Engine layer specs, and ModelOpt restore are rejected explicitly.

### Correctness validation

The DCA unit tests compare long-sequence outputs and Q/K/V gradients with a direct token-level implementation that applies one global softmax over all three DCA key partitions. They also verify exact log-sum-exp partition merging in float64, short-sequence parity with standard causal attention, causal independence from future tokens, and CUDA FP16 output and Q/K/V gradient parity between FlashAttention and the unfused path for both multi-head and grouped-query attention.

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
- [Dual Chunk Attention Paper](https://arxiv.org/abs/2402.17463)
- [Megatron-LM DCA Reference Implementation](https://github.com/NVIDIA/Megatron-LM/pull/4048)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Transformer Engine Attention Mechanisms](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html)
