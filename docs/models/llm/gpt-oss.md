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

## Conversion with 🤗 Hugging Face

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: GPT OSS 20B
bridge = AutoBridge.from_hf_pretrained("openai/gpt-oss-20b")
provider = bridge.to_megatron_provider()

# Configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 4
provider.expert_model_parallel_size = 4
provider.sequence_parallel = True

model = provider.provide_distributed_model(wrap_with_ddp=False)

provider.finalize()

# For GPT OSS 120B:
# bridge = AutoBridge.from_hf_pretrained("openai/gpt-oss-120b")
```

### Export Megatron → HF
```python
# Convert from a Megatron checkpoint directory to HF format
bridge.export_ckpt(
    megatron_path="/results/gpt_oss_120b/checkpoints/iter_0500000",
    hf_path="./gpt-oss-hf-export",
)
```

## Examples

- Checkpoint conversion: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)

## Pretrain recipes

### Example usage (GPT OSS 20B)
```python
from megatron.bridge.recipes.gpt_oss import gpt_oss_20b_pretrain_config

cfg = gpt_oss_20b_pretrain_config(
    name="gpt_oss_20b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/gpt_oss_20b",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=4096,
)
```

### Example usage (GPT OSS 120B)
```python
from megatron.bridge.recipes.gpt_oss import gpt_oss_120b_pretrain_config

cfg = gpt_oss_120b_pretrain_config(
    name="gpt_oss_120b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/gpt_oss_120b",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=4096,
)
```

### Key configuration options
- **Parallelism (20B)**: Default TP=2, PP=4, EP=4 for efficient MoE training
- **Parallelism (120B)**: Default TP=2, PP=4, EP=16 for large-scale training
- **Sequence parallel**: Enabled by default for memory efficiency
- **Context parallel**: Supports CP for long sequences (131K tokens)
- **Manual GC**: Aggressive garbage collection (interval=100) for stable memory usage
- **MoE optimizations**: Grouped GEMM and permute fusion enabled by default

## Finetuning recipes

### Example usage (GPT OSS 20B - LoRA finetuning)
```python
from megatron.bridge.recipes.gpt_oss import gpt_oss_20b_finetune_config

cfg = gpt_oss_20b_finetune_config(
    hf_path="openai/gpt-oss-20b",
    name="gpt_oss_20b_lora_finetune",
    pretrained_checkpoint="path/to/gpt_oss/checkpoint",
    peft="lora",  # or "dora" for DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
)
```

### Example usage (GPT OSS 20B - Full SFT)
```python
cfg = gpt_oss_20b_finetune_config(
    hf_path="openai/gpt-oss-20b",
    name="gpt_oss_20b_full_sft",
    pretrained_checkpoint="path/to/gpt_oss/checkpoint",
    peft=None,  # Full supervised finetuning
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=5e-6,  # Lower LR for full SFT
)
```

### Example usage (GPT OSS 120B - LoRA finetuning)
```python
from megatron.bridge.recipes.gpt_oss import gpt_oss_120b_finetune_config

cfg = gpt_oss_120b_finetune_config(
    hf_path="openai/gpt-oss-120b",
    name="gpt_oss_120b_lora_finetune",
    pretrained_checkpoint="path/to/gpt_oss/checkpoint",
    peft="lora",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
)
```

### Default configurations

#### GPT OSS 20B

**LoRA/DoRA (1 node, 8 GPUs)**
- TP=1, PP=1, EP=1, LR=1e-4
- Optimized for parameter-efficient training
- Lower memory footprint

**Full SFT (1 node, 8 GPUs)**
- TP=1, PP=1, EP=8, LR=5e-6
- Full model training with expert parallelism
- Higher throughput with distributed experts

#### GPT OSS 120B

**LoRA/DoRA (2 nodes, 16 GPUs)**
- TP=1, PP=4, EP=8, LR=1e-4
- Optimized for parameter-efficient training
- Pipeline parallelism for memory efficiency

**Full SFT (2 nodes, 16 GPUs)**
- TP=1, PP=1, EP=8, LR=5e-6
- Full model training with expert parallelism
- Requires more memory per GPU

## API reference

- GPT OSS recipes: [bridge.recipes.gpt_oss](../../apidocs/bridge/bridge.recipes.gpt_oss.md)
- GPT OSS model provider: [bridge.models.gpt_oss.GPTOSSProvider](../../apidocs/bridge/bridge.models.gpt_oss.md)

## Performance optimizations

### Memory efficiency
- **Manual GC**: Aggressive garbage collection (interval=100) for stable memory usage
- **Precision-aware optimizer**: BF16 gradients and optimizer states
- **Expert parallelism**: Distributes experts across GPUs (EP=4 for 20B, EP=16 for 120B)
- **Sequence parallel**: Reduces activation memory across tensor parallel ranks
- **Context parallel**: Splits long sequences across multiple GPUs

### Compute efficiency
- **MoE permute fusion**: Fuses expert permutation operations
- **Grouped GEMM**: Optimized expert computation with grouped matrix multiplications
- **AllToAll dispatcher**: Efficient token routing across expert parallel ranks
- **Bias activation fusion**: Fuses bias addition with activation functions
- **Gradient overlapping**: Overlaps gradient all-reduce with backward computation

### Attention optimizations
- **Sliding window attention**: Reduces attention computation with 128-token windows
- **Window attention skip frequency**: Alternates between windowed and full attention every 2 layers
- **Sink attention**: Learnable softmax offsets for improved attention stability
- **Flash Attention**: FlashAttention-2 support via Transformer Engine
- **Activation clamping**: Prevents numerical instability with [-7.0, 7.0] clamping

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

