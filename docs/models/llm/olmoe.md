# OLMoE

[OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0125) is a 7B-parameter Mixture-of-Experts (MoE) model from **Allen Institute for AI (AI2)** featuring 64 experts with top-8 routing. The model is designed to be fully open-source, with training data, code, and model weights publicly available. It's named "OLMoE-1B-7B" where 1B refers to the activated parameters and 7B refers to the total parameters.

The latest version (OLMoE-1B-7B-0125, released January 2025) is an improved version of the original September 2024 release (OLMoE-1B-7B-0924), trained on 5T tokens with performance improvements across multiple benchmarks.

The model features 16 decoder layers with 64 routed experts per layer, activating 8 experts per token for a total of approximately 1.3B active parameters per forward pass out of 7B total.

OLMoE models are supported via the Bridge system with specialized configurations for MoE optimizations.

## Model Architecture

- **Parameters**: 7B total, 1.3B activated per forward pass
- **Layers**: 16 decoder layers
- **Attention**: Multi-query attention with QK LayerNorm and RoPE
- **MoE**: 64 routed experts per layer with top-8 routing
- **Hidden size**: 2048
- **FFN hidden size**: 1024 (dense layers), 1024 (expert layers)
- **Attention heads**: 16 query heads, 16 key-value groups
- **Vocab size**: 50,304
- **Context Length**: 4K tokens
- **Activation**: SiLU with gated linear units
- **Training**: 5T tokens (OLMoE-1B-7B-0125)

## Key Features

- **QK LayerNorm**: Applies LayerNorm to query and key projections for training stability
- **RoPE**: Rotary Position Embeddings with base 10000
- **MoE Routing**: Softmax-based router with auxiliary loss for load balancing
- **Router Pre-Softmax**: Pre-softmax routing scores
- **Grouped GEMM**: Optimized grouped matrix multiplications for expert computation

## Conversion with 🤗 Hugging Face

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: OLMoE-1B-7B-0125 (latest version)
bridge = AutoBridge.from_hf_pretrained("allenai/OLMoE-1B-7B-0125")
provider = bridge.to_megatron_provider()

# Configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8
provider.sequence_parallel = False

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
# You can also use older versions:
# bridge = AutoBridge.from_hf_pretrained("allenai/OLMoE-1B-7B-0924")
```

### Export Megatron → HF
```python
# Convert from a Megatron checkpoint directory to HF format
bridge.export_ckpt(
    megatron_path="/results/olmoe_7b/checkpoints/iter_0500000",
    hf_path="./olmoe-hf-export",
)
```

## Examples

- Checkpoint conversion: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)

## Pretrain recipes

### Example usage (OLMoE-7B)
```python
from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

cfg = olmoe_7b_pretrain_config(
    name="olmoe_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/olmoe_7b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
)
```

### Key configuration options
- **Parallelism**: Default TP=1, PP=1, EP=8 for efficient MoE training
- **Sequence parallel**: Disabled by default (can be enabled with larger TP)
- **Recomputation**: Selective recomputation for memory optimization
- **RoPE fusion**: Optional optimization (`apply_rope_fusion=True`)
- **MoE optimizations**: Grouped GEMM and permute fusion enabled by default

## Finetuning recipes

### Example usage (LoRA finetuning)
```python
from megatron.bridge.recipes.olmoe import olmoe_7b_finetune_config

cfg = olmoe_7b_finetune_config(
    tokenizer_path="allenai/OLMoE-1B-7B-0125",
    name="olmoe_lora_finetune",
    pretrained_checkpoint="path/to/olmoe/checkpoint",
    peft="lora",  # or "dora" for DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
)
```

### Example usage (Full SFT)
```python
cfg = olmoe_7b_finetune_config(
    tokenizer_path="allenai/OLMoE-1B-7B-0125",
    name="olmoe_full_sft",
    pretrained_checkpoint="path/to/olmoe/checkpoint",
    peft=None,  # Full supervised finetuning
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=5e-6,  # Lower LR for full SFT
)
```

### Default configurations

#### LoRA/DoRA (1 node, 8 GPUs)
- TP=1, PP=1, EP=1, LR=1e-4
- Optimized for parameter-efficient training
- Lower memory footprint

#### Full SFT (1 node, 8 GPUs)
- TP=1, PP=1, EP=8, LR=5e-6
- Full model training with expert parallelism
- Higher throughput with distributed experts

## API reference

- OLMoE recipes: [bridge.recipes.olmoe](../../apidocs/bridge/bridge.recipes.olmoe.md)
- OLMoE model provider: [bridge.models.olmoe.OlMoEModelProvider](../../apidocs/bridge/bridge.models.olmoe.md)

## Performance optimizations

### Memory efficiency
- **Selective recomputation**: Reduces activation memory by recomputing during backward pass
- **Manual GC**: Aggressive garbage collection (interval=5) for stable memory usage
- **Precision-aware optimizer**: BF16 gradients and optimizer states with FP32 master weights
- **Expert parallelism**: Distributes experts across GPUs (EP=8 recommended)

### Compute efficiency
- **MoE permute fusion**: Fuses expert permutation operations
- **Grouped GEMM**: Optimized expert computation with grouped matrix multiplications
- **Router load balancing**: Auxiliary loss for balanced expert utilization
- **AllToAll dispatcher**: Efficient token routing across expert parallel ranks

## Pipeline parallelism layouts

OLMoE (7B) supports several PP configurations with pre-defined symmetric layouts:
- **PP=1**: No pipelining (default)
- **PP=2**: 8+8 layer split with embedding/loss
- **PP=4**: 4+4+4+4 layer split
- **VP (Virtual Pipeline)**: PP=2,VP=2 supported

## Hugging Face model cards

### Latest (January 2025)
- OLMoE-1B-7B-0125 (Base): [allenai/OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
- OLMoE-1B-7B-0125-SFT: [allenai/OLMoE-1B-7B-0125-SFT](https://huggingface.co/allenai/OLMoE-1B-7B-0125-SFT)
- OLMoE-1B-7B-0125-Instruct: [allenai/OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)

### Previous (September 2024)
- OLMoE-1B-7B-0924 (Base): [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- OLMoE-1B-7B-0924-Instruct: [allenai/OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)

## Technical Resources

- OLMoE Paper: [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
- OLMoE Model Card (Latest): [HuggingFace Model Card](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
- OLMoE GitHub Repository: [allenai/OLMoE](https://github.com/allenai/OLMoE)

## Related docs

- Recipe usage and customization: [Recipe usage](../../recipe-usage.md)
- Training configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

