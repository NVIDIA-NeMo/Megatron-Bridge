# Moonlight

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) is a 16B-parameter Mixture-of-Experts (MoE) model from **Moonshot AI** trained with 5.7T tokens using the innovative **Muon optimizer**. While Moonlight shares the same architecture as DeepSeek-V3 (featuring Multi-head Latent Attention and MoE), it is a distinct model that advances the Pareto frontier of performance vs training FLOPs through the use of Muon, which is ~2× more sample efficient than Adam with compute optimal training.

The model features 27 decoder layers with 64 routed experts and 8 shared experts per layer, with 3B activated parameters per forward pass out of 16B total parameters.

Moonlight models are supported via the Bridge system with specialized configurations for MoE and MLA optimizations.

## Model Architecture

- **Parameters**: 16B total, 3B activated per forward pass
- **Layers**: 27 decoder layers
- **Attention**: Multi-head Latent Attention (MLA) with RoPE fusion support
- **MoE**: 64 routed experts + 8 shared experts per layer
- **Hidden size**: 2048
- **Intermediate size**: 10944 (with MLP and expert gating)
- **Vocab size**: 151,936
- **Context Length**: 8K tokens
- **Training**: 5.7T tokens with Muon optimizer

## Conversion with 🤗 Hugging Face

Moonlight shares the same architecture as DeepSeek-V3, which enables compatibility with various inference engines like vLLM and SGLang. The model can be loaded from HuggingFace or used with Megatron checkpoints.

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: Moonlight-16B-A3B
bridge = AutoBridge.from_hf_pretrained("moonshotai/Moonlight-16B-A3B")
provider = bridge.to_megatron_provider()

# Configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Export Megatron → HF
```python
# Convert from a Megatron checkpoint directory to HF format
bridge.export_ckpt(
    megatron_path="/results/moonlight_16b/checkpoints/iter_0500000",
    hf_path="./moonlight-hf-export",
)
```

## Examples

- Checkpoint conversion: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)

## Pretrain recipes

### Example usage (Moonlight-16B)
```python
from megatron.bridge.recipes.moonlight import moonlight_16b_pretrain_config

cfg = moonlight_16b_pretrain_config(
    name="moonlight_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/moonlight_16b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
)
```

### Key configuration options
- **Parallelism**: Default TP=2, PP=1, EP=8 for efficient MoE training
- **Sequence parallel**: Enabled by default for memory efficiency
- **Recomputation**: Selective recomputation for memory optimization
- **RoPE fusion**: Optional MLA-specific optimization (`apply_rope_fusion=True`)
- **DeePEP**: Optional expert permutation optimization (`enable_deepep=True`)

## Finetuning recipes

### Example usage (LoRA finetuning)
```python
from megatron.bridge.recipes.moonlight import moonlight_16b_finetune_config

cfg = moonlight_16b_finetune_config(
    tokenizer_path="moonshotai/Moonlight-16B-A3B",
    name="moonlight_lora_finetune",
    pretrained_checkpoint="/results/moonlight_16b/checkpoints/iter_0500000",
    peft="lora",  # or "dora" for DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
)
```

### Example usage (Full SFT)
```python
cfg = moonlight_16b_finetune_config(
    tokenizer_path="moonshotai/Moonlight-16B-A3B",
    name="moonlight_full_sft",
    pretrained_checkpoint="/results/moonlight_16b/checkpoints/iter_0500000",
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
- TP=2, PP=1, EP=8, LR=5e-6
- Full model training with expert parallelism
- Higher throughput with distributed experts

## API reference

- Moonlight recipes: [bridge.recipes.moonlight](../../apidocs/bridge/bridge.recipes.moonlight.md)
- Moonlight model provider: [bridge.models.deepseek.MoonlightModelProvider16B](../../apidocs/bridge/bridge.models.deepseek.md)

## Performance optimizations

### Memory efficiency
- **Selective recomputation**: Reduces activation memory by recomputing during backward pass
- **Sequence parallel**: Distributes sequence dimension across GPUs
- **Manual GC**: Aggressive garbage collection (interval=5) for stable memory usage
- **Precision-aware optimizer**: BF16 gradients and optimizer states with FP32 master weights

### Compute efficiency
- **MoE permute fusion**: Fuses expert permutation operations
- **RoPE fusion**: Optional fusion for Multi-head Latent Attention
- **Expert parallelism**: Distributes experts across GPUs (EP=8 recommended)
- **Pipeline layouts**: Asymmetric PP layouts for balanced load (PP=2,4,8 supported)

## Pipeline parallelism layouts

Moonlight supports several PP configurations with pre-defined asymmetric layouts:
- **PP=1**: No pipelining (default)
- **PP=2**: 14+13 layer split with embedding/loss
- **PP=4**: 8+7+7+6 layer split
- **PP=8**: 5+4+4+4+4+4+4+4 layer split
- **VP (Virtual Pipeline)**: PP=2,VP=2 and PP=4,VP=2 supported

## Hugging Face model cards

- Moonlight-16B-A3B (Base): [moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
- Moonlight-16B-A3B-Instruct: [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct)

## Technical Paper

- Muon is Scalable for LLM Training: [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

## Related docs

- Recipe usage and customization: [Recipe usage](../../recipe-usage.md)
- Training configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

