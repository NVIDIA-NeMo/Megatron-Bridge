# Multi-Token Prediction (MTP)

## Overview

Multi-Token Prediction (MTP) is an advanced training technique introduced in the [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) that enables models to predict multiple future tokens simultaneously during pre-training. Instead of learning to predict only the next token at each position, MTP adds auxiliary prediction heads that predict tokens 2, 3, or more positions ahead.

### Key Benefits

- **Densified Training Signals**: Multiple learning signals per training iteration improve data efficiency
- **Pre-Planning Representations**: Models learn internal representations that encode information about future tokens
- **Speculative Decoding Foundation**: MTP-trained models can serve as foundation for faster inference via speculative decoding

### When to Use MTP

MTP is most beneficial for:

- **Large-scale pre-training** (models > 10B parameters)
- **Data-constrained scenarios** where maximizing learning from limited data is critical
- **Training foundation models** intended for downstream fine-tuning or speculative decoding

MTP is primarily used for pre-training.

### Additional Resources

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Original paper introducing MTP
- [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3) - Official implementation
- [Megatron Core MTP API Guide](../3rdparty/Megatron-LM/docs/source/api-guide/multi_token_prediction.md) - Low-level implementation details

## Configuration Parameters

MTP is controlled by two primary parameters:

| Parameter | Type | Default | Description | Typical Range |
|-----------|------|---------|-------------|---------------|
| `mtp_num_layers` | int | `None` (disabled) | Number of auxiliary prediction depths. Each layer predicts tokens N positions ahead (N=1,2,...,mtp_num_layers). | 1-2 |
| `mtp_loss_scaling_factor` | float | `0.1` | Weight applied to MTP losses relative to main next-token loss. Controls the contribution of auxiliary predictions to the total loss. | 0.05-0.2 |

### Loss Calculation

The total training loss combines the main next-token prediction loss with averaged MTP losses:

```
total_loss = main_loss + (avg_mtp_loss * mtp_loss_scaling_factor)

where:
  avg_mtp_loss = mean([mtp_1_loss, mtp_2_loss, ..., mtp_N_loss])
```

### Parameter Tuning Guidelines

**`mtp_num_layers`:**
- Start with `1` for most models (predicts 1 token ahead)
- Use `2` for models > 100B parameters if memory allows
- Higher values increase memory usage and training time proportionally

**`mtp_loss_scaling_factor`:**
- Default `0.1` works well for most models
- Increase to `0.15-0.2` if MTP losses aren't decreasing
- Decrease to `0.05-0.08` if main loss is being overshadowed
- Scale factor should be proportional to `mtp_num_layers` (more layers → lower factor)

## Basic Usage: Training from Scratch

### Minimal Configuration Example

Here's a minimal example using the Qwen3-Next recipe with MTP enabled:

```python
from megatron.bridge.recipes.qwen import qwen3_next_80b_a3b_pretrain_config
from megatron.bridge.training.pretrain import pretrain

config = qwen3_next_80b_a3b_pretrain_config(
    name="qwen3_next_mtp_test",
    data_paths=["/path/to/data.nvjsonl"],

    # MTP Configuration
    mtp_num_layers=1,
    mtp_loss_scaling_factor=0.1,
)

pretrain(config)
```

### Command-Line Launch Example

You can also launch training using the `run_recipe.py` script with command-line arguments:

```bash
torchrun --nproc-per-node=8 scripts/training/run_recipe.py \
    --recipe qwen3_next_80b_a3b_pretrain_config \
    --name qwen3_next_mtp \
    --data_paths /path/to/data.nvjsonl \
    --mtp_num_layers 1 \
    --mtp_loss_scaling_factor 0.1
```

## MTP with Pipeline Parallelism

When using Pipeline Parallelism (PP), **MTP layers must be placed in the last pipeline stage** alongside the loss computation layer. Configure this using custom pipeline layout settings (`pipeline_model_parallel_split_rank`).

### Pipeline Layout Guidelines

MTP layers take approximately the same training time as a regular transformer layer. When configuring your pipeline layout:

- **Place MTP in the last PP stage** (required for correct loss computation)
- **Reduce layers in other PP ranks** to balance computation time across stages
- Example: For a 60-layer model with PP=4 and `mtp_num_layers=1`, you might use splits like `[15, 30, 45, 60]` instead of `[15, 30, 46, 60]` to account for MTP overhead in the last stage

### Verifying MTP Placement

During training, you'll see log messages confirming MTP placement:

```
[Rank 15] Building model pipeline stage 16/16
[Rank 15] Building Multi-Token Prediction layers: 1 depth(s)
```

On other pipeline stages, you'll see (this is expected):

```
[Rank 5] MTP layers not found on this PP rank (expected on last stage only)
```

## Parallelism Support

MTP is compatible with all major parallelism strategies in Megatron-Bridge:

| Parallelism Type | Support Status | Notes |
|------------------|----------------|-------|
| **Tensor Parallelism (TP)** | ✅ Fully Supported | MTP layers are automatically sharded across TP ranks |
| **Pipeline Parallelism (PP)** | ✅ Supported with Constraint | MTP must be in last pipeline stage (see above) |
| **Expert Parallelism (EP)** | ✅ Fully Supported | Works with MoE models (DeepSeek-V3, Mixtral, etc.) |
| **Context Parallelism (CP)** | ✅ Fully Supported | MTP supports long-context training via CP |
| **Data Parallelism (DP)** | ✅ Fully Supported | Standard data parallelism works transparently |

## Monitoring MTP Training

### Per-Layer Loss Logging

During training, you'll see losses for each MTP depth logged separately:

```
iteration:     100/100000 | consumed samples:        51200 | elapsed time per iteration (ms): 1247.3
loss: 3.245 | mtp_1 loss: 3.512 | learning rate: 1.500E-04 | grad norm: 0.847
```

For models with `mtp_num_layers=2`:

```
iteration:     100/100000 | consumed samples:        51200 | elapsed time per iteration (ms): 1389.7
loss: 3.245 | mtp_1 loss: 3.512 | mtp_2 loss: 3.689 | learning rate: 1.500E-04 | grad norm: 0.847
```

### Interpreting Loss Values

**Expected Patterns:**

- **MTP losses are higher than main loss**: Predicting tokens further in the future is harder
  - Main loss: 3.245
  - MTP 1 loss: 3.512 (+0.267)
  - MTP 2 loss: 3.689 (+0.444)

- **All losses decrease over training**: Both main and MTP losses should trend downward

- **Loss gap remains relatively stable**: The difference between main and MTP losses should not grow significantly

**Red Flags:**

- **NaN values**: Indicates training instability (see Troubleshooting section)
- **Diverging losses**: If MTP losses increase while main loss decreases, reduce `mtp_loss_scaling_factor`
- **Widening gap**: If MTP losses fall behind by > 1.0, increase `mtp_loss_scaling_factor`

### TensorBoard Visualization

MTP losses are automatically logged to TensorBoard and/or WandB. Look for:

- `train/loss` - Main next-token prediction loss
- `train/mtp_1_loss` - First auxiliary prediction loss
- `train/mtp_2_loss` - Second auxiliary prediction loss (if `mtp_num_layers=2`)
- `train/total_loss` - Combined loss (main + weighted MTP average)


## Training Curves & Expected Results

**[NOTE: Training curves will be added after running experiments]**
## TODO
This section will include visual comparisons of:

- MTP enabled vs. disabled training runs
- Main loss, MTP 1 loss, and MTP 2 loss trajectories over iterations
- Convergence behavior differences

### Expected Patterns

**Loss Curves:**

- MTP losses are typically higher than main loss (predicting future tokens is harder)
- All losses should decrease over training
- The gap between main loss and MTP losses generally remains stable

**Training Characteristics:**

- MTP adds computational overhead due to additional forward passes
- Memory usage increases proportionally to `mtp_num_layers`
- MTP is designed to improve data efficiency during pre-training

**Model Performance:**

- MTP provides additional training signals at each token position
- Can potentially improve downstream task performance
- MTP-trained models can be used for speculative decoding during inference

## Current Limitations

The following features are not yet supported with MTP:

| Feature | Status | Workaround |
|---------|--------|------------|
| **HuggingFace ↔ Megatron Checkpoint Conversion** | ⚠️ Model-specific | Conversion support varies by model; check model-specific documentation |
| **Sequence Packing (Fine-Tuning)** | ❌ Not supported | For pre-training, no issues. For fine-tuning, set `packed_sequence_specs=None` |
| **Cross-Attention** | ❌ Not supported | MTP only works with decoder-only models (GPT, Llama, etc.) |
| **Learned Absolute Position Embeddings** | ❌ Not supported | Use RoPE (rotary position embeddings) or no position embeddings |
| **Block-Based Activation Recomputation** | ❌ Not supported | Use `recompute_granularity="selective"` or `"uniform"` |

### Important Notes

**Checkpoint Conversion:**

HuggingFace ↔ Megatron checkpoint conversion with MTP is model-specific. Some models have conversion support planned, while others may not support MTP parameter mapping. Check the documentation for your specific model.

**Sequence Packing:**

MTP is incompatible with fine-tuning sequence packing (e.g., SFT with packed sequences). For pre-training, there are no sequence packing restrictions.

## Troubleshooting Guide

### Error: Out of Memory (OOM)

MTP increases memory usage proportionally to `mtp_num_layers`. Try:
- Reduce `mtp_num_layers` to 1
- Enable activation recomputation: `recompute_granularity="selective"`
- Increase pipeline parallelism
- Reduce micro batch size

### Error: MTP Loss is NaN

Training instability. Try:
- Lower learning rate
- Enable gradient clipping: `clip_grad=1.0`
- Use BF16 instead of FP16
- Reduce `mtp_loss_scaling_factor` to 0.05

### Expected Log: `MTP layers not found on this PP rank`

This is normal. Only the last pipeline stage builds MTP layers.

## Additional Resources

### Code Examples

- **DeepSeek-V3 Recipe**: [`src/megatron/bridge/recipes/deepseek/deepseek_v3.py`](/Users/chcui/PycharmProjects/Megatron-Bridge/src/megatron/bridge/recipes/deepseek/deepseek_v3.py)
  - Example of MTP with large-scale MoE model
  - Predefined pipeline layouts for PP + MTP

- **Qwen3-Next Recipe**: [`src/megatron/bridge/recipes/qwen/qwen3_next.py`](/Users/chcui/PycharmProjects/Megatron-Bridge/src/megatron/bridge/recipes/qwen/qwen3_next.py)
  - Clean example of MTP configuration for dense models
  - Good starting point for custom recipes

- **MTP Core Implementation**: [`3rdparty/Megatron-LM/megatron/core/transformer/multi_token_prediction.py`](/Users/chcui/PycharmProjects/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/multi_token_prediction.py)
  - Low-level MTP layer implementation
  - Loss computation and logging helpers

### Documentation

- **Megatron Core MTP API Guide**: [`3rdparty/Megatron-LM/docs/source/api-guide/multi_token_prediction.md`](/Users/chcui/PycharmProjects/Megatron-Bridge/3rdparty/Megatron-LM/docs/source/api-guide/multi_token_prediction.md)
  - Detailed API documentation
  - Implementation notes and design decisions

- **Pipeline Parallelism Guide**: [`docs/parallelisms.md`](/Users/chcui/PycharmProjects/Megatron-Bridge/docs/parallelisms.md)
  - Understanding pipeline parallelism layouts
  - Best practices for PP configuration

### External Resources

- **DeepSeek-V3 Technical Report**: [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
  - Original paper introducing MTP
  - Section 3.2: "Multi-Token Prediction"
  - Training details and ablation studies

- **DeepSeek-V3 GitHub**: [https://github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
  - Official implementation and model weights
  - Training configurations and hyperparameters

- **Megatron-LM GitHub**: [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  - Upstream Megatron-Core implementation
  - Issues and discussions

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [Megatron-Bridge GitHub Issues](https://github.com/NVIDIA/Megatron-Bridge/issues)
2. Review the [Megatron-LM Discussions](https://github.com/NVIDIA/Megatron-LM/discussions)

When reporting issues, include:
- Full training configuration (recipe and parameters)
- Error messages and stack traces
- GPU type and count
- Megatron-Core version (`pip show megatron-core`)
