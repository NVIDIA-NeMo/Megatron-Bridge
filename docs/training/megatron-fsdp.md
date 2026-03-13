# Megatron FSDP

Megatron Fully Sharded Data Parallel (FSDP) is a memory-optimized data parallelism strategy that shards model parameters, gradients, and optimizer states across GPUs. This approach provides significant memory savings compared to standard Distributed Data Parallel (DDP), enabling training of larger models or using larger batch sizes on the same hardware.

## Overview

FSDP reduces memory consumption by:
- **Sharding model parameters** across data parallel ranks instead of replicating them
- **Sharding optimizer states** across data parallel ranks
- **Sharding gradients** across data parallel ranks
- **Gathering parameters on-demand** during forward and backward passes

This strategy is particularly effective for large models where parameter and optimizer state memory dominates GPU memory usage.

## Comparison with Other Strategies

| Feature | DDP | Distributed Optimizer | Megatron FSDP |
|---------|-----|----------------------|---------------|
| **Parameter Storage** | Replicated | Replicated | Sharded |
| **Optimizer States** | Replicated | Sharded | Sharded |
| **Gradient Communication** | All-reduce | Reduce-scatter | Reduce-scatter |
| **Parameter Communication** | None | All-gather (after update) | All-gather (on-demand) |
| **Memory Efficiency** | Baseline | High | Highest |
| **Communication Overhead** | Low | Medium | Medium-High |

**When to use each strategy:**
- **DDP**: Default choice for most training scenarios with sufficient memory
- **Distributed Optimizer**: Good balance of memory savings and performance
- **Megatron FSDP**: Maximum memory savings when training very large models

## Configuration

### Enable Megatron FSDP

To enable Megatron FSDP, set `use_megatron_fsdp=True` in both the {py:class}`bridge.training.config.DistributedInitConfig` and {py:class}`bridge.training.config.DistributedDataParallelConfig`:

```python
from megatron.bridge.training.config import (
    ConfigContainer,
    DistributedInitConfig,
    DistributedDataParallelConfig,
    CheckpointConfig,
)

# Enable Megatron FSDP
dist_config = DistributedInitConfig(
    use_megatron_fsdp=True,
)

ddp_config = DistributedDataParallelConfig(
    use_megatron_fsdp=True,
)

# Required checkpoint format
checkpoint_config = CheckpointConfig(
    ckpt_format="fsdp_dtensor",
    save="/path/to/checkpoints",
)

config = ConfigContainer(
    dist=dist_config,
    ddp=ddp_config,
    checkpoint=checkpoint_config,
    # ... other config parameters
)
```

### Checkpoint Format Requirement

**Important**: Megatron FSDP requires the `fsdp_dtensor` checkpoint format. This format is specifically designed to handle the sharded parameter layout used by FSDP.

```python
checkpoint_config = CheckpointConfig(
    ckpt_format="fsdp_dtensor",  # Required for Megatron FSDP
    save="/path/to/checkpoints",
    load="/path/to/checkpoints",  # Optional: resume from checkpoint
)
```

Attempting to use other checkpoint formats (`torch_dist`, `zarr`) with Megatron FSDP will result in a configuration error.

## Compatibility and Limitations

### Compatible With
- **Tensor Parallelism**: Can be combined with TP for additional memory savings
- **Pipeline Parallelism**: Can be combined with PP for multi-stage model training
- **Context Parallelism**: Can be combined with CP for long sequence training
- **Expert Parallelism**: Can be combined with EP for MoE models
- **Mixed Precision**: Supports BF16 and FP16 training
- **Distributed Checkpointing**: Uses `fsdp_dtensor` format

### Not Compatible With
- **`use_tp_pp_dp_mapping`**: This alternative rank initialization order conflicts with FSDP's sharding strategy
- **`reuse_grad_buf_for_mxfp8_param_ag`**: Gradient buffer reuse optimizations are disabled with FSDP
- **Legacy checkpoint formats**: Must use `fsdp_dtensor` format

### Automatic Configuration Adjustments

When Megatron FSDP is enabled, the following settings are automatically adjusted:
- `ddp.average_in_collective` is set to `False` (FSDP handles gradient synchronization differently)
- `model.gradient_accumulation_fusion` is set to `False` (not compatible with FSDP)
- `ddp.reuse_grad_buf_for_mxfp8_param_ag` is set to `False` (disabled for FSDP)
- `optimizer.reuse_grad_buf_for_mxfp8_param_ag` is set to `False` (disabled for FSDP)

## Complete Configuration Example

Here's a complete example showing how to configure training with Megatron FSDP:

```python
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training.config import (
    ConfigContainer,
    DistributedInitConfig,
    DistributedDataParallelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    CheckpointConfig,
)

# Model configuration with tensor parallelism
model_config = GPTModelProvider(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    seq_length=2048,
    tensor_model_parallel_size=2,  # Optional: combine with TP
    # ... other model parameters
)

# Enable Megatron FSDP
dist_config = DistributedInitConfig(
    use_megatron_fsdp=True,
)

ddp_config = DistributedDataParallelConfig(
    use_megatron_fsdp=True,
)

# Optimizer configuration
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=3e-4,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    clip_grad=1.0,
)

# Scheduler configuration
scheduler_config = SchedulerConfig(
    lr_decay_style="cosine",
    lr_warmup_iters=1000,
)

# Training configuration
train_config = TrainingConfig(
    micro_batch_size=2,
    global_batch_size=32,
    train_iters=10000,
)

# Checkpoint configuration with required format
checkpoint_config = CheckpointConfig(
    save="/path/to/checkpoints",
    save_interval=1000,
    ckpt_format="fsdp_dtensor",  # Required for FSDP
)

# Create complete configuration
config = ConfigContainer(
    model=model_config,
    dist=dist_config,
    ddp=ddp_config,
    optimizer=optimizer_config,
    scheduler=scheduler_config,
    train=train_config,
    checkpoint=checkpoint_config,
    # ... other config parameters
)
```

## Migration from DDP

To migrate from standard DDP to Megatron FSDP:

1. **Enable FSDP** in both `dist` and `ddp` configurations:
   ```python
   dist_config.use_megatron_fsdp = True
   ddp_config.use_megatron_fsdp = True
   ```

2. **Change checkpoint format** to `fsdp_dtensor`:
   ```python
   checkpoint_config.ckpt_format = "fsdp_dtensor"
   ```

3. **Remove incompatible settings**:
   - Remove `use_tp_pp_dp_mapping=True` if set
   - Remove `reuse_grad_buf_for_mxfp8_param_ag=True` if set

4. **Start training** - automatic configuration adjustments will be applied

**Note**: Checkpoints saved with DDP cannot be directly loaded with FSDP due to different parameter layouts. You'll need to restart training or convert checkpoints using the appropriate conversion tools.

## Torch FSDP2 Alternative

Megatron Bridge also supports an alternative FSDP implementation through PyTorch's FSDP2:

```python
dist_config = DistributedInitConfig(
    use_torch_fsdp2=True,  # Use PyTorch FSDP2 instead
)
```

**Important**: `use_megatron_fsdp` and `use_torch_fsdp2` are mutually exclusive - you can only enable one at a time.

**Limitations of Torch FSDP2**:
- Not currently compatible with Pipeline Parallelism
- Still in experimental stage with potential bugs
- Does not require `fsdp_dtensor` checkpoint format

## Performance Considerations

### Memory Savings
- **Parameters**: Reduced by factor of data parallel size
- **Optimizer States**: Reduced by factor of data parallel size
- **Gradients**: Reduced by factor of data parallel size
- **Activations**: Not affected by FSDP (use activation checkpointing separately)

### Communication Overhead
- **Parameter All-Gather**: Additional communication during forward and backward passes
- **Gradient Reduce-Scatter**: Similar to distributed optimizer
- **Network Sensitivity**: Performance depends on inter-GPU bandwidth

### Optimization Tips
1. **Combine with Tensor Parallelism**: Reduces memory further and improves compute efficiency
2. **Use Larger Batch Sizes**: Take advantage of freed memory for better throughput
3. **High-Bandwidth Interconnects**: FSDP benefits from fast inter-GPU communication (NVLink, InfiniBand)
4. **Enable Mixed Precision**: Reduces communication volume and memory footprint

## Troubleshooting

### Configuration Errors

**Error: "use_tp_pp_dp_mapping is not supported with Megatron FSDP"**
- Remove `use_tp_pp_dp_mapping=True` from your configuration
- FSDP requires standard rank initialization order

**Error: "Megatron FSDP only supports fsdp_dtensor checkpoint format"**
- Set `checkpoint.ckpt_format="fsdp_dtensor"` in your configuration
- Other formats are not compatible with FSDP's sharded layout

**Error: "Using use_megatron_fsdp and use_torch_fsdp2 at the same time is not supported"**
- Choose one FSDP implementation: either Megatron FSDP or Torch FSDP2
- Do not enable both simultaneously

### Performance Issues

**Slow Training with FSDP**
- Check inter-GPU bandwidth (use `nvidia-smi topo -m`)
- Ensure NVLink or high-speed interconnects are available
- Consider using larger micro-batch sizes to amortize communication
- Profile with `nsys` or PyTorch profiler to identify bottlenecks

**Out of Memory with FSDP Enabled**
- Verify FSDP is correctly enabled in both `dist` and `ddp` configs
- Check that `fsdp_dtensor` checkpoint format is being used
- Reduce micro-batch size or model size
- Enable activation checkpointing for additional memory savings

## Resources

- {doc}`checkpointing` - Checkpoint saving and loading with FSDP
- {doc}`../parallelisms` - Understanding data and model parallelism strategies
- {doc}`config-container-overview` - Complete configuration reference
- [Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
