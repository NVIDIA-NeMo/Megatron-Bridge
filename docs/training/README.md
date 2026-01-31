# Training and Customization

This directory contains comprehensive documentation for training and customizing models with Megatron Bridge. Learn how to configure training, optimize performance, and customize training workflows.

## Quick Navigation

### I want to

**🚀 Get started with training**
→ Start with [Configuration Container Overview](config-container-overview.md) to understand the training setup

**⚙️ Configure training parameters**
→ See [Training Loop Settings](training-loop-settings.md) and [Optimizer & Scheduler](optimizer-scheduler.md)

**📊 Monitor and profile training**
→ Check [Logging](logging.md) and [Profiling](profiling.md) guides

**💾 Manage checkpoints**
→ Read [Checkpointing](checkpointing.md) for saving and resuming training

**⚡ Optimize performance**
→ Explore [Performance Guide](../performance-guide.md) and [Performance Summary](../performance-summary.md)

**🔧 Customize training**
→ See [PEFT](peft.md), [Distillation](distillation.md), [Entry Points](entry-points.md), and [Callbacks](callbacks.md)

## Core Training Documentation

### Configuration and Setup

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[Configuration Container Overview](config-container-overview.md)** | Central configuration object for all training settings | First time setting up training |
| **[Entry Points](entry-points.md)** | Training entry points and execution flow | Understanding how training starts |
| **[Training Loop Settings](training-loop-settings.md)** | Training loop parameters and configuration | Configuring batch sizes, iterations, validation |

### Optimization and Performance

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[Optimizer & Scheduler](optimizer-scheduler.md)** | Optimizer and learning rate scheduler configuration | Setting up optimization |
| **[Mixed Precision](mixed-precision.md)** | Mixed precision training for memory efficiency | Reducing memory usage |
| **[Communication Overlap](communication-overlap.md)** | Overlapping communication with computation | Optimizing distributed training |
| **[Attention Optimizations](attention-optimizations.md)** | Optimizing attention mechanisms | Improving training speed |
| **[Activation Recomputation](activation-recomputation.md)** | Gradient checkpointing strategies | Reducing memory footprint |
| **[CPU Offloading](cpu-offloading.md)** | Offloading to CPU for memory management | Working with limited GPU memory |

### Monitoring and Debugging

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[Logging](logging.md)** | Logging configuration and TensorBoard/WandB integration | Monitoring training progress |
| **[Profiling](profiling.md)** | Performance profiling and analysis | Identifying bottlenecks |
| **[Resiliency](resiliency.md)** | Handling failures and recovery | Building robust training pipelines |

### Advanced Features

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[PEFT](peft.md)** | Parameter-Efficient Fine-Tuning (LoRA, etc.) | Fine-tuning with limited resources |
| **[Packed Sequences](packed-sequences.md)** | Sequence packing for efficiency | Optimizing data loading |
| **[Distillation](distillation.md)** | Knowledge distillation techniques | Transferring knowledge between models |
| **[Checkpointing](checkpointing.md)** | Checkpoint saving, loading, and resuming | Managing training state |
| **[Callbacks](callbacks.md)** | Inject custom logic into training loop | Custom logging, metrics, third-party integrations |

## Training Workflow

A typical training workflow involves:

1. **Configure Training** - Set up `ConfigContainer` with model, data, and training parameters
2. **Prepare Data** - Configure dataset loading and preprocessing
3. **Set Optimization** - Configure optimizer, scheduler, and mixed precision
4. **Enable Monitoring** - Set up logging and profiling
5. **Configure Checkpointing** - Set up checkpoint saving and resuming
6. **Launch Training** - Start training with configured entry points
7. **Monitor Progress** - Track metrics via logging and profiling
8. **Resume if Needed** - Use checkpointing to resume from saved state

## Related Documentation

- **[Main Documentation Index](../index.md)** - Return to main documentation
- **[Performance Guide](../performance-guide.md)** - Comprehensive performance optimization guide
- **[Performance Summary](../performance-summary.md)** - Quick performance reference
- **[Recipe Usage](../recipe-usage.md)** - Using training recipes
- **[Parallelisms](../parallelisms.md)** - Understanding distributed training strategies
- **[Bridge Guide](../bridge-guide.md)** - Working with Hugging Face models

## Common Training Scenarios

### 🆕 First-Time Training Setup

1. [Configuration Container Overview](config-container-overview.md) - Understand the configuration system
2. [Entry Points](entry-points.md) - Learn how to start training
3. [Training Loop Settings](training-loop-settings.md) - Configure basic training parameters
4. [Logging](logging.md) - Set up monitoring

### ⚡ Performance Optimization

1. [Performance Guide](../performance-guide.md) - Comprehensive optimization strategies
2. [Mixed Precision](mixed-precision.md) - Enable mixed precision training
3. [Communication Overlap](communication-overlap.md) - Optimize distributed training
4. [Activation Recomputation](activation-recomputation.md) - Reduce memory usage
5. [Profiling](profiling.md) - Identify bottlenecks

### 💾 Production Training

1. [Checkpointing](checkpointing.md) - Reliable checkpoint management
2. [Resiliency](resiliency.md) - Handle failures gracefully
3. [Logging](logging.md) - Comprehensive monitoring
4. [Profiling](profiling.md) - Performance analysis

### 🔧 Customization

1. [PEFT](peft.md) - Parameter-efficient fine-tuning
2. [Distillation](distillation.md) - Knowledge distillation
3. [Entry Points](entry-points.md) - Custom training workflows
4. [Callbacks](callbacks.md) - Inject custom logic (third-party integrations)

---

**Ready to start training?** Begin with [Configuration Container Overview](config-container-overview.md) or return to the [main documentation](../README.md).
