# Configuration Overview

The `ConfigContainer` is the central configuration object in Megatron Bridge that orchestrates all aspects of training. It acts as a single source of truth that brings together model architecture, training parameters, data loading, optimization, checkpointing, logging, and distributed training settings.

## What is ConfigContainer

`ConfigContainer` is a dataclass that holds all the configuration objects needed for training:

```python
from megatron.bridge.training.config import ConfigContainer

# ConfigContainer brings together all training configurations
config = ConfigContainer(
    model=model_provider,             # Model architecture and parallelism
    train=training_config,            # Training loop parameters  
    optimizer=optimizer_config,       # Megatron Optimization settings
    scheduler=scheduler_config,       # Learning rate scheduling
    dataset=dataset_config,           # Data loading configuration
    logger=logger_config,             # Logging and monitoring
    tokenizer=tokenizer_config,       # Tokenization settings
    checkpoint=checkpoint_config,     # Checkpointing and resuming
    dist=distributed_config,          # Distributed training setup
    ddp=ddp_config,                   # Megatron Distributed Data Parallel settings
    # Optional configurations
    peft=peft_config,                 # Parameter-efficient fine-tuning
    profiling=profiling_config,       # Performance profiling
    mixed_precision=mp_config,        # Mixed precision training
    comm_overlap=comm_overlap_config, # Communication overlap settings
    # ... and more
)
```

## Configuration Attributes

The `ConfigContainer` organizes training settings into three categories—**required**, **default**, and **optional** each controlling a specific aspect of model setup, execution, and advanced features.

### Required Attributes

The following configuration attributes must be provided when creating a `ConfigContainer` for training:

| Attribute | Purpose |
|-----------|---------|
| `model` | Model architecture and parallelism strategy (from model providers) |
| `train` | Training loop parameters (batch sizes, iterations, validation) |
| `optimizer` | Optimizer type and hyperparameters (from Megatron Core) |
| `scheduler` | Learning rate and weight decay scheduling |
| `dataset` | Data loading and preprocessing configuration |
| `logger` | Logging, TensorBoard, and WandB configuration |
| `tokenizer` | Tokenizer settings and vocabulary |
| `checkpoint` | Checkpointing, saving, and loading |

### Default Attributes

These attributes have predefined defaults, but can be customized to suit specific training needs:

| Attribute | Purpose |
|-----------|---------|
| `dist` | Distributed training initialization |
| `ddp` | Data parallel configuration (from Megatron Core) |
| `rng` | Random number generation settings |
| `rerun_state_machine` | Result validation and error injection |

### Optional Attributes

These configuration attributes are set to `None` by default and enable specific features when configured:

| Attribute | Purpose |
|-----------|---------|
| `peft` | Parameter-efficient fine-tuning (LoRA, DoRA, etc.) |
| `profiling` | Performance profiling with nsys or PyTorch profiler |
| `mixed_precision` | Mixed precision training settings |
| `comm_overlap` | Communication overlap optimizations |
| `ft` | Fault tolerance and automatic recovery |
| `straggler` | GPU straggler detection |
| `nvrx_straggler` | NVIDIA Resiliency Extension straggler detection |

## Megatron Core Integration

Several configuration components in `ConfigContainer` are directly derived from Megatron Core, ensuring compatibility with its optimization and data parallel frameworks:

| Configuration | Purpose |
|---------------|---------|
| `optimizer`   | Uses `OptimizerConfig` from Megatron Core for optimization settings |
| `ddp`         | Uses `DistributedDataParallelConfig` from Megatron Core for data parallel configuration |

These configurations provide seamless integration with the Megatron Core library.

## Automatic Configuration Processing

At the start of training, Megatron Bridge automatically validates, adjusts, and resolves configuration settings to ensure consistency and optimal runtime behavior:

1. **Validation**: All configurations are validated for consistency and compatibility.
2. **Runtime Overrides**: Mixed precision and communication overlap configurations apply runtime overrides based on environment variables and device-specific checks.
3. **Dependency Resolution**: Dependent values (like data parallel size and scheduler steps) are calculated automatically.

You do not need to manually validate configurations - this process occurs automatically at the start of training.

## Configuration Export and Import

### Export to YAML
```python
# Print yaml configuration to console
config.print_yaml()

# Save to file
config.to_yaml("config.yaml")
```

### Load from YAML
```python
# Load configuration from YAML file
config = ConfigContainer.from_yaml("config.yaml")
```