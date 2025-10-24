# Configuration Overview

The `ConfigContainer` is the central configuration object in Megatron Bridge that holds all settings for training. It acts as a single source of truth that brings together model architecture, training parameters, data loading, optimization, checkpointing, logging, and distributed training settings.

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

## Configuration Components

| Component | Purpose | Required | Default |
|-----------|---------|----------|---------|
| `model` | Model architecture and parallelism strategy (GPT, T5, Mamba) | ✅ | - |
| `train` | Training loop parameters (batch sizes, iterations, validation) | ✅ | - |
| `optimizer` | Optimizer type and hyperparameters (from Megatron Core) | ✅ | - |
| `scheduler` | Learning rate and weight decay scheduling | ✅ | - |
| `dataset` | Data loading and preprocessing configuration | ✅ | - |
| `logger` | Logging, TensorBoard, and WandB configuration | ✅ | - |
| `tokenizer` | Tokenizer settings and vocabulary | ✅ | - |
| `checkpoint` | Checkpointing, saving, and loading | ✅ | - |
| `dist` | Distributed training initialization | | `DistributedInitConfig()` |
| `ddp` | Data parallel configuration (from Megatron Core) | | `DistributedDataParallelConfig()` |
| `rng` | Random number generation settings | | `RNGConfig()` |
| `rerun_state_machine` | Result validation and error injection | | `RerunStateMachineConfig()` |
| `mixed_precision` | Mixed precision training settings | | `None` |
| `comm_overlap` | Communication overlap optimizations | | `None` |
| `peft` | Parameter-efficient fine-tuning (LoRA, DoRA, etc.) | | `None` |
| `profiling` | Performance profiling with nsys or PyTorch profiler | | `None` |
| `ft` | Fault tolerance and automatic recovery | | `None` |
| `straggler` | GPU straggler detection | | `None` |
| `nvrx_straggler` | NVIDIA Resiliency Extension straggler detection | | `None` |
| `inprocess_restart` | In-process restart for fault tolerance | | `None` |

## Design Philosophy

### **Interoperability with External Config Systems**

Megatron Bridge's Python configurations are designed to be amenable to other configuration systems you already use, such as:

- Programmatic configuration: Direct Python object manipulation
- argparse: Command-line arguments can be easily mapped to dataclass fields
- File-based overrides: JSON, YAML, or other config files can override Python configs

All of these approaches can be translated into Python dataclass instances. The framework provides utilities as a convenience for YAML-based overrides with OmegaConf, but the framework is not tied to any particular configuration system.

```python
# All of these approaches work seamlessly:

# 1. Direct Python configuration
config = ConfigContainer(
    model=GPTModelProvider(num_layers=24, hidden_size=2048),
    train=TrainingConfig(global_batch_size=256, train_iters=10000),
    # ... other configs
)

# 2. YAML-based serialization and deserialization (round-trip)
config.to_yaml("my_config.yaml")
config = ConfigContainer.from_yaml("my_config.yaml")  # Load previously saved config

# 3. Programmatic override after creation
config.train.global_batch_size = 512  # Override after instantiation
config.model.num_layers = 48          # Modify model architecture
```

### Centralized Configuration

Megatron provides extensive flexibility through a rich set of configuration options. The `ConfigContainer` brings all these settings together in a single, organized object. This centralization makes configuration discoverable and maintainable - you have one place to understand and control all aspects of your training run.

Unlike pure YAML-based configuration systems, `ConfigContainer` provides centralization with the full power of Python. You get the organizational benefits of a single configuration file combined with the programmatic flexibility of Python.

The configuration system is built using nested dataclasses, providing:

- **Modularity**: Each config component is independently defined and testable
- **Type safety**: Full static type checking
- **IDE support**: Autocomplete and type hints in development environments  
- **Serialization**: Easy conversion to/from YAML, JSON, or other formats
- **Validation**: Built-in field validation

```python
@dataclass
class ConfigContainer:
    model: GPTModelProvider      # Dataclass for model architecture
    train: TrainingConfig        # Dataclass for training parameters
    optimizer: OptimizerConfig   # Dataclass for optimization settings
    # ... nested dataclasses for each concern
```

### Lazy Configuration and Deferred Validation

For training workloads, configurations are lazy to support flexible user workflows:

**Problem with Eager Validation:**
```python
# This would be problematic with eager validation:
config = TrainingConfig(train_iters=1000)
# __post_init__ calculates dependent values immediately

config.train_iters = 5000  # User override
# Dependent values are now stale and incorrect!
```

**Solution with Lazy Finalization:**
```python
# Megatron Bridge approach - deferred validation
config = TrainingConfig(train_iters=1000)
config.train_iters = 5000  # User can safely override

# Validation happens automatically right when training starts
pretrain(config, forward_step_func)  # All dependent values calculated correctly
```

**Benefits:**
- Users can instantiate configs and subsequently override fields safely
- Dependent values are calculated correctly after all user modifications are applied
- Validation happens at the right time, right before training begins
- Flexible configuration workflows are supported

### **Model Independence**

Model configurations are designed to be independently usable outside the full training loop provided by thr framework:

```python
# Models can be used standalone
model_provider = GPTModelProvider(
    num_layers=24,
    hidden_size=2048,
    vocab_size=50000,    # Must be explicitly set
    seq_length=2048,     # Must be explicitly set
)

# This works independently of other configs
model_provider.finalize()
model = model_provider.provide()
```

**Trade-off**: The price for this flexibility is the need to explicitly set values like `seq_length` in multiple places during training. These settings are checked for consistency at the beginning of training.

## Usage

```python
# Create and configure
config = ConfigContainer(
    model=GPTModelProvider(num_layers=24, seq_length=2048),
    train=TrainingConfig(train_iters=1000),
    dataset=GPTDatasetConfig(sequence_length=2048),  # Must match model seq_length
    # ... other required configs
)

# Modify as needed
config.train.train_iters = 5000
config.model.hidden_size = 4096

# Start training - validation happens automatically
pretrain(config, forward_step_func)
```

## Configuration Export and Import

### Export to YAML
```python
# Print YAML configuration to console
config.print_yaml()

# Save to file
config.to_yaml("config.yaml")
```

### Load from YAML
```python
# Load configuration from YAML file
config = ConfigContainer.from_yaml("config.yaml")
```
