# Checkpointing

The {py:class}`bridge.training.config.CheckpointConfig` controls model checkpointing behavior, including saving and loading checkpoints, checkpoint formats, and various optimization features.

```{Note}
This documentation covers **Megatron-format checkpoints** used during training. For converting between 🤗 Hugging Face and Megatron formats, see the {doc}`../bridge-guide`.
```

## Overview

Megatron Bridge uses Megatron Core's distributed checkpointing system, which is designed for large-scale training across multiple GPUs and nodes. The distributed checkpoint approach saves the state of a distributed training job by sharding checkpoint data across multiple files, reducing memory overhead and improving GPU utilization during save/load operations.

### Distributed Checkpointing Benefits

**Memory Efficiency**: Instead of gathering all model parameters and optimizer states on a single rank, distributed checkpointing saves data directly from each rank, significantly reducing memory requirements during checkpointing.

**Parallelism Flexibility**: The system provides flexibility to resume training using different parallelism strategies. You can change tensor parallelism, pipeline parallelism, or data parallelism sizes between checkpoint save and load operations.

**Scalability**: Handles all types of parallelism including:
- **Data Parallelism (DP)**: Replicates the model across multiple GPUs with different data batches
- **Tensor Parallelism (TP)**: Distributes individual layer parameters across GPUs  
- **Pipeline Parallelism (PP)**: Assigns consecutive layers to different GPUs
- **Context Parallelism (CP)**: Shards tensors along the sequence dimension for long sequences
- **Expert Parallelism (EP)**: Distributes MoE expert weights across GPUs

**Performance**: The distributed optimizer shards optimizer states and master parameters across data-parallel ranks instead of replicating them, reducing memory usage and communication overhead.


## Save Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save` | `Optional[str]` | `None` | Output directory to save checkpoints to **in Megatron format** |
| `save_interval` | `Optional[int]` | `None` | Number of iterations between persistent checkpoint saves |
| `save_optim` | `bool` | `True` | Whether to save optimizer state |
| `save_rng` | `bool` | `True` | Whether to save random number generator state |
| `save_tokenizer_assets` | `bool` | `True` | Whether to save tokenizer files (vocab, config, special tokens) to checkpoint |

### Asynchronous Saving

Asynchronous saving allows training to continue while checkpoint data is persisted to disk in the background, reducing the impact of checkpointing on training throughput.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `async_save` | `bool` | `False` | Enable asynchronous checkpoint saving (requires `torch_dist` format) |

## Load Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load` | `Optional[str]` | `None` | Directory containing a model checkpoint to load **in Megatron format** |
| `load_optim` | `bool` | `True` | Whether to load optimizer state from checkpoint |
| `load_rng` | `bool` | `True` | Whether to load random number generator state from checkpoint |
| `load_main_params_from_ckpt` | `bool` | `False` | Load main parameters from checkpoint (use with `load_optim=False`) |
| `ckpt_step` | `Optional[int]` | `None` | Specific checkpoint iteration to load (overrides latest from tracker) |
| `exit_on_missing_checkpoint` | `bool` | `False` | Exit if specified checkpoint is not found instead of random initialization |
| `dist_ckpt_strictness` | `Literal[...]` | `"assume_ok_unexpected"` | Handling of key mismatches during distributed checkpoint load |

### Loading Specific Checkpoint Iterations

By default, Megatron Bridge loads the **latest checkpoint** available in the specified directory by reading from the tracker file (`latest_train_state.pt`). However, you can explicitly load from a specific checkpoint iteration using the `ckpt_step` parameter.

**Python API:**
```python
from megatron.bridge.training.config import CheckpointConfig

# Load latest checkpoint
checkpoint = CheckpointConfig(
    load="/path/to/checkpoint_dir"
)

# Load specific iteration
checkpoint = CheckpointConfig(
    load="/path/to/checkpoint_dir",
    ckpt_step=5000  # Overrides tracker, loads iter_0005000
)
```

```{note}
The `load` parameter should always point to the base checkpoint directory (not the `iter_N` subdirectory). The `ckpt_step` parameter overrides which iteration is loaded from that directory.

**Important:** If `ckpt_step` is specified but the checkpoint directory does not exist, training will **fail immediately** with a `FileNotFoundError`. This is intentional to prevent accidentally starting training from scratch when you meant to resume from a specific checkpoint.

**PEFT Note:** The `ckpt_step` parameter applies **only to the `load` path** (adapter checkpoints), not to `pretrained_checkpoint` (frozen base model). When resuming PEFT training:
- `pretrained_checkpoint`: Always loads the latest/release checkpoint (base model)
- `load` + `ckpt_step`: Can load a specific adapter checkpoint iteration


### Checkpoint Loading Strictness

When loading distributed checkpoints, there may be mismatches between the keys in the saved checkpoint and what the current model expects. This can happen when resuming training with different parallelism settings, model configurations, or software versions. The `dist_ckpt_strictness` parameter controls how these mismatches are handled:

- **`assume_ok_unexpected`**: Assume unexpected keys are acceptable (default, most permissive)
- **`log_unexpected`**: Log unexpected keys but continue loading
- **`log_all`**: Log all key mismatches for debugging
- **`raise_unexpected`**: Raise error on unexpected keys (stricter validation)
- **`raise_all`**: Raise error on any key mismatch (strictest validation)
- **`return_unexpected`**: Return information about unexpected keys
- **`return_all`**: Return information about all key mismatches
- **`ignore_all`**: Ignore all key mismatches completely

## Fine-tuning Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_checkpoint` | `Optional[str]` | `None` | Directory containing pretrained model checkpoint **in Megatron format** for fine-tuning |

## Checkpoint Format

Megatron Bridge supports multiple checkpoint formats optimized for different use cases:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ckpt_format` | `Literal["torch_dist", "zarr", "fsdp_dtensor"]` | `"torch_dist"` | Checkpoint format to use |

### Available Formats

**`torch_dist`** (Default)
- PyTorch distributed checkpoint format
- Compatible with most parallelism strategies (DP, TP, PP, CP, EP)
- Supports asynchronous saving when `async_save=True`
- Recommended for general use

**`zarr`**
- Zarr-based checkpoint format
- Alternative to `torch_dist` for certain use cases
- Compatible with distributed parallelism strategies

**`fsdp_dtensor`**
- Specialized format for Megatron FSDP (Fully Sharded Data Parallel)
- **Required when using `use_megatron_fsdp=True`**
- Optimized for sharded parameter layouts
- Not compatible with other FSDP implementations

### Format Selection

Choose your checkpoint format based on your training configuration:

```python
from megatron.bridge.training.config import CheckpointConfig

# Standard distributed training (DDP, TP, PP)
checkpoint = CheckpointConfig(
    ckpt_format="torch_dist",  # Default, works for most cases
    save="/path/to/checkpoints",
)

# Megatron FSDP training
checkpoint = CheckpointConfig(
    ckpt_format="fsdp_dtensor",  # Required for FSDP
    save="/path/to/checkpoints",
)
```

### Format Compatibility

| Format | DDP | Distributed Optimizer | Megatron FSDP | Torch FSDP2 | Async Save |
|--------|-----|----------------------|---------------|-------------|------------|
| `torch_dist` | ✅ | ✅ | ❌ | ✅ | ✅ |
| `zarr` | ✅ | ✅ | ❌ | ✅ | ❌ |
| `fsdp_dtensor` | ❌ | ❌ | ✅ | ❌ | ❌ |

**Important**: When using Megatron FSDP (`use_megatron_fsdp=True`), you must set `ckpt_format="fsdp_dtensor"`. Other formats are not compatible with FSDP's sharded parameter layout. See {doc}`megatron-fsdp` for complete FSDP configuration details.

## Performance Optimizations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fully_parallel_save` | `bool` | `True` | Apply full save parallelization across data parallel ranks |
| `fully_parallel_load` | `bool` | `False` | Apply full load parallelization across data parallel ranks |
| `ckpt_assume_constant_structure` | `bool` | `False` | Assume constant model/optimizer structure over successive checkpoint saves for performance optimizations |


## Checkpoint Contents

The checkpoint includes the following components when using the `torch_dist` checkpoint format:
- **Model parameters and optimizer states**: Stored across `.distcp` files to support distributed training.
- **Training state**: Captures the current iteration count, number of consumed samples, and the state of the learning rate scheduler.
- **Configuration**: Serialized as a YAML file (`run_config.yaml`) containing the complete `ConfigContainer`.
- **Tokenizer files**: All tokenizer artifacts (vocabulary, special tokens, config) for self-contained checkpoints.
- **Dataloader states**: Ensures deterministic resumption of data iteration.
- **Metadata**: Used for validating and correctly loading the checkpoint.

Megatron Bridge creates checkpoints with the following directory structure:

```
checkpoint_dir/
├── latest_train_state.pt                      # Latest training state (top-level)
├── iter_N/                                    # Checkpoint at iteration N
│   ├── __0_0.distcp                          # Distributed checkpoint shards: maps to PyTorch DCP weights format
│   ├── __0_1.distcp                          # Contains model parameters, optimizer states
│   ├── __1_0.distcp
│   ├── __1_1.distcp
│   ├── ...
│   ├── .metadata                             # PyTorch DCP checkpoint metadata
│   ├── common.pt                             # MCore dist ckpt states saved from rank 0
│   ├── metadata.json                         # MCore dist ckpt metadata
│   ├── run_config.yaml                       # Serialized ConfigContainer
│   ├── train_state.pt                        # Number of steps, consumed samples, etc
│   ├── tokenizer/                            # Tokenizer files (saved by default)
│   │   ├── tokenizer.json                   # Full tokenizer vocabulary
│   │   ├── tokenizer_config.json            # Tokenizer configuration
│   │   ├── special_tokens_map.json          # Special token definitions
│   │   └── ...                              # Other tokenizer artifacts
│   ├── dataloader_state/                     # Data iterator states
│   │   ├── train_dataloader_dprank000.pt    # DP rank 0 dataloader state
│   │   ├── train_dataloader_dprank001.pt    # DP rank 1 dataloader state
│   │   ├── train_dataloader_dprank002.pt    # DP rank 2 dataloader state
│   │   └── ...                              # One file per DP rank
```

### Tokenizer Assets

By default, Megatron Bridge saves all tokenizer files to the checkpoint directory, making checkpoints self-contained and portable. This is particularly important for:
- **Inference and evaluation**: Direct access to tokenizer for computing logprobs
- **Portability**: No dependency on original tokenizer file locations
- **Reproducibility**: Exact tokenizer state is preserved

The tokenizer files saved depend on the tokenizer type:
- **HuggingFace tokenizers**: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and vocab files
- **SentencePiece tokenizers**: `tokenizer.model` file
- **GPT2 BPE tokenizers**: `vocab.json` and `merges.txt`
- **BERT tokenizers**: `vocab.txt`
- **Tiktoken tokenizers**: `tokenizer.json`

To disable tokenizer asset saving for performance-sensitive scenarios:

```python
from megatron.bridge.training.config import CheckpointConfig

checkpoint = CheckpointConfig(
    save_tokenizer_assets=False,  # Skip tokenizer file saving
    ...
)
```

Or in YAML:

```yaml
checkpoint:
  save_tokenizer_assets: false
```

## Local Checkpointing

Local checkpointing saves model checkpoints directly to storage on each node (e.g., local SSDs or RAM disks), instead of relying solely on a shared network filesystem. This approach can significantly speed up the saving process and reduce the load on shared storage infrastructure.

Local checkpointing leverages the [NVIDIA Resiliency Extension](https://nvidia.github.io/nvidia-resiliency-ext/checkpointing/local/index.html) and provides several key features:

- **Local Saving**: Each node saves its part of the checkpoint locally, reducing network I/O and improving save performance.
- **Synchronous and Asynchronous Support**: Saving can happen synchronously or asynchronously, mirroring the configuration used for global checkpoints.
- **Automatic Cleanup**: Handles the removal of outdated or incomplete local checkpoints automatically.
- **Optional Replication**: For multi-node jobs, checkpoints are replicated to other nodes to allow recovery even if a node fails after saving. Single-node jobs do not use replication.
- **Automated Loading**: When resuming, the framework automatically finds the latest valid checkpoint, comparing local and global checkpoints, and retrieves any needed parts across nodes.
### Non-Persistent Checkpointing Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `non_persistent_save_interval` | `Optional[int]` | `None` | Iterations between non-persistent saves |
| `non_persistent_ckpt_type` | `Optional[Literal["global", "local", "in_memory", "None"]]` | `None` | Type of non-persistent checkpointing |
| `non_persistent_global_ckpt_dir` | `Optional[str]` | `None` | Directory for global non-persistent checkpoints |
| `non_persistent_local_ckpt_dir` | `Optional[str]` | `None` | Directory for local non-persistent checkpoints |
| `non_persistent_local_ckpt_algo` | `Literal["fully_parallel", "atomic"]` | `"fully_parallel"` | Algorithm for local non-persistent checkpointing |

### Replication and Fault Tolerance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `replication` | `bool` | `False` | Enable replication of local checkpoints across ranks |
| `replication_jump` | `Optional[int]` | `None` | Spacing between ranks storing replicas |
| `replication_factor` | `int` | `2` | Number of machines storing replica of each rank's data |

### Checkpointing Distributed Optimizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dist_ckpt_optim_fully_reshardable` | `bool` | `False` | Make optimizer distributed checkpoint fully reshardable (TP/PP/EP/DP) as opposed to plain DP reshardability |
| `distrib_optim_fully_reshardable_mem_efficient` | `bool` | `False` | Use as little memory as possible during save and load by using Gloo. Has affect only with `dist_ckpt_optim_fully_reshardable` flag |

## Related Documentation

- {doc}`megatron-fsdp` - Megatron FSDP configuration and `fsdp_dtensor` format requirements
- {doc}`../parallelisms` - Understanding data and model parallelism strategies
- {doc}`config-container-overview` - Complete configuration reference
