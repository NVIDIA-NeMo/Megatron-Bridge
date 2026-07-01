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

```{note}
This flexibility applies to the model and optimizer checkpoint. Restoring **Energon dataloader state** (see [Dataloader State (Energon)](#dataloader-state-energon) below) is saved per data-parallel rank, so it requires the **same data-parallel size** at load. Tensor, pipeline, and context parallelism may still change freely.
```

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

By default, `checkpoint.load` loads the **latest checkpoint** available in the specified base directory by reading from the tracker file (`latest_train_state.pt`). You can explicitly load from a specific checkpoint iteration using the `ckpt_step` parameter.

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
```

### Default Recipe Resume Behavior

Common recipes initialize both `checkpoint.save` and `checkpoint.load` to `./nemo_experiments/default/checkpoints`. If that directory already contains a checkpoint from a previous run, a new run with the same working directory may resume from it automatically.

For a fresh run, set a new `checkpoint.save` path and clear `checkpoint.load`:

```python
cfg.checkpoint.save = "/checkpoints/my_new_run"
cfg.checkpoint.load = None
```

For a full resume, keep `checkpoint.load` pointed at the base checkpoint directory:

```python
cfg.checkpoint.load = "/checkpoints/my_existing_run"
```

For model-weight initialization without optimizer, RNG, dataloader, or scheduler state, use `checkpoint.pretrained_checkpoint` instead of `checkpoint.load`.

## Fine-tuning and Initialization Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_checkpoint` | `Optional[str]` | `None` | Directory containing a pretrained full-model checkpoint for initialization |

`checkpoint.pretrained_checkpoint` is used for model-weight initialization before a new training or fine-tuning run. It can point to:

- A native Megatron base checkpoint directory containing tracker files such as `latest_train_state.pt` and `iter_*` subdirectories.
- A native Megatron iteration directory such as `/checkpoints/my_model/iter_0001000/` that directly contains the checkpoint payload.
- A local Hugging Face full-model directory containing `config.json` and model weight files. Remote Hugging Face model IDs are not accepted as checkpoint paths.

`checkpoint.pretrained_checkpoint` does not load optimizer, RNG, dataloader, or scheduler state. Use `checkpoint.load` for full native Megatron resume.

**PEFT note:** The `ckpt_step` parameter applies only to the `checkpoint.load` path, which is the adapter checkpoint when resuming PEFT. It does not select an iteration under `checkpoint.pretrained_checkpoint`. To use a specific frozen base checkpoint for PEFT, point `checkpoint.pretrained_checkpoint` directly at that `iter_N` directory.

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
- **Dataloader state (Energon only)**: Saved in a sibling `energon/` tree for deterministic data resumption; see [Dataloader State (Energon)](#dataloader-state-energon).
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
├── energon/                                   # Energon dataloader state
│   └── iter_N/                                # Keyed by iteration, matching the model checkpoint step
│       ├── train_dataloader_dprank000.pt    # DP rank 0 dataloader stream position
│       ├── train_dataloader_dprank001.pt    # DP rank 1 dataloader stream position
│       └── ...                              # One file per data-parallel rank
```

### Tokenizer Assets

By default, Megatron Bridge saves all tokenizer files to the checkpoint directory, making checkpoints self-contained and portable. This is particularly important for:
- **Inference and evaluation**: Direct access to tokenizer for computing logprobs
- **Portability**: No dependency on original tokenizer file locations
- **Reproducibility**: Exact tokenizer state is preserved

The tokenizer files saved depend on the tokenizer type:
- **Hugging Face tokenizers**: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and vocab files
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

## Dataloader State (Energon)

For [Megatron Energon](https://github.com/NVIDIA/Megatron-Energon) dataloaders, Megatron Bridge saves and restores the dataloader's stream position alongside the model checkpoint, so a resumed run continues over the **same sample stream** rather than restarting from an arbitrary position. This is what makes a resumed run reproduce the original losses one-to-one.

### How It Works

- **Save.** On every checkpoint, `save_state` is called on the train iterator's underlying `SavableDataLoader` and written to one file per data-parallel rank: `{dataloader_save}/iter_{step:07d}/train_dataloader_dprank{dp_rank:03d}.pt`. Only a single tensor/pipeline/context rank per DP replica writes, since the per-DP-rank state is identical across model-parallel ranks.
- **Restore.** On resume, after the data iterators are built, `restore_state` is called on **every** rank, keyed by the pure data-parallel rank. Energon's saved state is a periodic full snapshot (RNG + dataset state) at or before the consumed position, **plus an offset** of samples to skip forward from it. Restore rewinds each worker to that snapshot and replays the offset by re-running the data pipeline, landing exactly at the save-time position. (The snapshot-plus-offset form, rather than a single counter, is because the worker processes prefetch ahead of the main process.)

### Why Save and Restore Are Not Symmetric

Save writes **one file per DP replica**, while restore runs on **every** rank — this asymmetry is intentional.

Model-parallel ranks within a DP replica obtain the same data in one of two ways:

- **Broadcast** — one rank (tensor-parallel rank 0) reads from the dataloader and broadcasts the batch to its peers; only that rank advances an iterator. This is the classic GPT path.
- **Independent identical reads** — every rank holds its *own* dataloader instance (they are separate processes and cannot share one iterator object), seeded and sharded identically by DP rank, so each `next()` yields the same sample with no broadcast. The Qwen VL step function uses this so each pipeline stage can recompute MRoPE position IDs locally.

In the second pattern every rank advances its own iterator, so restoring only on one rank per replica would leave the others at stream position 0 — they would then read un-restored samples and diverge from the restored rank. Restore therefore runs on all ranks. Because the per-DP-rank state is identical across model-parallel ranks, save only needs to persist one copy (dedup to a single writer); restore fans that one file back out to every rank that shares the DP rank. Restoring on a rank that does *not* read from its iterator (the broadcast pattern) is harmless — it loads state it never consumes — so the restore path does not need to know which pattern a given model uses.

Keying by the **pure** data-parallel rank (excluding context parallelism) means context-parallel ranks within a replica share a file and restore to the same position; each then slices its own shard of the sequence locally afterward.

### Configuration

The save path defaults to an `energon/` subdirectory of `checkpoint.save`. The load path defaults to the `energon/` subdirectory of whichever checkpoint is actually restored (see below). So no configuration is needed for the common case: saving model checkpoints implies saving and restoring dataloader state.

```python
from megatron.bridge.data.energon.energon_provider import EnergonProvider  # or a subclass

dataset = EnergonProvider(
    ...,
    # dataloader_save defaults to "{checkpoint.save}/energon"
    # dataloader_load defaults to the "energon" subdir of the checkpoint actually loaded:
    #   checkpoint.save for a non-persistent or local checkpoint, checkpoint.load for a
    #   persistent one, or the parent of a directly specified iter_N directory.
)
```

The load default follows the actual source because the checkpoint restored on resume is not always rooted at `checkpoint.load`: a non-persistent or local checkpoint is selected from `checkpoint.save`, and a directly specified `iter_N` directory holds its `energon/` state one level up. Set `dataset.dataloader_save` / `dataset.dataloader_load` explicitly to override the destination. The fields have no effect for non-Energon dataloaders.

### Restore Behavior and Failure Modes

- If the dataloader state directory is **absent** (e.g. a checkpoint saved before this feature existed), the dataloader starts from the beginning and a message is logged.
- If the directory **exists** but the current rank's per-DP-rank file is **missing**, restore **raises**. This almost always means the data-parallel size changed since the checkpoint was saved; resuming would silently change the data order, so it fails loudly instead.

### Determinism Requirement

Energon checkpoints its workers periodically and, on restore, rewinds to the last checkpoint and *replays* the gap by re-running the data pipeline (decode → task encoder → packing) over the samples emitted since. It counts emitted samples to land on the right position (and, for sample packing, re-fetches the buffered samples by `restore_key`). So replay must be **deterministic per sample**: if re-encoding changes which samples are filtered out (`SkipSample`), the count desyncs and restore fails with `Unexpected skip sample during restoration` when a buffered packed sample can no longer be reproduced. Decorate stateless task encoders with `restore_seeds=True` so per-sample RNGs replay identically:

```python
@stateless(restore_seeds=True)
def encode_sample(self, sample):
    ...
```

See the Megatron-Energon source — [`SavableDataLoader.save_state_rank` / `restore_state_rank`](https://github.com/NVIDIA/Megatron-Energon/blob/bef8be243505959973cc07ee740432e7a2454cf1/src/megatron/energon/savable_loader.py#L924) and the periodic-checkpoint/skip-forward logic in `SavableDatasetWrapper` (`_store_checkpoint`, `get_checkpoint`), plus the `stateless` decorator's `restore_seeds`.

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

## Custom Checkpoint Manager

For advanced use cases, you can provide a custom checkpoint manager implementation to override the default save/load behavior. This enables integration with custom storage backends, alternative checkpoint formats, or organization-specific checkpointing workflows.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_manager_class` | `str \| None` | `None` | Fully qualified class name for a custom `CheckpointManager` implementation |

### Usage

Specify a custom checkpoint manager class in your configuration:

**YAML:**
```yaml
checkpoint:
  save: /path/to/checkpoints
  custom_manager_class: "mypackage.checkpoint.MyCheckpointManager"
```

**Python:**
```python
from megatron.bridge.training.config import CheckpointConfig

checkpoint = CheckpointConfig(
    save="/path/to/checkpoints",
    custom_manager_class="mypackage.checkpoint.MyCheckpointManager",
)
```

### Implementing a Custom Manager

Your custom manager must implement the `CheckpointManager` protocol defined in `megatron.bridge.training.checkpointing`:

```python
from megatron.bridge.training.checkpointing import (
    CheckpointManager,
    CheckpointSaveContext,
    CheckpointLoadContext,
    save_checkpoint,
    load_checkpoint,
    init_checkpointing_context,
)
from megatron.bridge.training.config import CheckpointConfig
from megatron.bridge.training.state import GlobalState


class MyCheckpointManager:
    """Custom checkpoint manager example."""

    def __init__(self, checkpoint_config: CheckpointConfig) -> None:
        self.checkpoint_config = checkpoint_config
        # Initialize internal context for caching strategies
        self._context = init_checkpointing_context(checkpoint_config)

    def save(self, ctx: CheckpointSaveContext) -> None:
        """Save a checkpoint with custom logic."""
        # Option 1: Completely custom implementation
        # my_custom_save(ctx.state, ctx.model, ...)

        # Option 2: Wrap the default implementation
        save_checkpoint(
            state=ctx.state,
            model=ctx.model,
            optimizer=ctx.optimizer,
            opt_param_scheduler=ctx.opt_param_scheduler,
            num_floating_point_operations_so_far=ctx.num_floating_point_operations_so_far,
            checkpointing_context=self._context,
            non_persistent_ckpt=ctx.non_persistent_ckpt,
            train_data_iterator=ctx.train_data_iterator,
        )
        # Add custom post-processing (e.g., upload to cloud)
        upload_to_s3(ctx.state.cfg.checkpoint.save)

    def load(self, ctx: CheckpointLoadContext) -> tuple[int, int]:
        """Load a checkpoint with custom logic."""
        # Returns (iteration, num_floating_point_operations_so_far)
        return load_checkpoint(
            state=ctx.state,
            model=ctx.model,
            optimizer=ctx.optimizer,
            opt_param_scheduler=ctx.opt_param_scheduler,
            strict=ctx.strict,
            checkpointing_context=self._context,
            skip_load_to_model_and_opt=ctx.skip_load_to_model_and_opt,
        )

    def finalize_async_saves(
        self, state: GlobalState, blocking: bool = False, terminate: bool = False
    ) -> None:
        """Finalize any pending asynchronous saves."""
        from megatron.bridge.training.checkpointing import maybe_finalize_async_save

        maybe_finalize_async_save(
            global_state=state,
            ckpt_cfg=self.checkpoint_config,
            blocking=blocking,
            terminate=terminate,
        )
```

### Context Dataclasses

The save and load methods receive context dataclasses that bundle all required parameters:

**`CheckpointSaveContext`:**
| Field | Type | Description |
|-------|------|-------------|
| `state` | `GlobalState` | Global training state (config, train_state, loggers) |
| `model` | `list[MegatronModule]` | Model modules to save |
| `optimizer` | `MegatronOptimizer \| None` | Optimizer instance |
| `opt_param_scheduler` | `Any \| None` | Learning rate scheduler |
| `num_floating_point_operations_so_far` | `int` | Cumulative FLOPs |
| `train_data_iterator` | `Any \| None` | Data iterator (optional) |
| `non_persistent_ckpt` | `bool` | Whether this is a non-persistent checkpoint |

**`CheckpointLoadContext`:**
| Field | Type | Description |
|-------|------|-------------|
| `state` | `GlobalState` | Global training state |
| `model` | `list[MegatronModule]` | Model modules to load into |
| `optimizer` | `MegatronOptimizer \| None` | Optimizer instance |
| `opt_param_scheduler` | `Any \| None` | Learning rate scheduler |
| `strict` | `bool` | Enforce strict loading (default: `True`) |
| `skip_load_to_model_and_opt` | `bool` | Skip loading into model/optimizer (default: `False`) |

### Limitations

The custom checkpoint manager is designed for customizing the save/load **operations** during training. The following limitations apply:

**Checkpoint format compatibility**: Custom managers that change the checkpoint directory structure or metadata files (e.g., `latest_train_state.pt`, `run_config.yaml`) are not well supported. Many utilities in Megatron Bridge assume the standard Megatron checkpoint format. For instance, Hugging Face ↔ custom format conversion is not supported.

**PEFT with custom checkpoints**: The custom manager only applies to the training save/load flow (the `save` and `load` configuration paths), not to base model loading for PEFT. `checkpoint.pretrained_checkpoint` is still loaded by the built-in base-model initialization path and should point to a native Megatron checkpoint or a local Hugging Face full-model directory.

**Inference loading**: Loading checkpoints for inference via `model_load_save.py` utilities is undefined behavior with custom checkpoint formats. Use your custom format's loading utilities instead.

### Default Behavior

When `custom_manager_class` is not set, Megatron Bridge uses `DefaultCheckpointManager`, which wraps the existing `save_checkpoint` and `load_checkpoint` functions. This ensures full backward compatibility—the checkpoint manager abstraction introduces no changes to existing training workflows.

## Related Documentation

- {doc}`megatron-fsdp` - Megatron FSDP configuration and `fsdp_dtensor` format requirements
- {doc}`../parallelisms` - Understanding data and model parallelism strategies
- {doc}`config-container-overview` - Complete configuration reference
