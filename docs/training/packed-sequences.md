# Packed Sequences

This guide explains how to use packed sequences in Megatron Bridge for efficient supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT).

## Overview

When fine-tuning large language models, GPU under-utilization often occurs due to inefficient input data structure. This inefficiency arises because many fine-tuning datasets have a skewed distribution of sequence lengths, with many short sequences and a few long ones, following [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law). Since transformer models require fixed-length inputs, shorter sequences must be padded with many padding tokens.

This leads to two main inefficiencies:

- Computation performed on the pad tokens is eventually masked out, resulting in wasted GPU computation.
- Micro batch size is often limited by the batch which contains longer sequences, so that most other micro batches have under-utilized GPU memory.

Packed sequences is a training technique where multiple training sequences (examples) are concatenated into one long sequence (pack). This technique greatly reduces the number of padding tokens, allowing more meaningful tokens to be processed in each micro batch. As a result, it maximizes both GPU compute and GPU memory utilization.

**Note:** Sequence packing is primarily beneficial for fine-tuning workloads. Megatron-style pretraining datasets (using `IndexedDataset` and `GPTDataset`) already concatenate documents during sampling to fill sequences to the target length, eliminating padding tokens without requiring the boundary-aware packing infrastructure described here. For supervised fine-tuning, however, naive concatenation is insufficient—each training example must be treated individually to preserve data quality.

The conventional solution is to build a custom attention mask (specifically, a block triangular mask) to mask out attention values between sequences. However, this increases the complexity of attention from $\sum_i {s_i}^2$ to $\Big({\sum_i {s_i}}\Big)^2$, where $s_i$ is the length of the $i$th subsequence. In practice, the conventional solution puts a limit on the packed sequence size.

Instead, Megatron Bridge provides a highly optimized version of sequence packing which makes use of variable-length attention kernels in FlashAttention and TransformerEngine. Instead of providing a custom attention mask, information about sequence boundaries is passed in with the `cu_seqlens` variable (short for cumulative sequence length). With this approach, attention values between sequences are never calculated, so the complexity of attention remains at $\sum_i {s_i}^2$. This allows the packed sequence size to increase to arbitrary lengths without affecting the memory complexity, so that GPU memory can be fully utilized.

The packed sequence implementation automatically creates {py:class}`bridge.data.datasets.sft.GPTSFTPackedDataset` instances when `.npy` files are detected, providing optimized data loading and batching for packed sequences.

## Using Packed Sequences

### Prepare the Dataset

In Megatron Bridge, the packed dataset is automatically prepared before training using the {py:func}`bridge.data.datasets.packed_sequence.prepare_packed_sequence_data` function, eliminating the need for any additional preprocessing steps.

### Configure Packed Sequences

Packed sequences are configured through the {py:class}`bridge.training.config.FinetuningDatasetConfig` by specifying `packed_sequence_specs`:

```python
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

config = ConfigContainer(
    # ... other configurations
    dataset=FinetuningDatasetConfig(
        dataset_root="/path/to/your/dataset",
        seq_length=2048,
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=2048,
            tokenizer_model_name="your_tokenizer_name",
        ),
    ),
    # ... other configurations
)
```

### PackedSequenceSpecs Configuration

The {py:class}`bridge.data.datasets.packed_sequence.PackedSequenceSpecs` class provides the following configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `packed_sequence_size` | `int` | `-1` | If positive, enables sequence packing with the specified pack size. If ≤ 0, sequence packing is disabled. |
| `tokenizer_model_name` | `str` | `None` | Tokenizer model name for tracking, since different tokenizers produce different packed datasets. |
| `packed_train_data_path` | `str` | `None` | Custom path for packed training dataset file (`.npy` format). |
| `packed_val_data_path` | `str` | `None` | Custom path for packed validation dataset file (`.npy` format). |
| `packed_metadata_path` | `str` | `None` | Custom path for packing metadata file (`.jsonl` format). |
| `pad_cu_seqlens` | `bool` | `False` | Whether to pad `cu_seqlens` to constant size, required for CUDA graphs. |

### Batch Size Considerations

When using packed sequences, you must adjust your batch sizes:

1. **Micro batch size must be set to 1**: This constraint arises because samples in a micro batch are no longer stacked; they are now concatenated during the data preparation step. Consequently, micro batch size becomes irrelevant when using packed sequences.

2. **Global batch size must be adjusted**: Since each pack now contains multiple sequences, the global batch size needs to be reduced by the average number of sequences per pack `n` where `n = num_sequences_in_dataset / num_packs` (equivalently, `n = packed_sequence_size / average_seq_len`). This ensures that each gradient iteration sees, on average, the same number of tokens. The value of `n` is printed out during the data preparation step. You may need to run training once, obtain the value of `n` from the logs, then run your training script again with the updated global batch size.

### Full Configuration Example

```python
from megatron.bridge.training.config import (
    ConfigContainer, TrainingConfig, CheckpointConfig, SchedulerConfig
)
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.peft.lora import LoRA
from megatron.core.optimizer import OptimizerConfig

config = ConfigContainer(
    model=model_provider,
    train=TrainingConfig(
        train_iters=1000,
        global_batch_size=32,  # Reduced from original due to packing
        micro_batch_size=1,    # Required for packed sequences
        eval_interval=100,
    ),
    optimizer=OptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=True,
    ),
    scheduler=SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=100,
        lr_decay_iters=1000,
    ),
    dataset=FinetuningDatasetConfig(
        dataset_root="/path/to/dataset",
        seq_length=2048,
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=2048,
            tokenizer_model_name="llama2_tokenizer",
        ),
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/pretrained/model",
        save="/path/to/checkpoints",
        save_interval=200,
    ),
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=16,
        alpha=32,
        dropout=0.1,
    ),
    # ... other configurations
)
```

## File Organization

When using packed sequences, the {py:class}`bridge.data.builders.finetuning_dataset.FinetuningDatasetBuilder` automatically organizes files in your dataset directory:

```
dataset_root/
├── training.jsonl          # Original training data
├── validation.jsonl        # Original validation data
└── packed/
    └── {tokenizer_name}/
        ├── training_{packed_size}.npy      # Packed training data
        ├── validation_{packed_size}.npy    # Packed validation data
        └── {packed_size}_metadata.jsonl    # Packing metadata
```

The tokenizer name and packed sequence size are automatically incorporated into the file paths to avoid conflicts when using different configurations.

## Advanced Configuration

### Custom File Paths

You can specify custom paths for packed data files:

```python
packed_sequence_specs = PackedSequenceSpecs(
    packed_sequence_size=4096,
    tokenizer_model_name="custom_tokenizer",
    packed_train_data_path="/custom/path/training_packed.npy",
    packed_val_data_path="/custom/path/validation_packed.npy",
    packed_metadata_path="/custom/path/metadata.jsonl",
)
```

### CUDA Graphs Support

For CUDA graphs compatibility, enable `pad_cu_seqlens`:

```python
packed_sequence_specs = PackedSequenceSpecs(
    packed_sequence_size=2048,
    pad_cu_seqlens=True,  # Required for CUDA graphs
    tokenizer_model_name="your_tokenizer",
)
```

When `pad_cu_seqlens=True`, you must also set `pad_to_max_length=True` in your dataset configuration.

## API Reference

For detailed API documentation, see:

- {py:class}`bridge.training.config.FinetuningDatasetConfig` - Main dataset configuration class
- {py:class}`bridge.data.datasets.packed_sequence.PackedSequenceSpecs` - Packed sequence configuration
- {py:func}`bridge.data.datasets.packed_sequence.prepare_packed_sequence_data` - Data preparation function
- {py:class}`bridge.data.datasets.sft.GPTSFTPackedDataset` - Packed sequence dataset implementation
- {py:class}`bridge.data.builders.finetuning_dataset.FinetuningDatasetBuilder` - Dataset builder with packing support
- {py:func}`bridge.training.gpt_step.get_packed_seq_params` - Packed sequence parameter extraction for training
