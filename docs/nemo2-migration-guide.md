# NeMo 2.0 to Megatron Bridge Migration Guide

This guide helps you migrate from NeMo 2.0 training and recipes to Megatron Bridge. Megatron Bridge retains the Pythonic, code-first API that NeMo 2.0 developed while simplifying configuration into a single {py:class}`bridge.training.config.ConfigContainer` with typed sub-configs. Model parallelism and performance features from Megatron Core remain first-class.

## What Stays the Same

- **Megatron Core Foundation**: Megatron Bridge uses the same Megatron Core engine under the hood.
- **Model Parallelism**: Same TP/PP/CP/EP concepts with identical distributed training semantics.
- **High-Performance Features**: Mixed Precision, communication overlap, and other performance features are supported natively.
- **Pythonic API Retained**: Megatron Bridge preserves NeMo 2.0's philosophy of "configuration as code."


## Model Configuration Mapping

Megatron Bridge offers model providers that directly map to NeMo 2.0 model configs.

### Examples

| NeMo 2.0 | Megatron Bridge |
|----------|-----------------|
| `llm.Llama3Config8B` | {py:class}`bridge.models.Llama3ModelProvider8B` |
| `llm.Llama31Config70B` | {py:class}`bridge.models.Llama31ModelProvider70B` |
| `llm.Qwen2Config7B` | {py:class}`bridge.models.Qwen2ModelProvider7B` |
| `llm.DeepseekV2Config` | {py:class}`bridge.models.DeepseekV2ModelProvider` |

### Supported Model Families

Megatron Bridge supports the following model families with preset providers:
- **Base Models**: `GPTModelProvider`, `T5ModelProvider`, `MambaProvider`
- **Llama**: Llama2, Llama3, Llama3.1, Llama3.2, CodeLlama, Llama4
- **Qwen**: Qwen2, Qwen2.5, Qwen3, Qwen3MoE, Qwen2.5VL
- **DeepSeek**: DeepSeek, DeepSeekV2, DeepSeekV2Lite, DeepSeekV3, Moonlight
- **Nemotron**: Nemotron3, Nemotron4, NemotronH, NemotronNano
- **NVIDIA Mamba**: Mamba variants and hybrid models

For a complete list of all model providers and their parameters, see {py:mod}`bridge.models`.

<!-- TODO: Create a dedicated model support table with tested HF checkpoint mappings -->

---

## Quick Start: Migration Examples

This section shows complete migration examples for common training scenarios. For detailed configuration mappings, see [Configuration Migration](#configuration-migration). For entry point API details, see [Entry Points](#entry-points-pretrain-and-finetune).

### Pretraining Migration Example

#### Before: NeMo 2.0
```python
from nemo import lightning as nl
from nemo.collections import llm
import nemo_run as run
from megatron.core.distributed import DistributedDataParallelConfig

# Model configuration
model = run.Config(
    llm.LlamaModel,
    config=run.Config(llm.Llama3Config8B),  # Preset config with all defaults
)

# Strategy with parallelism settings
strategy = run.Config(
    nl.MegatronStrategy,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    context_parallel_size=1,
    sequence_parallel=False,
    ddp=run.Config(
        DistributedDataParallelConfig,
        grad_reduce_in_fp32=True,
    ),
)

# Trainer setup
trainer = run.Config(
    nl.Trainer,
    max_steps=1000,
    val_check_interval=100,
    limit_val_batches=50,
    log_every_n_steps=10,
    devices=8,
    num_nodes=1,
    strategy=strategy,
    plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
)

# Data configuration
data = run.Config(
    llm.PreTrainingDataModule,
    paths="/path/to/data_text_document",
    seq_length=8192,
    micro_batch_size=1,
    global_batch_size=512,
)

# Optimizer configuration
optim = llm.distributed_fused_adam_with_cosine_annealing(
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=100,
)

# Execute training
llm.pretrain(model, data, trainer, optim=optim)
```

#### Now: Megatron Bridge
```python
# Megatron Bridge configuration pattern
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    TrainingConfig,
)
from megatron.bridge.models import Llama3ModelProvider8B  # Direct equivalent to Llama3Config8B
from megatron.core.optimizer import OptimizerConfig
from megatron.bridge.training.config import SchedulerConfig
from megatron.bridge.training import pretrain
# Use the provided GPT forward step
from megatron.bridge.training.gpt_step import forward_step

def create_config():
    return ConfigContainer(
        # Model with parallelism built-in - using preset 8B config
        model=Llama3ModelProvider8B(
            # Parallelism settings (moved from MegatronStrategy)
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            sequence_parallel=False,
            # Can still override any model params if needed
            seq_length=8192,
        ),
        # Training loop configuration
        train=TrainingConfig(
            global_batch_size=512,
            micro_batch_size=1,
            train_iters=1000,           # was max_steps
            eval_interval=100,          # was val_check_interval
            eval_iters=50,              # was limit_val_batches
        ),
        # Optimization and scheduling
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=3e-4,
            min_lr=3e-5,
            use_distributed_optimizer=True,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="cosine",
            lr_warmup_iters=100,
            lr_decay_iters=1000,
        ),
        # Data configuration
        dataset=GPTDatasetConfig(
            blend=["/path/to/data_text_document"],
            sequence_length=8192,
        ),
        # Checkpointing and logging  
        checkpoint=CheckpointConfig(
            save="/path/to/checkpoints",
            save_interval=100,
            ckpt_format="torch_dist",
        ),
        logger=LoggerConfig(log_interval=10),  # was log_every_n_steps
        # Mixed precision
        mixed_precision="bf16_mixed",
    )

# Execute training
cfg = create_config()

pretrain(cfg, forward_step_func=forward_step)
```

### Fine-Tuning Migration Example (SFT/PEFT)

For fine-tuning, use {py:class}`bridge.training.config.FinetuningDatasetConfig` for data and set `checkpoint.pretrained_checkpoint` to the base model. Optionally add a `peft` configuration for parameter-efficient training.

#### Before: NeMo 2.0
```python
from nemo import lightning as nl
from nemo.collections import llm
import nemo_run as run

# Model and trainer configuration
model = run.Config(llm.LlamaModel, config=run.Config(llm.Llama3Config8B))
trainer = run.Config(
    nl.Trainer,
    max_steps=500, 
    val_check_interval=100,
    devices=8,
    num_nodes=1,
)

# Data configuration
data = run.Config(
    llm.FineTuningDataModule,
    dataset_root="/path/to/sft/data",
    seq_length=2048,
    micro_batch_size=1,
    global_batch_size=128,
)

# PEFT configuration
lora = llm.peft.LoRA(
    target_modules=['linear_qkv', 'linear_proj'],
    dim=32,
    alpha=16,
)

# Execute fine-tuning with PEFT
llm.finetune(
    model=model,
    data=data,
    trainer=trainer,
    peft=lora,
    tokenizer="model",
)
```

#### Now: Megatron Bridge
```python  
# Megatron Bridge fine-tuning configuration (with optional PEFT)
from megatron.bridge.models import Llama3ModelProvider8B
from megatron.bridge.peft import LoRA

def create_finetune_config():
    return ConfigContainer(
        model=Llama3ModelProvider8B(
            # Preset config matching Llama3Config8B
        ),
        train=TrainingConfig(
            micro_batch_size=1,
            global_batch_size=128,
            train_iters=500,
        ),
        # Finetuning dataset instead of pretraining dataset
        dataset=FinetuningDatasetConfig(
            dataset_root="/path/to/sft/data",
            seq_length=2048,
            do_validation=True,
            do_test=True,
            # Optional: packed sequence support
            packed_sequence_specs=PackedSequenceSpecs(
                packed_sequence_size=2048,
            ),
        ),
        # Must specify pretrained checkpoint
        checkpoint=CheckpointConfig(
            pretrained_checkpoint="/path/to/pretrained/model",
            save="/path/to/sft/checkpoints",
            load="/path/to/sft/checkpoints",
            save_interval=50,
        ),
        # Optional: Enable PEFT
        peft=LoRA(
            target_modules=["linear_qkv", "linear_proj"],
            dim=32,
            alpha=16,
        ),
        # ... other configs
    )
```

---

## Recipe Migration

NeMo 2.0 and Megatron Bridge both provide pre-built recipes for popular models. In NeMo 2.0, recipes return `run.Partial` configurations. Megatron Bridge recipes return `ConfigContainer` objects.

### Using Pre-Built Recipes

Both frameworks offer ready-to-use recipes that you can customize:

**NeMo 2.0**: Recipes in `nemo.collections.llm.recipes/`
```python
from nemo.collections import llm

# Use pre-built recipe
recipe = llm.llama3_8b.pretrain_recipe(name="my_run", num_nodes=2)
```

**Megatron Bridge**: Recipes in `megatron.bridge.recipes/`  
```python
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training import pretrain
from megatron.bridge.training.gpt_step import forward_step

# Use pre-built recipe
cfg = pretrain_config()

# Customize as needed
cfg.train.train_iters = 10000
cfg.model.tensor_model_parallel_size = 4

# Launch training
pretrain(cfg, forward_step_func=forward_step)
```

For details on using and customizing recipes, see {doc}`recipe-usage`.

### Migrating a Custom Recipe

If you've created a custom NeMo 2.0 recipe, here's how to migrate it to Megatron Bridge:

#### Before: NeMo 2.0 Recipe Structure

```python
# nemo/collections/llm/recipes/llama3_8b.py
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm

@run.cli.factory(name="llama3_8b")
def model() -> run.Config[pl.LightningModule]:
    return run.Config(llm.LlamaModel, config=run.Config(llm.Llama3Config8B))

def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1000,
) -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
    )
    return run.Config(
        nl.Trainer,
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        max_steps=max_steps,
        strategy=strategy,
        val_check_interval=100,
        limit_val_batches=50,
    )

@run.cli.factory(target=llm.pretrain, name="llama3_8b")
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    return run.Partial(
        llm.pretrain,
        model=model(),
        trainer=trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node),
        data=run.Config(
            llm.PreTrainingDataModule,
            paths="/path/to/data_text_document",
            seq_length=8192,
            global_batch_size=512,
            micro_batch_size=1,
        ),
        log=llm.default_log(dir=dir, name=name),
        optim=llm.distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=llm.default_resume(),
    )

# Usage
if __name__ == "__main__":
    recipe = pretrain_recipe(name="my_run", num_nodes=2)
    # Submitted via nemo-run or executed directly
```

#### Now: Megatron Bridge Recipe Structure

```python
# my_recipes/llama3_8b.py
from typing import Optional
from megatron.bridge.training.config import (
    ConfigContainer,
    TrainingConfig,
    GPTDatasetConfig,
    CheckpointConfig,
    SchedulerConfig,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.bridge.models import Llama3ModelProvider8B
from megatron.bridge.training import pretrain

def llama3_8b_config(
    # Model/parallelism params
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    # Training params
    train_iters: int = 1000,
    eval_interval: int = 100,
    eval_iters: int = 50,
    # Data params
    data_path: str = "/path/to/data_text_document",
    seq_length: int = 8192,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    # Checkpoint params
    checkpoint_dir: Optional[str] = None,
    save_interval: int = 1000,
) -> ConfigContainer:
    """Create a Llama3 8B pretraining configuration."""
    return ConfigContainer(
        model=Llama3ModelProvider8B(
            # Preset architecture from Llama3Config8B (num_layers=32, hidden_size=4096, etc.)
            # Only need to specify parallelism and overrides
            tensor_model_parallel_size=tensor_parallelism,
            pipeline_model_parallel_size=pipeline_parallelism,
        ),
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        dataset=GPTDatasetConfig(
            blend=[data_path],
            sequence_length=seq_length,
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=3e-4,
            use_distributed_optimizer=True,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="cosine",
            lr_warmup_iters=100,
        ),
    checkpoint=CheckpointConfig(
            save=checkpoint_dir or "/results/checkpoints",
            save_interval=save_interval,
        ),
        mixed_precision="bf16-mixed",
    )

# Usage
if __name__ == "__main__":
    from megatron.bridge.training.gpt_step import forward_step
    
    cfg = llama3_8b_config(
        train_iters=10000,
        checkpoint_dir="/my/checkpoints",
        tensor_parallelism=2,
    )
    pretrain(cfg, forward_step_func=forward_step)
```

**Migration steps:**
1. Replace `run.Partial` with a function returning `ConfigContainer`
2. Move all `trainer`, `strategy`, and distributed settings into model provider
3. Consolidate `log`, `optim`, `resume` into respective config objects
4. Remove `@run.cli.factory` decorators (optional: use your own CLI framework)
5. Launch with `torchrun` or similar launcher—device count no longer passed to training function

---

## Configuration Migration

### Overview
What used to be configured across Lightning `Trainer` arguments, callbacks, and `MegatronStrategy` parameters is now centralized into a set of configuration classes:

| Configuration Area | Megatron Bridge Config Class |
|-------------------|-------------------|
| Training loop settings | {py:class}`bridge.training.config.TrainingConfig` |
| Checkpointing | {py:class}`bridge.training.config.CheckpointConfig` |
| Logging and monitoring | {py:class}`bridge.training.config.LoggerConfig` |
| Distributed training initialization | {py:class}`bridge.training.config.DistributedInitConfig` |
| Mixed precision | {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` |
| Performance profiling | {py:class}`bridge.training.config.ProfilingConfig` |

For detailed documentation on each configuration area, see the training documentation:
- {doc}`training/config-container-overview` - Overview of the configuration system
- {doc}`training/training-loop-settings` - Training loop parameters and validation
- {doc}`training/checkpointing` - Checkpointing and model persistence
- {doc}`training/optimizer-scheduler` - Optimization and learning rate scheduling
- {doc}`training/logging` - Logging, TensorBoard, and Weights & Biases
- {doc}`training/profiling` - Performance profiling with Nsys and PyTorch



### Training Configuration Migration
Lightning `Trainer` parameters are now managed through dedicated configuration classes.

| **Setting Category** | **NeMo 2.0 Location** | **Megatron Bridge Location** | **Details** |
|---------------------|----------------------|-------------------|-------------|
| **Training iterations** | `trainer.max_steps` | {py:attr}`bridge.training.config.TrainingConfig.train_iters` | Total training iterations |
| **Validation frequency** | `trainer.val_check_interval` | {py:attr}`bridge.training.config.TrainingConfig.eval_interval` | Steps between validation runs |
| **Validation iterations** | `trainer.limit_val_batches` | {py:attr}`bridge.training.config.TrainingConfig.eval_iters` | Number of validation steps per run |
| **Test iterations** | `trainer.limit_test_batches` | {py:attr}`bridge.training.config.TrainingConfig.eval_iters` | Number of test steps (shares eval_iters) |
| **Logging frequency** | `trainer.log_every_n_steps` | {py:attr}`bridge.training.config.LoggerConfig.log_interval` | Logging frequency |

#### Before: NeMo 2.0
```python
trainer = run.Config(
    nl.Trainer,
    max_steps=1000,
    val_check_interval=100,     # validation frequency
    limit_val_batches=50,       # validation iterations per run
    limit_test_batches=100,     # test iterations
    log_every_n_steps=10,
)
```

#### Now: Megatron Bridge
```python  
train_config = TrainingConfig(
    train_iters=1000,           # was max_steps
    eval_interval=100,          # was val_check_interval  
    eval_iters=50,              # was limit_val_batches (for both val and test)
)
logger_config = LoggerConfig(log_interval=10)  # was log_every_n_steps
```

### Data Configuration Migration

NeMo 2.0 uses `PreTrainingDataModule` and `FineTuningDataModule` classes. Megatron Bridge uses configuration objects: {py:class}`bridge.training.config.GPTDatasetConfig` for pretraining and {py:class}`bridge.training.config.FinetuningDatasetConfig` for fine-tuning.

#### Pretraining Data

##### Before: NeMo 2.0 PreTrainingDataModule

```python
from nemo.collections.llm.gpt.data import PreTrainingDataModule

# Single dataset
data = PreTrainingDataModule(
    paths="/path/to/train_data_text_document",
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=512,
    num_workers=8,
    split="949,50,1",  # train/val/test split ratios
)

# Multiple datasets with weights
data = PreTrainingDataModule(
    paths=["30", "/path/to/dataset1_text_document", 
           "70", "/path/to/dataset2_text_document"],
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=512,
    split="949,50,1",
)

# Separate train/val/test datasets
data = PreTrainingDataModule(
    paths={
        "train": ["/path/to/train_data_text_document"],
        "validation": ["/path/to/val_data_text_document"],
        "test": ["/path/to/test_data_text_document"],
    },
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=512,
)
```

##### Now: Megatron Bridge GPTDatasetConfig

```python
from megatron.bridge.training.config import GPTDatasetConfig, TrainingConfig

# Single dataset
dataset_config = GPTDatasetConfig(
    blend=["/path/to/train_data_text_document"],
    sequence_length=4096,
    split="949,50,1",
)
train_config = TrainingConfig(
    micro_batch_size=1,
    global_batch_size=512,
)

# Multiple datasets with weights (blending)
dataset_config = GPTDatasetConfig(
    blend=[
        "/path/to/dataset1_text_document",
        "/path/to/dataset2_text_document",
    ],
    blend_weights=[0.3, 0.7],  # Explicit weights (not zipped with paths)
    sequence_length=4096,
    split="949,50,1",
)
```

**Key differences:**
- NeMo 2.0's `paths` → Megatron Bridge's `blend`
- NeMo 2.0's zipped list `["30", "path1", "70", "path2"]` → Megatron Bridge's separate `blend` and `blend_weights`
- Batch sizes move from data module to `TrainingConfig`
- Dataloader options (`num_workers`, `pin_memory`, etc.) available in both configs

#### Fine-Tuning Data

##### Before: NeMo 2.0 FineTuningDataModule

```python
from nemo.collections.llm.gpt.data import FineTuningDataModule

data = FineTuningDataModule(
    dataset_root="/path/to/instruction_data",
    seq_length=2048,
    micro_batch_size=1,
    global_batch_size=128,
    num_workers=8,
)
```

##### Now: Megatron Bridge FinetuningDatasetConfig

```python
from megatron.bridge.training.config import FinetuningDatasetConfig, TrainingConfig

dataset_config = FinetuningDatasetConfig(
    dataset_root="/path/to/instruction_data",
    seq_length=2048,
    do_validation=True,
    do_test=False,
    # Dataloader options (inherited from DataloaderConfig)
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
)
train_config = TrainingConfig(
    micro_batch_size=1,
    global_batch_size=128,
)
```

**Key differences:**
- Batch sizes move to `TrainingConfig`
- Explicit control over finetuning validation/test splits via `do_validation` and `do_test`
- Dataloader options (`num_workers`, `pin_memory`, etc.) available via `FinetuningDatasetConfig`


### Tokenizer Migration

Megatron Bridge uses {py:class}`bridge.training.tokenizers.config.TokenizerConfig` for consistent tokenizer setup across different model types.

#### Before: NeMo 2.0
```python
# Option 1: Using get_nmt_tokenizer utility
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

tokenizer = get_nmt_tokenizer(
    library="megatron",
    model_name="GPT2BPETokenizer",
    vocab_file="/path/to/vocab.json",
    merges_file="/path/to/merges.txt",
)

# Option 2: Using run.Config with tokenizer classes
import nemo_run as run
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

tokenizer = run.Config(
    AutoTokenizer,
    pretrained_model_name="meta-llama/Llama-3-8B",
)
```

#### Now: Megatron Bridge
```python
# Dedicated tokenizer configuration
from megatron.bridge.training.tokenizers.config import TokenizerConfig

# GPT2 BPE Tokenizer
tokenizer_config = TokenizerConfig(
    tokenizer_type="GPT2BPETokenizer",
    vocab_file="/path/to/vocab.json", 
    merge_file="/path/to/merges.txt",
)

# HuggingFace Tokenizer
tokenizer_config = TokenizerConfig(
    tokenizer_type="HuggingFaceTokenizer",
    tokenizer_model="meta-llama/Llama-3-8B",
)
```

#### Vocab Size Priority

In Megatron Bridge, vocabulary size can be specified in either the model provider or derived from the tokenizer. The priority order is:

1. **Model provider `vocab_size` is set**: Uses the model's vocab size
   - Must be `>= tokenizer.vocab_size` (raises error if smaller)
   - Sets `should_pad_vocab=False` (no automatic padding)
   - Useful when you need a specific vocab size (e.g., for checkpoint compatibility)

2. **Model provider `vocab_size` is None**: Uses tokenizer's vocab size
   - Automatically derived from `tokenizer.vocab_size` after building the tokenizer.
   - Sets `should_pad_vocab=True` (enables padding for efficient parallelism)

```python
# Option 1: Let tokenizer determine vocab size
config = ConfigContainer(
    model=Llama3ModelProvider8B(
        # vocab_size not set - will use tokenizer's vocab size
        vocab_size=None,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="meta-llama/Llama-3-8B",
    ),
)

# Option 2: Explicitly set vocab size in model
config = ConfigContainer(
    model=Llama3ModelProvider8B(
        vocab_size=128256,  # Explicitly set (must be >= tokenizer vocab size)
    ),
    tokenizer=TokenizerConfig(...),
)
```


### Parallelism Configuration Migration
In NeMo 2.0, parallelism settings were configured on `MegatronStrategy`. In Megatron Bridge, these are set directly on the model provider:

| **Parallelism Type** | **NeMo 2.0** | **Megatron Bridge** |
|---------------------|-------------|-----------|
| **Tensor Parallel** | `strategy.tensor_model_parallel_size` | `model.tensor_model_parallel_size` |
| **Pipeline Parallel** | `strategy.pipeline_model_parallel_size` | `model.pipeline_model_parallel_size` |
| **Virtual Pipeline** | `strategy.virtual_pipeline_model_parallel_size` | `model.virtual_pipeline_model_parallel_size` |
| **Microbatch Group Size** | `strategy.microbatch_group_size_per_vp_stage` | `model.microbatch_group_size_per_vp_stage` |
| **Pipeline Layer Distribution** | `strategy.num_layers_in_first_pipeline_stage` | `model.num_layers_in_first_pipeline_stage` |
| **Pipeline Layer Distribution** | `strategy.num_layers_in_last_pipeline_stage` | `model.num_layers_in_last_pipeline_stage` |
| **Context Parallel** | `strategy.context_parallel_size` | `model.context_parallel_size` |
| **Sequence Parallel** | `strategy.sequence_parallel` | `model.sequence_parallel` |
| **Expert Parallel** | `strategy.expert_model_parallel_size` | `model.expert_model_parallel_size` |
| **Expert Tensor Parallel** | `strategy.expert_tensor_parallel_size` | `model.expert_tensor_parallel_size` |
| **Pipeline Layout** | `strategy.pipeline_model_parallel_layout` | `model.pipeline_model_parallel_layout` |
| **Pipeline Comm Backend** | `strategy.pipeline_model_parallel_comm_backend` | `model.pipeline_model_parallel_comm_backend` |
| **Pipeline Dtype** | `strategy.pipeline_dtype` | `model.pipeline_dtype` |
| **Encoder Tensor Parallel** | `strategy.encoder_tensor_model_parallel_size` | `model.encoder_tensor_model_parallel_size` |
| **Encoder Pipeline Parallel** | `strategy.encoder_pipeline_model_parallel_size` | `model.encoder_pipeline_model_parallel_size` |
| **Embedding in Pipeline** | `strategy.account_for_embedding_in_pipeline_split` | `model.account_for_embedding_in_pipeline_split` |
| **Loss in Pipeline** | `strategy.account_for_loss_in_pipeline_split` | `model.account_for_loss_in_pipeline_split` |
| **TE RNG Tracker** | `strategy.use_te_rng_tracker` | `model.use_te_rng_tracker` |

#### Before: NeMo 2.0
```python
strategy = run.Config(
    MegatronStrategy,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    sequence_parallel=True,
)
```

#### Now: Megatron Bridge
```python
model = GPTModelProvider(
    # Model architecture
    num_layers=32,
    hidden_size=4096,
    # Parallelism co-located with model
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    sequence_parallel=True,
)
```

### DDP Configuration Migration
Some `MegatronStrategy` parameters move to {py:class}`bridge.training.config.DistributedDataParallelConfig`:

| **Setting** | **NeMo 2.0** | **Megatron Bridge** |
|-------------|-------------|-----------|
| **Distributed Optimizer Instances** | `strategy.num_distributed_optimizer_instances` | {py:attr}`bridge.training.config.DistributedDataParallelConfig.num_distributed_optimizer_instances` |

### Strategy Settings Migration
Additional `MegatronStrategy` parameters move to {py:class}`bridge.training.config.DistributedInitConfig`:

| **Setting** | **NeMo 2.0** | **Megatron Bridge** |
|-------------|-------------|-----------|
| **Process Groups** | `strategy.use_gloo_process_groups` | {py:attr}`bridge.training.config.DistributedInitConfig.use_gloo_process_groups` |
| **SHARP** | `strategy.use_sharp` | {py:attr}`bridge.training.config.DistributedInitConfig.use_sharp` |
| **NCCL Config** | `strategy.nccl_communicator_config_path` | {py:attr}`bridge.training.config.DistributedInitConfig.nccl_communicator_config_path` |
| **Mapping Order** | `strategy.use_tp_pp_dp_mapping` | {py:attr}`bridge.training.config.DistributedInitConfig.use_tp_pp_dp_mapping` |
| **Lazy Init** | `strategy.lazy_init` | {py:attr}`bridge.training.config.DistributedInitConfig.lazy_init` |


### Mixed Precision Migration
Mixed precision in NeMo 2.0 is controlled via precision plugins passed to the trainer. In Megatron Bridge, this moves to a dedicated configuration class:

#### Before: NeMo 2.0
```python
# Mixed precision via plugin
from nemo.lightning.pytorch.plugins import MegatronMixedPrecisionPlugin

trainer = run.Config(
    nl.Trainer,
    plugins=[MegatronMixedPrecisionPlugin(precision="bf16-mixed")]
)
```

#### Now: Megatron Bridge
```python
# Option 1: Use preset strings
config = ConfigContainer(
    mixed_precision="bf16_mixed",  # Simple preset
    # ... other configs
)

# Option 2: Detailed configuration
config = ConfigContainer(
    mixed_precision=MixedPrecisionConfig(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    ),
    # ... other configs
)
```

```{Note}
The `mixed_precision` configuration automatically synchronizes precision settings across model, optimizer, and DDP configurations, overriding any conflicting settings. This ensures consistent precision behavior throughout training. For details on configuration precedence and available recipes, see {doc}`training/mixed-precision`.
```

### Checkpointing Configuration Migration
Checkpointing configuration moves from `MegatronStrategy` parameters and `ModelCheckpoint` callback to {py:class}`bridge.training.config.CheckpointConfig`:

| **Checkpoint Setting** | **NeMo 2.0** | **Megatron Bridge** |
|------------------------|-------------|-----------|
| **Save directory** | `ModelCheckpoint(dirpath=...)` | {py:attr}`bridge.training.config.CheckpointConfig.save` |
| **Load directory** | `trainer.ckpt_path` | {py:attr}`bridge.training.config.CheckpointConfig.load` |
| **Pretrained checkpoint (for finetuning)** | `AutoResume.import_path` or manually load | {py:attr}`bridge.training.config.CheckpointConfig.pretrained_checkpoint` |
| **Save frequency** | `ModelCheckpoint(every_n_train_steps=...)` | {py:attr}`bridge.training.config.CheckpointConfig.save_interval` |
| **Save top-k** | `ModelCheckpoint(save_top_k=...)` | No direct equivalent - Megatron Bridge can keep the most recent checkpoints |
| **Most recent checkpoints** | No direct equivalent | {py:attr}`bridge.training.config.CheckpointConfig.most_recent_k` |
| **Save last** | `ModelCheckpoint(save_last=...)` | Always enabled in Megatron Bridge |
| **Checkpoint format** | `strategy.save_ckpt_format` | {py:attr}`bridge.training.config.CheckpointConfig.ckpt_format` |
| **Async saving** | `strategy.ckpt_async_save` | {py:attr}`bridge.training.config.CheckpointConfig.async_save` |
| **Parallel save** | `strategy.ckpt_parallel_save` | {py:attr}`bridge.training.config.CheckpointConfig.fully_parallel_save` |
| **Parallel load** | `strategy.ckpt_parallel_load` | {py:attr}`bridge.training.config.CheckpointConfig.fully_parallel_load` |
| **Load optimizer** | `strategy.ckpt_load_optimizer` | {py:attr}`bridge.training.config.CheckpointConfig.load_optim` |
| **Save optimizer** | `strategy.ckpt_save_optimizer` | {py:attr}`bridge.training.config.CheckpointConfig.save_optim` |
| **Load main params** | `strategy.ckpt_load_main_params` | {py:attr}`bridge.training.config.CheckpointConfig.load_main_params_from_ckpt` |
| **Save weights only** | `ModelCheckpoint(save_weights_only=...)` | Inverse of `save_optim` |
| **Load strictness** | `strategy.ckpt_load_strictness` | {py:attr}`bridge.training.config.CheckpointConfig.dist_ckpt_strictness` |
| **Assume constant structure** | `strategy.ckpt_assume_constant_structure` | {py:attr}`bridge.training.config.CheckpointConfig.ckpt_assume_constant_structure` |
| **Save optim on train end** | `ModelCheckpoint(save_optim_on_train_end=...)` | Controlled by `save_optim` |
| **Resume from directory** | `AutoResume(resume_from_directory=...)` | {py:attr}`bridge.training.config.CheckpointConfig.load` |
| **Resume if exists** | `AutoResume(resume_if_exists=...)` | Automatic if `load` is set |
| **Resume ignore no checkpoint** | `AutoResume(resume_ignore_no_checkpoint=...)` | {py:attr}`bridge.training.config.CheckpointConfig.exit_on_missing_checkpoint` (inverse) |

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning import AutoResume, NeMoLogger

# ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="/path/to/checkpoints",
    every_n_train_steps=1000,
    save_top_k=3,           # Saves best 3 checkpoints based on monitored metric
    save_last=True,
    save_weights_only=False,
    monitor="val_loss",     # Metric to monitor for top-k selection
)

# AutoResume for checkpoint resumption
resume = AutoResume(
    resume_if_exists=True,
    resume_ignore_no_checkpoint=True,
    resume_from_directory="/path/to/checkpoints"
)

# NeMoLogger ties everything together
logger = NeMoLogger(
    log_dir="/path/to/logs",
    name="my_experiment", 
    ckpt=checkpoint_callback,
)

# MegatronStrategy parameters
strategy = run.Config(
    MegatronStrategy,
    save_ckpt_format="torch_dist",
    ckpt_async_save=True,
    ckpt_parallel_save=True,
    ckpt_load_optimizer=True,
    ckpt_save_optimizer=True,
    ckpt_load_strictness=None,
)

trainer = nl.Trainer(strategy=strategy)
logger.setup(trainer, resume.resume_if_exists)
resume.setup(trainer)
```

#### Now: Megatron Bridge
```python
checkpoint_config = CheckpointConfig(
    # Saving configuration
    save="/path/to/checkpoints",
    save_interval=1000,
    most_recent_k=3,        # Keeps 3 most recent checkpoints (not metric-based)
    save_optim=True,
    save_rng=True,
    
    # Loading/resumption configuration
    load="/path/to/checkpoints",  # Resume from this directory (if exists)
    load_optim=True,               # Load optimizer state
    exit_on_missing_checkpoint=False,  # Don't exit if no checkpoint found (was resume_ignore_no_checkpoint)
    
    # Format and performance options
    ckpt_format="torch_dist",
    async_save=True,
    fully_parallel_save=True,
    fully_parallel_load=True,
    dist_ckpt_strictness="assume_ok_unexpected",
)
```

**Key differences:**
- **Resume behavior**: Setting `load` enables automatic resume if checkpoint exists (no separate `AutoResume` needed)
- **Pretrained checkpoint**: Use `pretrained_checkpoint` to specify base model weights for fine-tuning (loaded before training starts)
- **Top-k**: NeMo 2.0's `save_top_k` monitors metrics; Megatron Bridge's `most_recent_k` keeps recent checkpoints
- **Configuration location**: All checkpoint settings unified in one config (not split across callback, logger, and strategy)

```{Important}
All checkpoint paths (`save`, `load`, `pretrained_checkpoint`) must point to **Megatron-format checkpoints**. Hugging Face checkpoints cannot be used directly—convert them first using {py:meth}`bridge.models.conversion.auto_bridge.AutoBridge.import_ckpt`. See {doc}`bridge-guide` for conversion details.
```

For comprehensive documentation on checkpoint formats, local checkpointing, fault tolerance, and advanced features, see {doc}`training/checkpointing`.

---

### Optimizer and LR Scheduler Migration

Optimization configuration moves from NeMo 2.0's `MegatronOptimizerModule` approach to Megatron Bridge's direct {py:class}`megatron.core.optimizer.OptimizerConfig` and {py:class}`bridge.training.config.SchedulerConfig`.

#### Before: NeMo 2.0
```python
# NeMo 2.0 optimizer configuration with MegatronOptimizerModule
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.collections.llm.recipes import distributed_fused_adam_with_cosine_annealing

# Option 1: Using recipe helper functions
optim_config = distributed_fused_adam_with_cosine_annealing(
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
)

# Option 2: Direct MegatronOptimizerModule
optim = MegatronOptimizerModule(
    config=OptimizerConfig(
        optimizer="adam",
        lr=3e-4,
        use_distributed_optimizer=True,
    ),
    lr_scheduler=CosineAnnealingScheduler(
        warmup_steps=2000,
        constant_steps=0,
        decay_steps=100000,
    )
)
```

#### Now: Megatron Bridge
```python
# Megatron Bridge direct configuration
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing

# Option 1: Using utility functions
optimizer_config, scheduler_config = distributed_fused_adam_with_cosine_annealing(
    max_lr=3e-4,
    min_lr=3e-5,
    lr_warmup_iters=2000,
    lr_decay_iters=100000,
)

# Option 2: Direct configuration
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    use_distributed_optimizer=True,
)

scheduler_config = SchedulerConfig(
    lr_decay_style="cosine",
    lr_warmup_iters=2000,
    lr_decay_iters=100000,
)
```

### Logging Configuration Migration

NeMo 2.0 uses `NeMoLogger` for TensorBoard and Weights & Biases (W&B) integration. Megatron Bridge consolidates logging configuration in {py:class}`bridge.training.config.LoggerConfig`.

#### Before: NeMo 2.0

```python
from nemo.lightning import NeMoLogger

logger = NeMoLogger(
    log_dir="/path/to/logs",
    name="my_experiment",
    use_datetime_version=True,
    tensorboard=dict(
        log_dir="/path/to/tensorboard",
    ),
    wandb=dict(
        project="my_project",
        name="my_run",
        entity="my_team",
    ),
)
```

#### Now: Megatron Bridge

```python
from megatron.bridge.training.config import LoggerConfig

logger_config = LoggerConfig(
    # General logging
    log_interval=10,              # Log metrics every N iterations
    log_throughput=True,          # Log throughput per GPU
    
    # TensorBoard configuration
    tensorboard_dir="/path/to/tensorboard",
    tensorboard_log_interval=1,   # Write to TensorBoard every N iterations
    log_timers_to_tensorboard=False,
    log_validation_ppl_to_tensorboard=False,
    
    # Weights & Biases configuration
    wandb_project="my_project",
    wandb_exp_name="my_run",
    wandb_entity="my_team",
    wandb_save_dir="/path/to/wandb",
)
```

**Key differences:**
- TensorBoard and W&B configuration unified in single `LoggerConfig`
- Fine-grained control over what gets logged (timers, memory, validation perplexity, etc.)
- No separate `NeMoLogger.setup()` call needed

For more details on logging configuration and available options, see {doc}`training/logging`.

### Profiling Configuration Migration

Megatron Bridge centralizes all profiling functionality in {py:class}`bridge.training.config.ProfilingConfig`, replacing multiple NeMo callbacks.

#### Nsys Profiling Migration

##### Before: NeMo 2.0
```python
# NeMo 2.0 used NsysCallback
from nemo.lightning.pytorch.callbacks import NsysCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        NsysCallback(
            start_step=100,
            end_step=110,
            ranks=[0],
            gen_shape=True
        )
    ]
)
```

##### Now: Megatron Bridge
```python
# Megatron Bridge uses ProfilingConfig  
profiling_config = ProfilingConfig(
    use_nsys_profiler=True,
    profile_step_start=100,
    profile_step_end=110,
    profile_ranks=[0],
    record_shapes=True,
)
```

#### PyTorch Profiler Migration

##### Before: NeMo 2.0
```python
# NeMo 2.0 used PytorchProfilerCallback
from nemo.lightning.pytorch.callbacks import PytorchProfilerCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        PytorchProfilerCallback(
            start_step=100,
            end_step=110,
            warmup_steps=1,
            active_steps=5,
            trace_dir="/path/to/traces",
        )
    ]
)
```

##### Now: Megatron Bridge
```python
# Megatron Bridge uses ProfilingConfig
profiling_config = ProfilingConfig(
    use_pytorch_profiler=True,
    profile_step_start=100,
    profile_step_end=110,
    profile_ranks=[0],
    record_memory_history=True,
    memory_snapshot_path="memory_profile.pickle",
)
```

### PEFT Configuration Migration

PEFT (Parameter-Efficient Fine-Tuning) enables fine-tuning with a small fraction of trainable parameters by freezing the base model and training only adapter modules.

#### Before: NeMo 2.0

```python
from nemo.collections import llm
import nemo_run as run

# Create PEFT configuration
lora = llm.peft.LoRA(
    target_modules=['linear_qkv', 'linear_proj'],
    dim=32,
    alpha=16,
    dropout=0.0,
)

# Pass to finetune()
llm.finetune(
    model=model,
    data=data,
    trainer=trainer,
    peft=lora,  # PEFT as argument
)
```

#### Now: Megatron Bridge

```python
from megatron.bridge.peft import LoRA
from megatron.bridge.training.config import ConfigContainer, CheckpointConfig

# Include PEFT in ConfigContainer
config = ConfigContainer(
    model=Llama3ModelProvider8B(),
    # ... other configs
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",  # Required for PEFT
        save="/path/to/peft/checkpoints",
    ),
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=32,
        alpha=16,
        dropout=0.0,
    ),
)
```

**Key differences:**
- PEFT config is part of `ConfigContainer`, not a separate argument to `finetune()`
- Must set `checkpoint.pretrained_checkpoint` when using PEFT (enforced at validation)
- Target module names are the same between NeMo 2.0 and Megatron Bridge

**Supported PEFT methods:**
- **LoRA**: Low-Rank Adaptation via {py:class}`bridge.peft.lora.LoRA`
- **DoRA**: Weight-Decomposed Low-Rank Adaptation via {py:class}`bridge.peft.dora.DoRA`

For comprehensive PEFT documentation including adapter design, checkpoint handling, wildcard targeting, and best practices, see {doc}`training/peft`.

## Entry Points: `pretrain` and `finetune`

NeMo 2.0's `llm.pretrain()` and `llm.finetune()` API functions map directly to Megatron Bridge's entry point functions with unified configuration.

### NeMo 2.0 Entry Points

In NeMo 2.0, you call `llm.pretrain()` or `llm.finetune()` from `nemo.collections.llm.api`:

```python
from nemo.collections import llm
import nemo_run as run

# Pretraining
result = llm.pretrain(
    model=model_config,
    data=data_config,
    trainer=trainer_config,
    log=logger_config,
    resume=resume_config,
    optim=optimizer_config,
)

# Fine-tuning
result = llm.finetune(
    model=model_config,
    data=data_config,
    trainer=trainer_config,
    log=logger_config,
    resume=resume_config,
    optim=optimizer_config,
    peft=peft_config,   # Optional PEFT
    tokenizer="model",  # or "data"
)
```

### Megatron Bridge Entry Points

In Megatron Bridge, training entry points take a single `ConfigContainer` and a `forward_step_func`:

```python
from megatron.bridge.training import pretrain, finetune
from megatron.bridge.training.config import ConfigContainer

# Create unified configuration
cfg = ConfigContainer(
    model=model_provider,
    train=train_config,
    dataset=dataset_config,
    optimizer=optimizer_config,
    scheduler=scheduler_config,
    checkpoint=checkpoint_config,
    logger=logger_config,
    mixed_precision="bf16_mixed",
    # peft=peft_config,  # Optional for fine-tuning
)

# Pretraining
from megatron.bridge.training.gpt_step import forward_step
pretrain(cfg, forward_step_func=forward_step)

# Fine-tuning (same function signature)
finetune(cfg, forward_step_func=forward_step)
```

#### Understanding `forward_step_func`

The `forward_step_func` combines three responsibilities into a single function:

1. **Fetch a batch** from the data iterator
2. **Run the forward pass** through the model  
3. **Define the loss function** to compute loss from the model output

**Signature:**
```python
def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: nn.Module,
) -> tuple[torch.Tensor, Callable]:
    """
    Args:
        state: Global training state (contains config, timers, etc.)
        data_iterator: Iterator over training/validation data
        model: The model to run forward pass on
        
    Returns:
        output_tensor: Model output (logits)
        loss_func: Callable that computes loss from output_tensor
    """
```

For GPT models, use the provided {py:func}`bridge.training.gpt_step.forward_step`. For custom models or specialized training logic, implement your own following this pattern.

**Key differences:**
- All configuration consolidated into single `ConfigContainer` object
- Training mode determined by dataset type and checkpoint configuration, not separate function calls
- Must provide `forward_step_func` that handles batch fetching, forward pass, and loss computation
- No separate `resume`, `log`, `optim` arguments - all configurations are part of the `ConfigContainer`

### `pretrain`

Use `pretrain()` with `GPTDatasetConfig` for training models from scratch:

```python
from megatron.bridge.training import pretrain
from megatron.bridge.training.gpt_step import forward_step

config = ConfigContainer(
    model=Llama3ModelProvider8B(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
    ),
    train=TrainingConfig(
        train_iters=100000,
        eval_interval=1000,
        micro_batch_size=1,
        global_batch_size=512,
    ),
    dataset=GPTDatasetConfig(
        blend=["/path/to/train_data_text_document"],
        sequence_length=4096,
        split="949,50,1",
    ),
    optimizer=OptimizerConfig(optimizer="adam", lr=3e-4),
    checkpoint=CheckpointConfig(save="/path/to/checkpoints", save_interval=1000),
    mixed_precision="bf16_mixed",
)

pretrain(config, forward_step_func=forward_step)
```

### `finetune`

Use `finetune()` with `FinetuningDatasetConfig` for both full fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT):

#### Supervised Fine-Tuning (SFT)

Full fine-tuning without PEFT - all model parameters are updated:

```python
from megatron.bridge.training import finetune
from megatron.bridge.training.gpt_step import forward_step

config = ConfigContainer(
    model=Llama3ModelProvider8B(),
    train=TrainingConfig(
        train_iters=1000,
        eval_interval=100,
        micro_batch_size=1,
        global_batch_size=128,
    ),
    dataset=FinetuningDatasetConfig(
        dataset_root="/path/to/instruction_data",
        seq_length=4096,
        do_validation=True,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",  # Must be Megatron format
        save="/path/to/sft_checkpoints",
    ),
    optimizer=OptimizerConfig(optimizer="adam", lr=1e-5),
    mixed_precision="bf16_mixed",
)

finetune(config, forward_step_func=forward_step)
```

#### Parameter-Efficient Fine-Tuning (PEFT)

Add a `peft` configuration to enable parameter-efficient training:

```python
from megatron.bridge.peft import LoRA

config = ConfigContainer(
    model=Llama3ModelProvider8B(),
    train=TrainingConfig(
        train_iters=1000,
        eval_interval=100,
        micro_batch_size=1,
        global_batch_size=128,
    ),
    dataset=FinetuningDatasetConfig(
        dataset_root="/path/to/instruction_data",
        seq_length=4096,
        do_validation=True,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",
        save="/path/to/peft_checkpoints",
    ),
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=32,
        alpha=16,
    ),
    optimizer=OptimizerConfig(optimizer="adam", lr=1e-4),
    mixed_precision="bf16_mixed",
)

finetune(config, forward_step_func=forward_step)
```

**Converting Hugging Face checkpoints**: If you have a Hugging Face model, convert it to Megatron checkpoint format first:

```python
from megatron.bridge import AutoBridge

# Convert HuggingFace to Megatron format
AutoBridge.import_ckpt(
    "meta-llama/Meta-Llama-3-8B",
    "/path/to/megatron/checkpoint"
)
```

See {doc}`bridge-guide` for more details on model conversion.

### Advanced: Custom Forward Step and Loss Reduction

For comprehensive documentation on entry points, forward step functions, and customization patterns, see {doc}`training/entry-points`.

#### Forward Step Customization

In NeMo 2.0, custom `forward_step` and `data_step` functions can be attached to the model configuration. In Megatron Bridge, the forward step function is passed directly as an argument to `pretrain()` or `finetune()`.

##### NeMo 2.0: Custom Steps Attached to Config

```python
# NeMo 2.0: Define custom functions and attach to model config
import torch

def custom_forward_step(model, batch) -> torch.Tensor:
    """Custom forward step for specialized loss computation."""
    output = model(batch['tokens'], batch['attention_mask'])
    loss = compute_custom_loss(output, batch['labels'])
    return loss

# Attach to config in NeMo 2.0
model_config = llm.Llama3Config8B()
model_config.forward_step_fn = custom_forward_step  # Override default forward step

model = run.Config(llm.LlamaModel, config=model_config)
```

##### Megatron Bridge: Pass Custom Forward Step

```python
# Megatron Bridge: Define and pass forward step function
import torch
from typing import Iterable
from functools import partial
from megatron.bridge.training.state import GlobalState

def custom_forward_step(
    state: GlobalState,
    data_iterator: Iterable, 
    model: torch.nn.Module,
) -> tuple[torch.Tensor, partial]:
    """Custom forward step for specialized loss computation."""
    # Get batch from iterator
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    labels = batch['labels'].cuda()
    loss_mask = batch['loss_mask'].cuda()
    
    # Custom forward logic
    output = model(tokens, attention_mask=batch.get('attention_mask'))
    
    # Define custom loss function
    def loss_func(output_tensor):
        return compute_custom_loss(output_tensor, labels, loss_mask)
    
    return output, loss_func

# Pass to training function
pretrain(cfg, forward_step_func=custom_forward_step)
```

#### Loss Reduction Pattern

NeMo 2.0 uses `MegatronLossReduction` for custom loss computation and reduction across microbatches. Megatron Bridge achieves the same through the loss function returned by `forward_step`.

##### NeMo 2.0: MegatronLossReduction

```python
from nemo.lightning.megatron_parallel import MegatronLossReduction

class CustomLossReduction(MegatronLossReduction):
    def forward(self, batch, forward_out):
        """Compute loss from forward output."""
        loss = compute_loss(forward_out, batch['labels'])
        return loss, {"custom_metric": some_metric}
    
    def reduce(self, losses_reduced_per_micro_batch):
        """Reduce losses across microbatches."""
        losses = [x["custom_metric"] for x in losses_reduced_per_micro_batch]
        return torch.stack(losses).mean()

# Attach to model
model._training_loss_reduction = CustomLossReduction()
```

##### Megatron Bridge: Loss Function Pattern

```python
def custom_forward_step(state, data_iterator, model):
    """Forward step with custom loss reduction."""
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    labels = batch['labels'].cuda()
    loss_mask = batch['loss_mask'].cuda()
    
    output = model(tokens)
    
    def loss_func(output_tensor):
        """Compute and return loss in reduction-friendly format.
        
        Return formats:
        - Single value: loss (averaged over microbatches only)
        - Tuple: (loss, num_tokens) - averaged over microbatches and tokens
        - Dict: {"loss": loss, "custom_metric": value, ...} - for logging
        """
        loss = compute_loss(output_tensor, labels, loss_mask)
        num_tokens = loss_mask.sum()
        
        # Return (loss, num_tokens) for proper averaging
        # Training loop automatically reduces across microbatches and data parallel ranks
        return {
            "loss": torch.cat([loss.view(1), num_tokens.view(1)]),
            "custom_metric": torch.cat([some_metric.view(1), num_tokens.view(1)]),
        }
    
    return output, loss_func

# Pass to training - reduction handled automatically
pretrain(cfg, forward_step_func=custom_forward_step)
```

**Key differences:**
- **NeMo 2.0**: Separate `MegatronLossReduction` class with `forward()` and `reduce()` methods
- **Megatron Bridge**: Loss function returns dict with format `{key: [value, count]}` for automatic reduction
- **Reduction logic**: Megatron Bridge automatically averages `value/count` across microbatches and data parallel ranks

The training loop in Megatron Bridge (see {py:func}`bridge.training.train.train_step`) automatically:
1. Calls the loss function for each microbatch
2. Aggregates results across microbatches
3. Performs data-parallel all-reduce
4. Computes final averaged values

#### When to Customize

Use custom forward steps when you need:
- Custom loss functions beyond standard language modeling
- Multi-task learning with multiple loss components
- Additional metrics computed during training
- Specialized batch preprocessing

For complete documentation on entry point signatures, loss calculation patterns, state access, and more advanced customization options, see {doc}`training/entry-points`.

---

## Callback Migration

Megatron Bridge converts most NeMo 2.0 callbacks into explicit configuration fields or utility functions.

### DDP Parity Checker

Validates that model weights are synchronized across data-parallel replicas.

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import DDPParityChecker

trainer = run.Config(
    nl.Trainer,
    callbacks=[DDPParityChecker(check_interval=100)]
)
```

#### Now: Megatron Bridge
```python
# Built into TrainingConfig
train_config = TrainingConfig(
    check_weight_hash_across_dp_replicas_interval=100,
)
```

### Garbage Collection

Manual garbage collection to free memory during training.

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        GarbageCollectionCallback(
            gc_interval_train=100,
            gc_interval_val=100,
        )
    ]
)
```

#### Now: Megatron Bridge
```python
# Built into TrainingConfig
train_config = TrainingConfig(
    manual_gc=True,              # Enable manual garbage collection
    manual_gc_interval=100,      # GC interval during training (was gc_interval_train)
    manual_gc_eval=True,         # Enable GC at start/end of evaluation (was gc_interval_val)
)
```

### Communication Overlap

Enables overlapping of tensor/pipeline parallel communication with computation.

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import MegatronCommOverlapCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[
        MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            ...
        )
    ]
)
```

#### Now: Megatron Bridge
```python
from megatron.bridge.training.comm_overlap import CommOverlapConfig

config = ConfigContainer(
    comm_overlap=CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=...,  # Detailed TP overlap settings
    ),
)
```

For comprehensive documentation on communication overlap strategies (TP, PP, DP, CP, MoE), hardware requirements, and performance tuning, see {doc}`training/communication-overlap`.

### Preemption Handling

Gracefully handle SLURM/cluster preemption signals.

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import PreemptionCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[PreemptionCallback()]
)
```

#### Now: Megatron Bridge
```python
# Built into TrainingConfig
train_config = TrainingConfig(
    exit_signal_handler=True,  # Enable preemption signal handling
)
```

For more details on preemption handling and fault tolerance, see {doc}`training/resiliency`.

### Experimental Features

Enable Megatron Core experimental features.

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import MegatronEnableExperimentalCallback

trainer = run.Config(
    nl.Trainer,
    callbacks=[MegatronEnableExperimentalCallback()]
)
```

#### Now: Megatron Bridge
```python
from megatron.bridge.training.config import DistributedInitConfig

dist_config = DistributedInitConfig(
    enable_megatron_core_experimental=True,
)
```

### MoE Token Drop

Configures expert capacity and token padding for MoE models.

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import MegatronTokenDropCallback

callbacks = [
    MegatronTokenDropCallback(
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=True
    )
]
```

#### Now: Megatron Bridge
```python
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

model = GPTModelProvider(
    # MoE architecture
    num_moe_experts=8,
    moe_router_topk=2,
    moe_token_dispatcher_type="alltoall",
)

# Apply token drop optimization
apply_moe_token_drop(
    model,
    moe_expert_capacity_factor=1.0,
    moe_pad_expert_input_to_capacity=True
)
```

### DeepEP for MoE

Enables DeepEP optimizations for MoE models on supported hardware (Ampere/Hopper GPUs).

#### Before: NeMo 2.0
```python
from nemo.lightning.pytorch.callbacks import DeepEPCallback

callbacks = [DeepEPCallback()]  # Automatically applies if hardware supports it
```

#### Now: Megatron Bridge
```python
from megatron.bridge.training.deepep import apply_deepep

model = GPTModelProvider(
    num_moe_experts=8,
    # ... other MoE settings
)

# Apply DeepEP optimizations (only on Ampere/Hopper GPUs)
# Hardware validation is performed automatically during training
apply_deepep(model)
```

---

## NeMo-Run, Plugins, and Launching

Megatron Bridge supports both direct Python execution and NeMo-Run orchestration. While NeMo 2.0 relied heavily on NeMo-Run's recipe system, Megatron Bridge provides more flexibility.

For complete details on launching training jobs, configuration overrides, and NeMo-Run integration, see {doc}`recipe-usage`.

### NeMo-Run Integration

#### Direct Python Execution
Megatron Bridge supports standard PyTorch distributed execution patterns:

```bash
# Direct script execution with torchrun
python -m torch.distributed.run --nproc_per_node=8 my_training_script.py

# Multi-node execution
torchrun --nnodes=4 --nproc_per_node=8 \
    --master_addr="node0" --master_port=12345 \
    my_training_script.py
```

#### NeMo-Run with Plugins (Script Mode Recommended)
If using NeMo-Run, **strongly recommend using `run.Script` mode** for better dependency management. Megatron Bridge plugins are designed to work well with this approach:

```python
# Megatron Bridge NeMo-Run integration
import nemo_run as run
from megatron.bridge.recipes.run_plugins import (
    NsysPlugin, WandbPlugin, PreemptionPlugin, FaultTolerancePlugin
)

# Create task 
task = run.Script("my_training_script.py", args=[])

# Configure executor with plugins
executor = run.SlurmExecutor(nodes=2, nproc_per_node=8)
executor.plugins = [
    NsysPlugin(profile_step_start=100, profile_step_end=110),
    WandbPlugin(project="my_project", entity="my_team"),
    PreemptionPlugin(preempt_time=120),
]

# Submit job
run.run(task, executor=executor)
```

### Plugin Migration Comparison

| **Plugin** | **NeMo 2.0** | **Megatron Bridge** |
|------------|-------------|-----------|
| **Nsys** | `nemo.lightning.run.plugins.NsysPlugin` | {py:class}`bridge.recipes.run_plugins.NsysPlugin` |
| **Wandb** | `nemo.lightning.run.plugins.WandbPlugin` | {py:class}`bridge.recipes.run_plugins.WandbPlugin` |
| **Preemption** | `nemo.lightning.run.plugins.PreemptionPlugin` | {py:class}`bridge.recipes.run_plugins.PreemptionPlugin` |
| **Fault Tolerance** | `nemo.lightning.run.plugins.FaultTolerancePlugin` | {py:class}`bridge.recipes.run_plugins.FaultTolerancePlugin` |
