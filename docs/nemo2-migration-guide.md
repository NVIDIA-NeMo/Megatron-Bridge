# NeMo 2.0 → Megatron Bridge Migration Guide

This guide helps you migrate from NeMo 2.0 training and recipes to Megatron Bridge. Megatron Bridge retains the Pythonic, code-first API that NeMo 2.0 developed while simplifying configuration into a single {py:class}`bridge.training.config.ConfigContainer` with typed sub-configs. Model parallelism and performance features from Megatron-Core remain first-class.

## Key Concepts

## What Stays the Same

- **Megatron-Core Foundation**: Megatron Bridge uses the same Megatron-Core engine under the hood.
- **Model Parallelism**: Same TP/PP/CP/EP concepts with identical distributed training semantics.
- **High-Performance Features**: Mixed Precision, communication overlap, and other performance features are supported natively.
- **Pythonic API Retained**: Megatron Bridge preserves NeMo 2.0's philosophy of "configuration as code."


## Model Configuration Mapping
The model configuration largely maps from NeMo 2.0 to Megatron Bridge providers:

| NeMo 2.0 | Megatron Bridge |
|----------|-----------------|
| `GPTConfig` | {py:class}`bridge.models.GPTModelProvider` |
| `T5Config` | {py:class}`bridge.models.T5ModelProvider` |
| `SSMConfig` | {py:class}`bridge.models.MambaProvider` |

Most model fields map 1:1 between NeMo's model configs and Megatron Bridge's model providers.

## Training Configuration
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
- {doc}`training/training-config` - Training loop parameters and validation
- {doc}`training/checkpointing-config` - Checkpointing and model persistence
- {doc}`training/optimizer-scheduler-config` - Optimization and learning rate scheduling
- {doc}`training/logging` - Logging, TensorBoard, and Weights & Biases
- {doc}`training/profiling` - Performance profiling with Nsys and PyTorch

---

## Core Configuration Migration

This section covers the essential configuration mappings from NeMo 2.0 to Megatron Bridge. These form the foundation that all training modes (pretraining, SFT, PEFT) build upon. **Start here to understand the fundamental differences**, then refer to the training mode examples and callback migration sections that follow.

### Before (NeMo 2.0)
```python
# NeMo 2.0 Python API
from nemo import lightning as nl
from nemo.collections import llm

# Basic trainer setup
trainer = nl.Trainer(
    max_steps=1000,
    val_check_interval=100,
    limit_val_batches=50,
    log_every_n_steps=10,
)

model = llm.LlamaModel(
    config=llm.LlamaConfig(
        seq_length=2048,
        num_layers=24,
        hidden_size=1024,
    )
)

# For fine-tuning
data = llm.FineTuningDataModule(
    dataset_root="/path/to/data", 
    seq_length=2048, 
    micro_batch_size=1, 
    global_batch_size=128
)

trainer.fit(model, data)
```

### After (Megatron Bridge)
```python
# Megatron Bridge configuration pattern
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.models import GPTModelProvider

def create_config():
    return ConfigContainer(
        # Model with parallelism built-in
        model=GPTModelProvider(
            seq_length=2048,
            num_layers=24,
            hidden_size=1024,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            # ... other model params
        ),
        # Training loop configuration
        train=TrainingConfig(
            global_batch_size=512,
            micro_batch_size=1,
            train_iters=1000,
            eval_interval=100,
            eval_iters=10,
        ),
        # Optimization and scheduling
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=3e-4,
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
            sequence_length=2048,
            # ... other dataset params
        ),
        # Checkpointing and logging  
        checkpoint=CheckpointConfig(
            save="/path/to/checkpoints",
            save_interval=100,
            ckpt_format="torch_dist",
        ),
        logger=LoggerConfig(log_interval=10),
    )
```

## Basic Configuration Example

The following shows the basic pattern for moving from NeMo 2.0 recipes to Megatron Bridge configurations. For detailed configuration mappings, see the sections below. For specific training modes (pretraining, SFT, PEFT), see the [Training Modes section](#training-modes-pretraining-and-finetuning).

### SFT (Supervised Fine-Tuning)
For SFT, use {py:class}`bridge.training.config.FinetuningDatasetConfig` which provides specialized data handling including sequence length management, dataset root specification, packed sequence support, and validation/test dataset configuration.

### PEFT (Parameter-Efficient Fine-Tuning)
Enable PEFT by supplying a `peft` configuration block and setting `checkpoint.pretrained_checkpoint`. Megatron Bridge enforces that a pretrained checkpoint is provided when PEFT is enabled.

### Before (NeMo 2.0)
```python
# NeMo 2.0 PEFT with Python API
from nemo import lightning as nl
from nemo.collections import llm
import nemo_run as run

# For PEFT, create LoRA instance
lora = llm.peft.LoRA(
    target_modules=['linear_qkv', 'linear_proj'],
    dim=32,
)

trainer = nl.Trainer(
    max_steps=500, 
    val_check_interval=100,
    callbacks=[lora]  # PEFT callback
)

model = llm.LlamaModel(
    config=llm.LlamaConfig(...),
    model_transform=lora  # PEFT transform
)

data = llm.SquadDataModule(seq_length=2048, micro_batch_size=1, global_batch_size=128)

trainer.fit(model, data)

# Or with NeMo-Run finetune API
sft = llm.finetune(
    model=llm.llama3_8b,
    data=llm.squad,
    trainer=trainer,
    peft=llm.peft.LoRA(target_modules=['linear_qkv', 'linear_proj'], dim=32),
)
sft.resume.import_path = "hf://meta-llama/Meta-Llama-3-8B"
sft.resume.adapter_path = "/path/to/checkpoints/last"
```

### After (Megatron Bridge)  
```python  
# Megatron Bridge SFT configuration
def create_sft_config():
    return ConfigContainer(
        model=LlamaProvider(
            # ... model architecture
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
        peft=LoraConfig(
            target_modules=["qkv_proj", "o_proj"],
            dim=64,
        ),
        # ... other configs
    )
```


## Configuration Migration Details

### Training Configuration Migration
Lightning `Trainer` parameters are now managed through dedicated configuration classes.

| **Setting Category** | **NeMo 2.0 Location** | **Megatron Bridge Location** | **Details** |
|---------------------|----------------------|-------------------|-------------|
| **Training iterations** | `trainer.max_steps` | {py:attr}`bridge.training.config.TrainingConfig.train_iters` | Total training iterations |
| **Validation frequency** | `trainer.val_check_interval` | {py:attr}`bridge.training.config.TrainingConfig.eval_interval` | Steps between validation runs |
| **Validation iterations** | `trainer.limit_val_batches` | {py:attr}`bridge.training.config.TrainingConfig.eval_iters` | Number of validation steps per run |
| **Test iterations** | `trainer.limit_test_batches` | {py:attr}`bridge.training.config.TrainingConfig.eval_iters` | Number of test steps (shares eval_iters) |
| **Logging frequency** | `trainer.log_every_n_steps` | {py:attr}`bridge.training.config.LoggerConfig.log_interval` | Logging frequency |

#### Before (NeMo 2.0)
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

#### After (Megatron Bridge)
```python  
train_config = TrainingConfig(
    train_iters=1000,           # was max_steps
    eval_interval=100,          # was val_check_interval  
    eval_iters=50,              # was limit_val_batches (for both val and test)
)
logger_config = LoggerConfig(log_interval=10)  # was log_every_n_steps
```

### Parallelism Configuration Migration
In NeMo 2.0, parallelism settings were configured on `MegatronStrategy`. In Megatron Bridge, many of these are set directly on the model provider:

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

#### Before (NeMo 2.0)
```python
strategy = run.Config(
    MegatronStrategy,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    sequence_parallel=True,
)
```

#### After (Megatron Bridge)
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

#### Before (NeMo 2.0)
```python
# Mixed precision via plugin
from nemo.lightning.pytorch.plugins import MegatronMixedPrecisionPlugin

trainer = run.Config(
    nl.Trainer,
    plugins=[MegatronMixedPrecisionPlugin(precision="bf16-mixed")]
)
```

#### After (Megatron Bridge)
```python
# Option 1: Use preset strings
config = ConfigContainer(
    mixed_precision="bf16-mixed",  # Simple preset
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

### Checkpointing Configuration Migration
Checkpointing configuration moves from `MegatronStrategy` parameters and `ModelCheckpoint` callback to {py:class}`bridge.training.config.CheckpointConfig`:

| **Checkpoint Setting** | **NeMo 2.0** | **Megatron Bridge** |
|------------------------|-------------|-----------|
| **Save directory** | `ModelCheckpoint(dirpath=...)` | {py:attr}`bridge.training.config.CheckpointConfig.save` |
| **Load directory** | `trainer.ckpt_path` | {py:attr}`bridge.training.config.CheckpointConfig.load` |
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

#### Before (NeMo 2.0)
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

#### After (Megatron Bridge)
```python
checkpoint_config = CheckpointConfig(
    save="/path/to/checkpoints",
    save_interval=1000,
    most_recent_k=3,        # Keeps 3 most recent checkpoints (not metric-based)
    ckpt_format="torch_dist",
    async_save=True,
    fully_parallel_save=True,
    fully_parallel_load=True,
    load_optim=True,
    save_optim=True,
    save_rng=True,
    dist_ckpt_strictness="assume_ok_unexpected",
)
```

**Note**: Megatron Bridge writes the same Megatron distributed checkpoint format as NeMo 2. If you point `save` to a directory containing a `weights/` subdirectory, the format is fully compatible with NeMo 2's Megatron-Core artifacts.

---

## Optimizer and LR Scheduler Migration

Optimization configuration moves from NeMo 2.0's `MegatronOptimizerModule` approach to Megatron Bridge's direct {py:class}`megatron.core.optimizer.OptimizerConfig` and {py:class}`bridge.training.config.SchedulerConfig`.

### Before (NeMo 2.0)
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

### After (Megatron Bridge)
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

**Key advantages**: Megatron Bridge automatically computes derived scheduler steps during `cfg.validate()` based on `train.train_iters` and `train.global_batch_size`, reducing configuration errors.

## Tokenizer Configuration Migration

Megatron Bridge uses {py:class}`bridge.training.tokenizers.config.TokenizerConfig` for consistent tokenizer setup across different model types.

### Before (NeMo 2.0)  
```python
# Tokenizer often configured within model config
model_config = GPTConfig(
    tokenizer=dict(
        vocab_file="/path/to/vocab.json",
        merge_file="/path/to/merges.txt",
        tokenizer_type="GPT2BPETokenizer",
    )
)
```

### After (Megatron Bridge)
```python
# Dedicated tokenizer configuration
tokenizer_config = TokenizerConfig(
    tokenizer_type="GPT2BPETokenizer",
    vocab_file="/path/to/vocab.json", 
    merge_file="/path/to/merges.txt",
)
```

## Profiling Migration

Megatron Bridge centralizes all profiling functionality in {py:class}`bridge.training.config.ProfilingConfig`, replacing multiple NeMo callbacks.

### Nsys Profiling Migration

#### Before (NeMo 2.0)
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

#### After (Megatron Bridge)
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

### PyTorch Profiler Migration

#### Before (NeMo 2.0)
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

#### After (Megatron Bridge)
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

## Training Modes: Pretraining and Finetuning

After understanding the core configuration mappings above, you can apply them to different training modes. The same configuration patterns work across pretraining, SFT, and PEFT with small variations in data handling and checkpoint loading.

### Pretraining 
For pretraining from scratch, use {py:class}`bridge.training.config.GPTDatasetConfig` with raw text data blends:

```python
# Complete pretraining configuration
config = ConfigContainer(
    model=GPTModelProvider(...),
    train=TrainingConfig(train_iters=100000, eval_interval=1000),
    dataset=GPTDatasetConfig(
        blend=["/path/to/train_data_text_document"],
        sequence_length=4096,
        split="949,50,1",  # train/val/test split
    ),
    # ... other core configs from sections above
)
```

### Supervised Fine-Tuning (SFT)
For SFT, use {py:class}`bridge.training.config.FinetuningDatasetConfig` and specify a pretrained checkpoint. **Important**: The pretrained checkpoint must be in Megatron checkpoint format.

```python
# SFT configuration  
config = ConfigContainer(
    model=GPTModelProvider(...),
    train=TrainingConfig(train_iters=1000, eval_interval=100),
    dataset=FinetuningDatasetConfig(
        dataset_root="/path/to/instruction_data",
        seq_length=4096,
        do_validation=True,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",  # Must be Megatron format
        save="/path/to/sft_checkpoints",
    ),
    # ... other core configs
)
```

**Converting HuggingFace models**: If you have a HuggingFace model, use {py:meth}`bridge.models.conversion.auto_bridge.AutoBridge.import_ckpt` to convert it first:

```python
# Convert HuggingFace model to Megatron format before finetuning
from megatron.bridge.models.conversion.auto_bridge import AutoBridge

# Import HuggingFace model and save as Megatron checkpoint
AutoBridge.import_ckpt(
    "meta-llama/Meta-Llama-3-8B",           # HuggingFace model ID
    "/path/to/megatron/checkpoint"          # Output Megatron checkpoint path
)

# Now use this checkpoint for finetuning
config = ConfigContainer(
    # ... model and training configs
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/megatron/checkpoint",
        save="/path/to/sft_checkpoints",
    ),
)
```

For more details on model conversion and the AutoBridge API, see {doc}`bridge-guide`.

### Parameter-Efficient Fine-Tuning (PEFT)
Enable PEFT by adding a `peft` configuration alongside the pretrained checkpoint:

```python
# PEFT configuration
config = ConfigContainer(
    # ... same model, train, dataset as SFT above
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/pretrained/model",
        save="/path/to/peft_checkpoints",
    ),
    peft=LoraConfig(
        target_modules=["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        dim=64,
        alpha=16,
    ),
    # ... other core configs  
)
```

**Key Pattern**: All training modes use the same core configuration structure. The differences are:
- **Data**: `GPTDatasetConfig` for pretraining, `FinetuningDatasetConfig` for SFT/PEFT
- **Checkpoints**: No pretrained checkpoint for pretraining, required for SFT/PEFT
- **PEFT**: Optional `peft` config for parameter-efficient training

For more details on each training mode, see:
- {doc}`training/entry-points` - Training entry points and customization
- {doc}`training/peft-config` - PEFT configuration and supported methods

---

## Callback and Utility Mappings

Megatron Bridge converts most NeMo 2.0 callbacks into explicit configuration fields or utility functions, providing better type safety and easier configuration management:

| **Feature** | **NeMo 2.0** | **Megatron Bridge** | **Migration Notes** |
|-------------|-------------|-----------|-------------------|
| **DDP Parity Check** | `DDPParityChecker` callback | {py:attr}`bridge.training.config.TrainingConfig.check_weight_hash_across_dp_replicas_interval` | Now a simple config field |
| **DeepEP Enablement** | `DeepEPCallback` | {py:func}`bridge.training.deepep.validate_deepep` + {py:func}`bridge.training.deepep.apply_deepep` | Runtime validation and utility functions |
| **MoE Token Drop** | `MegatronTokenDropCallback` | {py:func}`bridge.training.utils.moe_token_drop.apply_moe_token_drop` | Utility function + model config |
| **Manual GC** | `GarbageCollectionCallback` | {py:attr}`bridge.training.config.TrainingConfig.manual_gc` | Built into training config |
| **Comm Overlap** | `MegatronCommOverlapCallback` | {py:class}`bridge.training.comm_overlap.CommOverlapConfig` | Comprehensive config object |
| **Experimental Features** | `MegatronEnableExperimentalCallback` | {py:attr}`bridge.training.config.DistributedInitConfig.enable_megatron_core_experimental` | Simple boolean flag |
| **Memory Profiling** | `MemoryProfileCallback` | {py:attr}`bridge.training.config.ProfilingConfig.record_memory_history` | Unified in profiling config |
| **Preemption** | `PreemptionCallback` | {py:attr}`bridge.training.config.TrainingConfig.exit_signal_handler` | Direct config setting |

### MoE Configuration Migration
MoE-related settings are consolidated on the model provider with utility functions for specific optimizations:

#### Before (NeMo 2.0)
```python
callbacks = [
    MegatronTokenDropCallback(
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=True
    )
]
```

#### After (Megatron Bridge)
```python
# Apply utility function to model
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

model = GPTModelProvider(
    # MoE architecture
    num_moe_experts=8,
    moe_router_topk=2,
    moe_token_dispatcher_type="alltoall",
    # ... other MoE settings
)

# Apply token drop optimization
apply_moe_token_drop(
    model,
    moe_expert_capacity_factor=1.0,
    moe_pad_expert_input_to_capacity=True
)
```

---

## NeMo-Run, Plugins, and Launching

### NeMo-Run Integration
Megatron Bridge supports both direct Python execution and NeMo-Run orchestration. While NeMo 2.0 relied heavily on NeMo-Run's recipe system, Megatron Bridge provides more flexibility.

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

| **Plugin** | **NeMo 2.0** | **Megatron Bridge** | **Key Differences** |
|------------|-------------|-----------|-------------------|
| **Nsys** | `nemo.lightning.run.plugins.NsysPlugin` | {py:class}`bridge.recipes.run_plugins.NsysPlugin` | Optimized for `run.Script` with automatic CLI overrides |
| **Wandb** | `nemo.lightning.run.plugins.WandbPlugin` | {py:class}`bridge.recipes.run_plugins.WandbPlugin` | Simpler configuration, automatic CLI override generation |
| **Preemption** | `nemo.lightning.run.plugins.PreemptionPlugin` | {py:class}`bridge.recipes.run_plugins.PreemptionPlugin` | Direct config setting, better dependency isolation |
| **Fault Tolerance** | `nemo.lightning.run.plugins.FaultTolerancePlugin` | {py:class}`bridge.recipes.run_plugins.FaultTolerancePlugin` | Config-based approach with script mode benefits |

**Recommendation**: Use `run.Script` mode with Megatron Bridge for better dependency management, environment isolation, and cleaner configuration override handling.

## Quick Migration Checklist

1. **Configuration Structure**: Move from NeMo-Run recipes to Megatron Bridge `ConfigContainer` with typed sub-configs
2. **Model Setup**: Convert `GPTConfig`/`T5Config` to corresponding Megatron Bridge providers  
3. **Parallelism**: Move parallelism settings from `MegatronStrategy` to model provider
4. **Training Settings**: Map `Trainer` options to {py:class}`bridge.training.config.TrainingConfig`
5. **Infrastructure**: Move strategy settings to {py:class}`bridge.training.config.DistributedInitConfig`
6. **Optimization**: Port optimizer to {py:class}`megatron.core.optimizer.OptimizerConfig` and scheduler to {py:class}`bridge.training.config.SchedulerConfig`
7. **Observability**: Add {py:class}`bridge.training.tokenizers.config.TokenizerConfig` and {py:class}`bridge.training.config.ProfilingConfig` as needed
8. **Fine-tuning**: For SFT/PEFT, use {py:class}`bridge.training.config.FinetuningDatasetConfig` and set `checkpoint.pretrained_checkpoint`

