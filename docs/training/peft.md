# Parameter-Efficient Fine-Tuning (PEFT)

This guide explains how to configure and use PEFT in Megatron Bridge—covering LoRA and DoRA, required checkpoints, example configurations, and the internal design and training workflow—so you can integrate, scale, and checkpoint adapters efficiently.

## Model Customization
Customizing models enables you to adapt a general pre-trained model to a specific use case or domain. This process produces a fine-tuned model that retains the broad knowledge from pretraining while delivering more accurate outputs for targeted downstream tasks.

Model customization is typically achieved through supervised fine-tuning, which falls into two main approaches: Full-Parameter Fine-Tuning, known as Supervised Fine-Tuning (SFT), and Parameter-Efficient Fine-Tuning (PEFT).

In SFT, all model parameters are updated to align the model’s outputs with the task-specific requirements. This approach often yields the highest performance but can be computationally intensive.

PEFT, by contrast, updates only a small subset of parameters that are inserted into the base model at strategic locations. The base model weights remain frozen, and only the adapter modules are trained. This significantly reduces the number of trainable parameters—often to less than 1%—while still achieving near-SFT levels of accuracy.

As language models continue to grow in size, PEFT is gaining popularity for its efficiency and minimal hardware demands, making it a practical choice for many real-world applications.

## PEFT Configuration

PEFT is configured as an optional attribute in `ConfigContainer`:

```python
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.peft.lora import LoRA

config = ConfigContainer(
    # ... other required configurations
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=16,
        alpha=32,
        dropout=0.1,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/pretrained/checkpoint",  # Required for PEFT
        save="/path/to/peft/checkpoints",
    ),
)
```

```{note}
**Requirements**: PEFT requires `checkpoint.pretrained_checkpoint` to be set to load the base model weights.
```

## Supported PEFT Methods

### [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

LoRA makes fine-tuning efficient by representing weight updates with two low-rank decomposition matrices. The original model weights remain frozen, while the low-rank decomposition matrices are updated to adapt to the new data, keeping the number of trainable parameters low. In contrast with adapters, the original model weights and adapted weights can be combined during inference, avoiding any architectural change or additional latency in the model at inference time.

In Megatron Bridge, you can configure both the adapter bottleneck dimension and the target modules where LoRA is applied. LoRA supports any linear layer, which in transformer models typically includes:

1. Query, key, and value (QKV) attention projections  
2. The attention output projection  
3. One or both MLP layers  

Megatron Bridge fuses the QKV projections into a single linear layer. As a result, LoRA learns a unified low-rank adaptation for the combined QKV representation.

```python
from megatron.bridge.peft.lora import LoRA

lora_config = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=16,                    # Rank of adaptation
    alpha=32,                  # Scaling parameter  
    dropout=0.1,               # Dropout rate
    network_alpha=None,        # Network alpha for scaling
)
```

#### Key Parameters
The following table lists key hyperparameters for configuring DoRA, which control its module targeting, adaptation rank, scaling behavior, and regularization strategy.
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_modules` | `List[str]` | All linear layers | Modules to apply DoRA to |
| `dim` | `int` | `32` | Rank of the low-rank adaptation |
| `alpha` | `float` | `16` | Scaling parameter for DoRA |
| `dropout` | `float` | `0.0` | Dropout rate for DoRA layers |

#### Target Modules
The following table lists specific submodules within transformer architectures that are commonly targeted for LoRA, enabling efficient fine-tuning of attention and feedforward components:
| Module        | Description                                 |
|---------------|---------------------------------------------|
| `linear_qkv`  | Query, key, value projections in attention  |
| `linear_proj` | Attention output projection                 |
| `linear_fc1`  | First MLP layer                             |
| `linear_fc2`  | Second MLP layer                            |

#### Wildcard Target Modules
For more granular targeting, individual layers can be targeted for the adapters.
```python
# Target specific layers only
lora_config = LoRA(
    target_modules=[
        "*.layers.0.*.linear_qkv",   # First layer only
        "*.layers.1.*.linear_qkv",   # Second layer only
    ]
)
```

### [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

DoRA decomposes the pre-trained weight into magnitude and direction. It learns a separate magnitude parameter while employing LoRA for directional updates, efficiently minimizing the number of trainable parameters. DoRA enhances both the learning capacity and training stability of LoRA, while avoiding any additional inference overhead. DoRA has been shown to consistently outperform LoRA on various downstream tasks.

In Megatron Bridge, DoRA leverages the same adapter structure as LoRA. Megatron Bridge adds support for Tensor Parallelism and Pipeline Parallelism for DoRA, enabling DoRA to be scaled to larger model variants.

```python
from megatron.bridge.peft.dora import DoRA

dora_config = DoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=16,                    # Rank of adaptation
    alpha=32,                  # Scaling parameter
    dropout=0.1,               # Dropout rate
)
```

#### Key Parameters

The following parameters define how LoRA is applied to your model. They control which modules are targeted, the adaptation rank, scaling behavior, and dropout configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_modules` | `List[str]` | All linear layers | Modules to apply DoRA to |
| `dim` | `int` | `32` | Rank of the low-rank adaptation |
| `alpha` | `float` | `16` | Scaling parameter for DoRA |
| `dropout` | `float` | `0.0` | Dropout rate for DoRA layers |

## Full Configuration Example

```python
from megatron.bridge.training.config import (
    ConfigContainer, TrainingConfig, CheckpointConfig
)
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.peft.lora import LoRA
from megatron.core.optimizer import OptimizerConfig

# Configure PEFT fine-tuning
config = ConfigContainer(
    model=model_provider,
    train=TrainingConfig(
        train_iters=1000,
        global_batch_size=64,
        micro_batch_size=1,  # Required for packed sequences if used
        eval_interval=100,
    ),
    optimizer=OptimizerConfig(
        optimizer="adam",
        lr=1e-4,  # Lower learning rate for fine-tuning
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=True,
    ),
    scheduler=SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=100,
        lr_decay_iters=1000,
    ),
    dataset=HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=512,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/pretrained/model",  # Required
        save="/path/to/peft/checkpoints",
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

## PEFT Design in Megatron Bridge

This section describes the internal design and architecture for how PEFT is integrated into Megatron Bridge.

### Architecture Overview

The PEFT framework introduces a modular design for integrating adapters into large-scale models. Its architecture consists of the following components:

1. **Base PEFT Class**: All PEFT methods inherit from the abstract {py:class}`bridge.peft.base.PEFT` base class, which defines the core interface for module transformation.
2. **Module Transformation**: PEFT traverses the model structure to identify and transform target modules individually.
3. **Adapter Integration**: Adapters are injected into selected modules using a pre-wrap hook during model initialization.
4. **Checkpoint Integration**: Only adapter parameters are saved and loaded during checkpointing; base model weights remain frozen and unchanged.

### PEFT Workflow in Training

The training workflow for PEFT follows a structured sequence that ensures efficient fine-tuning with minimal overhead:
1. **Model Loading**: The base model is initialized from a specified pretrained checkpoint.
2. **PEFT Application**: Adapter transformations are applied after Megatron Core model initialization, but before distributed wrapping.
3. **Parameter Freezing**: Base model parameters are frozen to reduce training complexity; only adapter parameters are updated.
4. **Adapter Weight Loading**: When resuming training, adapter weights are restored from the checkpoint.
5. **Checkpoint Saving**: Only adapter states are saved, resulting in significantly smaller checkpoint files.

### Key Benefits

PEFT offers several advantages for scalable and efficient model fine-tuning:

- **Reduced Checkpoint Size**: Adapter-only checkpoints are dramatically smaller than full model checkpoints.
- **Memory Efficiency**: Since gradients are computed only for adapter parameters, memory usage is significantly reduced.
- **Resume Support**: Training can be resumed seamlessly using adapter-only checkpoints, without reloading full model weights.