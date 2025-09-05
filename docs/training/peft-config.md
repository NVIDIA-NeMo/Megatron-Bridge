# Parameter-Efficient Fine-Tuning (PEFT)

Customizing models enables you to adapt a general pre-trained models to a specific use case or domain. This process results in a fine-tuned model that benefits from the extensive pretraining data, while also yielding more accurate outputs for the specific downstream task. Model customization is achieved through supervised fine-tuning and falls into two popular categories:

- Full-Parameter Fine-Tuning, which is referred to as Supervised Fine-Tuning (SFT)

- Parameter-Efficient Fine-Tuning (PEFT)

In SFT, all of the model parameters are updated to produce outputs that are adapted to the task.

PEFT, on the other hand, tunes a much smaller number of parameters which are inserted into the base model at strategic locations. When fine-tuning with PEFT, the base model weights remain frozen, and only the adapter modules are trained. As a result, the number of trainable parameters is significantly reduced, often to less than 1%.

While SFT often yields the best possible results, PEFT methods can often achieve nearly the same degree of accuracy, while significantly reducing the computational cost. As language models continue to grow in size, PEFT is gaining popularity due to its lightweight requirements on training hardware.


## Configuration

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

LoRA makes fine-tuning efficient by representing weight updates with two low rank decomposition matrices. The original model weights remain frozen, while the low-rank decomposition matrices are updated to adapt to the new data, keeping the number of trainable parameters low. In contrast with adapters, the original model weights and adapted weights can be combined during inference, avoiding any architectural change or additional latency in the model at inference time.

In Megatron-Bridge, you can customize the adapter bottleneck dimension and the target modules to apply LoRA. LoRA can be applied to any linear layer. In a transformer model, this includes 1) Q, K, V attention projections, 2) attention output projection layer, and 3) either or both of the two transformer MLP layers. For QKV, Megatron-Bridge's attention implementation fuses QKV into a single projection, so our LoRA implementation learns a single low-rank projection for QKV combined.

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

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_modules` | `List[str]` | All linear layers | Modules to apply LoRA to |
| `dim` | `int` | `32` | Rank of the low-rank adaptation |
| `alpha` | `float` | `16` | Scaling parameter for LoRA |
| `dropout` | `float` | `0.0` | Dropout rate for LoRA layers |
| `network_alpha` | `Optional[float]` | `None` | Network-wide alpha scaling |

**Target Modules:**
- `linear_qkv`: Query, key, value projections in attention
- `linear_proj`: Attention output projection  
- `linear_fc1`: First MLP layer
- `linear_fc2`: Second MLP layer

### Wildcard Target Modules
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

DoRA decomposes the pre-trained weight into magnitude and direction. It learns a separate magnitude parameter while employing LoRA for directional updates, efficiently minimizing the number of trainable parameters. DoRA enhances both the learning capacity and training stability of LoRA while avoiding any additional inference overhead. DoRA has been shown to consistently outperform LoRA on various downstream tasks.

In Megatron-Bridge, DoRA leverages the same adapter structure as LoRA. Megatron-Bridge adds support for Tensor Parallelism and Pipeline Parallelism for DoRA, enabling DoRA to be scaled to larger model variants.

```python
from megatron.bridge.peft.dora import DoRA

dora_config = DoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=16,                    # Rank of adaptation
    alpha=32,                  # Scaling parameter
    dropout=0.1,               # Dropout rate
)
```

**Key Parameters:**

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

## PEFT Design in Megatron-Bridge

This section describes the internal design and architecture for how PEFT is integrated into Megatron-Bridge.

### Architecture Overview

1. **Base PEFT Class**: All PEFT methods inherit from the abstract `PEFT` base class, which defines the core transformation interface.
2. **Module Transformation**: PEFT methods transform individual modules by walking through the model structure.
3. **Adapter Integration**: Adapters are injected into target modules using a pre-wrap hook during model initialization.
4. **Checkpoint Integration**: Only adapter parameters are saved/loaded, while base model weights remain frozen.

### PEFT Workflow in Training

1. **Model Loading**: The Base model is loaded from the specified pretrained checkpoint during setup.
2. **PEFT Application**: The PEFT transformation is applied after the Megatron Core model initialization but before distributed wrapping.
3. **Parameter Freezing**: Base model parameters are frozen, only adapter parameters remain trainable.
4. **Adapter Weight Loading**: If resuming training, adapter weights are restored from the checkpoint.
5. **Checkpoint Saving**: Only adapter states are saved, significantly reducing checkpoint size.

### Key Benefits

- **Reduced Checkpoint Size**: Checkpoints are orders of magnitude smaller than full model checkpoints.
- **Memory Efficiency**: Base model weights don't require gradients, only adapters do.
- **Resume Support**: Can resume PEFT training from adapter-only checkpoints.