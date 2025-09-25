# Mixed Precision Training

Mixed precision training significantly enhances computational efficiency by conducting operations in low-precision format, while selectively maintaining minimal data in single-precision to preserve critical information throughout key areas of the network. Megatron Bridge supports FP16, BF16, and FP8 via Transformer Engine (TE) across most models through the {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` configuration.

## Configuration Overview

Mixed precision is configured in Megatron Bridge through the `mixed_precision` field in {py:class}`bridge.training.config.ConfigContainer`, which accepts either:
- A string name referencing a predefined recipe (e.g., `"bf16_mixed"`)  
- A {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` object for custom configurations

The mixed precision configuration automatically updates the model, optimizer, and distributed data parallel settings with the appropriate precision parameters.

## Half-Precision Training

Megatron Bridge supports half-precision FP16 and BF16 computation training via Megatron Core and the distributed optimizer. This training recipe uses half-precision in all layer computation while keeping the model states (optimizer states and master parameters) in single-precision. To avoid repeated data type casting at each layer computation, Megatron Core keeps a separate copy of half-precision parameters that is updated after each optimizer step.

### Using Predefined Recipes

The simplest way to enable mixed precision is using predefined recipe names:

```python
from megatron.bridge.training.config import ConfigContainer

# Configure with BF16 mixed precision
config = ConfigContainer(
    mixed_precision="bf16_mixed",
    # ... other config parameters
)

# Configure with FP16 mixed precision  
config = ConfigContainer(
    mixed_precision="fp16_mixed",
    # ... other config parameters
)
```

### Custom Mixed Precision Configuration

For more control, create a custom {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig`:

```python
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
import torch

# Custom BF16 configuration
bf16_config = MixedPrecisionConfig(
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    autocast_enabled=False,
    grad_reduce_in_fp32=True,
)

config = ConfigContainer(
    mixed_precision=bf16_config,
    # ... other config parameters
)
```

## FP8 Training

NVIDIA H100 GPU introduced support for a new datatype, FP8 (8-bit floating point), enabling higher throughput of matrix multiplies and convolutions. Megatron Bridge uses the NVIDIA TransformerEngine (TE) to leverage speedups from FP8. For a more detailed overview, refer to the [TE documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html), specifically the FP8 format and recipe.

### FP8 Configuration Parameters

The {py:class}`bridge.training.mixed_precision.MixedPrecisionConfig` provides several FP8-specific parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fp8` | `Optional[str]` | `None` | FP8 format: `"hybrid"` (E4M3 for activations/weights, E5M2 for gradients) or `"e4m3"` |
| `fp8_recipe` | `str` | `"tensorwise"` | FP8 recipe type: `"tensorwise"`, `"delayed"`, `"blockwise"`, `"mxfp8"` (Blackwell only) |
| `first_last_layers_bf16` | `bool` | `False` | If True, retains first and last N TransformerBlocks in BF16 as opposed to FP8 |
| `num_layers_at_start_in_bf16` | `int` | `0` | Number of layers at the start of the model to keep in BF16 precision when `first_last_layers_bf16` is True |
| `num_layers_at_end_in_bf16` | `int` | `0` | Number of layers at the end of the model to keep in BF16 precision when `first_last_layers_bf16` is True |
| `fp8_margin` | `int` | `0` | Scaling factor shift by $2^{margin}$ |
| `fp8_amax_history_len` | `int` | `1` | Window size for amax history storage |
| `fp8_amax_compute_algo` | `str` | `"most_recent"` | Amax selection algorithm: `"max"` or `"most_recent"` |
| `fp8_param` | `Optional[bool]` | `None` | Store module-level parameters in FP8 |
| `fp8_param_gather` | `bool` | `False` | Enable FP8 parameter gathering |

### FP8 Recipe Examples

Use any of the predefined FP8 recipe names with the `mixed_precision` parameter:

```python
# Example: BF16 with FP8 current scaling
config = ConfigContainer(
    mixed_precision="bf16_with_fp8_current_scaling_mixed",
    # ... other config parameters
)
```

## Available Mixed Precision Recipes

Megatron Bridge provides numerous predefined mixed precision recipes for different use cases. You can use the {py:func}`~megatron.bridge.training.mixed_precision.get_mixed_precision_config` utility function to convert from a string shortname to a class instance. For the complete list of available recipes and their specific configurations, see the {py:mod}`megatron.bridge.training.mixed_precision` module.


### Custom FP8 Configuration

For advanced use cases, create a custom FP8 configuration:

```python
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
import torch

# Custom FP8 configuration
fp8_config = MixedPrecisionConfig(
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    fp8="hybrid",
    fp8_recipe="tensorwise", 
    fp8_margin=0,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo="max",
    fp8_param_gather=True,
)

config = ConfigContainer(
    mixed_precision=fp8_config,
    # ... other config parameters
)
```

### Registering Custom Mixed Precision Recipes

You can also register your own custom mixed precision configurations to work with the shortname system. Use the {py:func}`~megatron.bridge.training.mixed_precision.register` decorator on a function that returns a `MixedPrecisionConfig` object:

```python
from megatron.bridge.training.mixed_precision import register, MixedPrecisionConfig

@register
def my_custom_fp8_recipe() -> MixedPrecisionConfig:
    """Custom FP8 recipe with specific settings for my use case."""
    return MixedPrecisionConfig(
        bf16=True,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        fp8_param_gather=True,
        # ... other custom settings
    )

# Now you can use it with the utility function
config = get_mixed_precision_config("my_custom_fp8_recipe")
```

Common recipe categories include:
- **Half-precision recipes**: Basic BF16 and FP16 mixed precision
- **FP8 recipes**: Various FP8 scaling strategies (delayed, current, subchannel)
- **Architecture-specific recipes**: Optimized for specific GPU architectures (Hopper, Blackwell)
- **Model-specific recipes**: Tuned for particular model families

## Configuration Synchronization

When a mixed precision configuration is provided, it automatically synchronizes precision-related settings across the model, optimizer, and distributed data parallel (DDP) configurations. This ensures consistent precision behavior throughout the training pipeline.

**Important**: Mixed precision settings will override any conflicting precision parameters that may have been set directly on the model, optimizer, or DDP configurations. The mixed precision configuration acts as the authoritative source for all precision-related parameters.

For example, if you specify both:
```python
# This will be overridden
model_config.bf16 = False
optimizer_config.bf16 = False

config = ConfigContainer(
    model=model_config,
    optimizer=optimizer_config,
    mixed_precision="bf16_mixed",  # This takes precedence during training
    # ... other configs
)
```

The mixed precision configuration will set `bf16=True` on both the model and optimizer configs, overriding the explicitly set `False` values. This synchronization prevents configuration mismatches that could lead to training issues.

## Performance Considerations

- **FP8 recipes are experimental** and convergence has not been fully validated for all models
- **BF16** is generally recommended over FP16 for better numerical stability
- **FP8** provides the best performance on H100 GPUs but requires careful tuning
- **MXFP8** recipes are only supported on Blackwell architecture GPUs
- **Blockwise scaling** recipes are optimized for Hopper architecture GPUs

## Resources

- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [Intro to FP8, floating point formats, and mixed precision training](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Introduction-to-FP8)
- [Performance optimizations](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html) that are natively supported in Megatron Bridge by enabling FP8 training with TE
