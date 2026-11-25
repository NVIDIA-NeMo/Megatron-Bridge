# Optimizer and Scheduler Configuration

The optimizer and scheduler configurations control optimization algorithms, learning rate schedules, and weight decay strategies.

## OptimizerConfig (from Megatron Core)

The `OptimizerConfig` contains all parameters for the optimization algorithm and comes directly from Megatron Core. Key parameters include:

| Parameter | Type | Description |
|-----------|------|-------------|
| `optimizer` | `str` | Optimizer type ("adam", "sgd", etc.) |
| `lr` | `float` | Base learning rate |
| `min_lr` | `float` | Minimum learning rate for decay schedules |
| `weight_decay` | `float` | L2 regularization coefficient |
| `adam_beta1` | `float` | Adam optimizer beta1 parameter |
| `adam_beta2` | `float` | Adam optimizer beta2 parameter |
| `adam_eps` | `float` | Adam optimizer epsilon parameter |
| `clip_grad` | `float` | Gradient clipping threshold |
| `use_distributed_optimizer` | `bool` | Enable distributed optimizer for memory efficiency |
| `overlap_grad_reduce` | `bool` | Overlap gradient reduction with computation |
| `overlap_param_gather` | `bool` | Overlap parameter gathering with computation |
| `bf16` | `bool` | Use BF16 precision for training |
| `fp16` | `bool` | Use FP16 precision for training |

## Recipe parameter-group overrides

Use the same Core mapping as MLM, or a callable that receives the constructed
model and optimizer config:

```python
from megatron.core.optimizer import ParamKey

cfg.optimizer_config_overrides = {
    ParamKey(name="*.linear_out.weight"): {"max_lr": 0.001, "min_lr": 0.0001}
}
cfg.optimizer_config_overrides_mode = "extend"  # default
```

For type annotations, import `OptimizerConfigOverrides` and
`OptimizerConfigOverrideHook` from `megatron.training.config.optimizer_config`.
This is the shared public import path for MLM and Bridge recipes.

A callable is invoked as `hook(optimizer_config=..., model=...)` after model
construction and must return a mapping, including an empty one. Use an importable
function or configured dataclass callable for serialized recipes. Static selector
mappings use the shared config serializer and existing target allowlist; this
changes configuration serialization, not optimizer checkpoint data.

Default `extend` mode retains standard weight-decay and decoupled-LR rules.
Explicit `replace` mode replaces those standard rules, including with an empty
mapping. Enabled muP rules are composed in either mode. Core's
`merge_config_overrides` merges equal/disjoint fields on identical selectors;
conflicts raise, including overlaps detected when constructing actual parameter
groups. There is no dictionary-order precedence. To change a standard WD rule,
use replace mode and provide the intended complete mapping.

For LSQ scale learning, use the callable `LSQOptimizerConfigOverrides`:

```python
from megatron.bridge.training.post_training.lsq import LSQOptimizerConfigOverrides

cfg.optimizer_config_overrides = LSQOptimizerConfigOverrides(
    scale_lr=3e-5, scale_min_lr=None, scale_wd_mult=0.0
)
```

MLM's `get_lsq_config_overrides` owns the scale LR and weight-decay rules; the
Bridge callable adds the ModelOpt scale-gradient check. With the default extend
mode, the resulting parameter groups match the old LSQ provider. The old LSQ
class delegates to the callable and remains loadable from existing configurations.
The new callable's dataclass settings remain configurable through YAML and CLI overrides, such as
`optimizer_config_overrides.scale_lr=3e-5`.

The provider API is still used by downstream
[NeMo RL draft optimizers](https://github.com/NVIDIA-NeMo/RL/blob/d93ba7ddb4b5978a27d7433fd028c42ef766a9bc/nemo_rl/models/megatron/draft/optimizer.py)
and [BioNeMo Hyena recipes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/2711f74b13aff333c26031725eae06eedf47ac0f/recipes/evo2_megatron/src/bionemo/evo2/models/evo2_provider.py).
These integrations must migrate before the provider field and classes can be removed.
New recipes should use `optimizer_config_overrides`.

Existing `optimizer_config_override_provider` implementations remain supported
as complete policies. Providers using `super().build_config_overrides()`,
including LSQ, keep their standard rules; a custom provider returning `None` or
`{}` retains its previous behavior. LoRA+ warmup constraints remain enforced for
both APIs. A custom provider and the new mapping/callable cannot be configured
together. When porting a complete provider mapping to the new API, select replace
mode; for contributions containing only additions, select extend mode.

The ordinary hook rejects custom MIMO policy, which must be configured in MIMO's
per-module optimizer factories. This interface requires the companion Core
mapping-composition and configuration-serialization helpers.

## Distributed Muon

Use Bridge's `OptimizerConfig` to select Muon and distribute whole parameter tensors:

```python
from megatron.bridge.training.config import OptimizerConfig

cfg.optimizer = OptimizerConfig(
    optimizer="muon",
    use_layer_wise_distributed_optimizer=True,
    lr=1e-3,
    bf16=True,
)
```

When changing an existing distributed Adam recipe to Muon, Bridge also accepts
`use_distributed_optimizer=True` on the optimizer or DDP configuration and
resolves it to layer-wise optimizer distribution before constructing the model.
The legacy `optimizer="dist_muon"` spelling resolves to the same mode. Adam and
SGD continue to use their ordinary distributed optimizer. To run replicated
Muon, disable both distributed optimizer flags and the DDP distributed flag.

Parameter-gather overlap can be configured through `CommOverlapConfig` for
layer-wise Muon. The current Core implementation requires one distributed
optimizer instance and does not support `overlap_param_gather_with_optimizer_step`
or either FSDP implementation with Muon.

Runtime setup applies explicit communication-overlap overrides before validating
the effective optimizer settings. For LayerWise optimizers, an unspecified
`CommOverlapConfig.bucket_size` preserves `ddp.bucket_size`; when that is also
unset, Core selects its DP-dependent bucket size. An explicit overlap bucket size
still takes precedence. Likewise, an unspecified `overlap_grad_reduce` preserves
the DDP setting, including at DP=1 where CP ranks may still reduce gradients.
Explicit gradient-overlap overrides take precedence. Adam/SGD retain the ordinary
distributed-optimizer presets.

By default, `use_layer_wise_param_layout=False` retains compact DDP buffers and
LayerWise's variable-size parameter synchronization path. This preserves the
optimizer representation of existing Muon checkpoints while retaining the
configured parameter-gather behavior. Set it to `True` to opt into shard-aligned
buffers for a compatible model and optimizer.

The current Core full-layout path does not support Lion/SOAP matrices marked
`use_muon=False`, expert-parallel scalar fallback parameters (such as MoE expert
biases), or optimizer domains whose trainable parameters all use the scalar
fallback (such as bias-only tuning or a scalar-only MIMO component). Keep compact
layout for these cases. Also retain the checkpoint's layout when resuming
optimizer state: loading an older compact Muon checkpoint into the full-layout
optimizer representation is not supported.

Existing Kimi recipes keep the compact layout and parameter-gather overlap
disabled until their MoE and pipeline-parallel configurations are validated
with the shard-aligned layout; other recipes can opt in independently.
They also retain Core-selected bucket sizes and gradient-reduce overlap, including
DP=1 with CP>1 and the H100 Kimi K2 performance recipes.
Explicit user overrides remain supported.

Resuming Muon training restores Muon optimizer state. When changing optimization
algorithms between runs, load model weights with fresh optimizer state using
`checkpoint.load_optim=False` or the appropriate finetuning configuration;
Bridge does not convert Adam moments into Muon optimizer state.
Checkpoints created before distributed-optimizer metadata was recorded retain
the historical model-space sharding fallback when that metadata key is absent.

## SchedulerConfig

The `SchedulerConfig` controls learning rate scheduling and weight decay progression throughout training.

### Learning Rate Scheduling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_decay_style` | `Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"]` | `"linear"` | Learning rate decay function |
| `lr_decay_iters` | `Optional[int]` | `None` | Iterations to decay LR over (defaults to `train_iters`). Use for iteration-based training. |
| `lr_decay_samples` | `Optional[int]` | `None` | Samples to decay LR over (defaults to `train_samples`). Use for sample-based training. |
| `lr_warmup_iters` | `int` | `0` | Iterations to linearly warmup learning rate. Use for iteration-based training. |
| `lr_warmup_samples` | `int` | `0` | Samples to linearly warmup learning rate. Use for sample-based training. |
| `lr_warmup_fraction` | `Optional[float]` | `None` | Fraction of decay iterations/samples to use for warmup (works with both modes) |
| `lr_warmup_init` | `float` | `0.0` | Initial learning rate for warmup phase |

**Scheduler Mode Selection**

The scheduler supports two modes that must align with your training configuration:

1. **Iteration-based scheduling**: Use `lr_decay_iters` and `lr_warmup_iters` with `train_iters`.
2. **Sample-based scheduling**: Use `lr_decay_samples` and `lr_warmup_samples` with `train_samples`.

**Important constraints**
- Cannot mix iteration-based and sample-based scheduler parameters.
- Your scheduler mode must match your training mode (iteration-based vs sample-based).
- `lr_warmup_fraction` is compatible with both modes but cannot be used with explicit warmup iterations/samples.

### WSD (Warmup-Stable-Decay) Scheduling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_wsd_decay_style` | `Literal["exponential", "linear", "cosine"]` | `"exponential"` | Decay style for WSD annealing phase |
| `lr_wsd_decay_iters` | `Optional[int]` | `None` | Iterations for WSD annealing phase. Use for iteration-based training. |
| `lr_wsd_decay_samples` | `Optional[int]` | `None` | Samples for WSD annealing phase. Use for sample-based training. |

### Weight Decay Scheduling

Parameters for controlling the progression of weight decay during training, including start and end values and the scheduling strategy:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_weight_decay` | `Optional[float]` | `None` | Initial weight decay coefficient |
| `end_weight_decay` | `Optional[float]` | `None` | Final weight decay coefficient |
| `weight_decay_incr_style` | `Literal["constant", "linear", "cosine"]` | `"constant"` | Weight decay progression style |

### Checkpoint Integration

Parameters for managing how scheduler settings are applied during checkpoint loading, allowing control over whether to prioritize config values or restore from saved state:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `override_opt_param_scheduler` | `bool` | `False` | Reset scheduler values from config, ignoring checkpoint |
| `use_checkpoint_opt_param_scheduler` | `bool` | `False` | Use scheduler values from checkpoint, ignoring config |

### Computed Fields

These fields are automatically calculated during configuration validation and help align training schedules with the configured batch size and iteration counts:

| Field | Description |
|-------|-------------|
| `lr_warmup_steps` | Total steps for warmup (calculated from iterations and batch size) |
| `lr_decay_steps` | Total steps for decay (calculated from iterations and batch size) |
| `wd_incr_steps` | Total steps for weight decay progression |
| `wsd_decay_steps` | Total steps for WSD annealing phase |

## Learning Rate Schedules

The following scheduling strategies define how the learning rate evolves during training, each suited to different convergence behaviors and model types:
| Schedule Type           | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Constant**            | Learning rate remains fixed throughout training.                            |
| **Linear**              | Learning rate decreases linearly from the base LR to the minimum LR.        |
| **Cosine**              | Learning rate follows a cosine decay curve from base LR to minimum LR.      |
| **Inverse Square Root** | Learning rate decays proportionally to the inverse square root of the step. |

## WSD (Warmup-Stable-Decay)
The WSD schedule divides learning rate progression into three distinct phases, offering fine-grained control over early ramp-up, mid-training stability, and final decay:
| Phase     | Description                                              |
|-----------|----------------------------------------------------------|
| **Warmup** | Learning rate increases linearly from initial value to base LR. |
| **Stable** | Learning rate remains constant at base LR.              |
| **Decay**  | Learning rate decays to minimum LR using a specified style (e.g., exponential, linear, cosine). |

## Weight Decay Scheduling

These scheduling options control how the weight decay coefficient changes over time, allowing for regularization strategies that adapt to different training phases:
| Schedule Type | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **Constant**  | Fixed weight decay throughout training.                                     |
| **Linear**    | Linear progression from start to end weight decay.                          |
| **Cosine**    | Cosine progression from start to end weight decay.                          |
