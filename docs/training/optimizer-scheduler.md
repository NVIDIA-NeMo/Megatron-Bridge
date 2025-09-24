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

## SchedulerConfig

The `SchedulerConfig` controls learning rate scheduling and weight decay progression throughout training.

### Learning Rate Scheduling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_decay_style` | `Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"]` | `"linear"` | Learning rate decay function |
| `lr_decay_iters` | `Optional[int]` | `None` | Iterations to decay LR over (defaults to `train_iters`) |
| `lr_warmup_iters` | `int` | `0` | Iterations to linearly warmup learning rate |
| `lr_warmup_fraction` | `Optional[float]` | `None` | Fraction of decay iterations to use for warmup |
| `lr_warmup_init` | `float` | `0.0` | Initial learning rate for warmup phase |

### WSD (Warmup-Stable-Decay) Scheduling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_wsd_decay_style` | `Literal["exponential", "linear", "cosine"]` | `"exponential"` | Decay style for WSD annealing phase |
| `lr_wsd_decay_iters` | `Optional[int]` | `None` | Iterations for WSD annealing phase |

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