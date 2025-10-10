# Logging and Monitoring

This guide describes how to configure logging in Megatron Bridge. It introduces the high-level `LoggerConfig`, explains experiment logging to TensorBoard and Weights & Biases (W&B), and documents console logging behavior.

## LoggerConfig Overview

{py:class}`~bridge.training.config.LoggerConfig` is the dataclass that encapsulates logging‑related settings for training. It resides inside the overall {py:class}`bridge.training.config.ConfigContainer`, which represents the complete configuration for a training run.

### Timer Configuration Options

Use the following options to control which timing metrics are collected during training and how they are aggregated and logged.

#### `timing_log_level`
Controls which timers are recorded during execution:

- **Level 0**: Logs only the overall iteration time.
- **Level 1**: Includes once-per-iteration operations, such as gradient all-reduce.
- **Level 2**: Captures frequently executed operations, providing more detailed insights but with increased overhead.

#### `timing_log_option`
Specifies how timer values are aggregated across ranks. Valid options:

- `"max"`: Logs the maximum value across ranks.
- `"minmax"`: Logs both minimum and maximum values.
- `"all"`: Logs all values from all ranks.

#### `log_timers_to_tensorboard`
When enabled, the framework records timer metrics to supported backends such as TensorBoard.


### Diagnostic Options

The framework provides several optional toggles for enhanced monitoring and diagnostics:

- **Loss Scale**: Enables dynamic loss scaling for mixed-precision training.
- **Validation Perplexity**: Tracks model perplexity during validation.
- **CUDA Memory Statistics**: Reports detailed GPU memory usage.
- **World Size**: Displays the total number of distributed ranks.

### Logging Options

Use the following options to enable additional diagnostics and performance monitoring during training.

- **`log_params_norm`**: Computes and logs the L2 norm of model parameters. If available, it also logs the gradient norm.
- **`log_energy`**: Activates the energy monitor, which records per-GPU energy consumption and instantaneous power usage.


## Experiment Logging
Both TensorBoard and W&B are supported for metric logging. When using W&B, it’s recommended to also enable TensorBoard to ensure that all scalar metrics are consistently logged across backends.

### TensorBoard

 
#### What Gets Logged

TensorBoard captures a range of training and system metrics, including:

- **Learning rate**, including decoupled LR when applicable
- **Per-loss scalars** for detailed breakdowns
- **Batch size** and **loss scale**
- **CUDA memory usage** and **world size** (if enabled)
- **Validation loss**, with optional **perplexity**
- **Timers**, when timing is enabled
- **Energy consumption** and **instantaneous power**, if energy logging is active


#### Enable TensorBoard Logging
  1) Install TensorBoard (if not already available):
  ```bash
  pip install tensorboard
  ```
  2) **Configure logging** in your training setup. In these examples, `cfg` refers to a `ConfigContainer` instance (such as one produced by a recipe), which contains a `logger` attribute representing the `LoggerConfig`:
  
  ```python
  from megatron.bridge.training.config import LoggerConfig

  cfg.logger = LoggerConfig(
      tensorboard_dir="./runs/tensorboard",
      tensorboard_log_interval=10,
      log_timers_to_tensorboard=True,   # optional
      log_memory_to_tensorboard=False,  # optional
  )
  ```

  ```{note}
  The writer is created lazily on the last rank when `tensorboard_dir` is set.
  ```

#### Set the Output Directory

TensorBoard event files are saved to the directory specified by `tensorboard_dir`.

**Example with additional metrics enabled:**
```python
cfg.logger.tensorboard_dir = "./logs/tb"
cfg.logger.tensorboard_log_interval = 5
cfg.logger.log_loss_scale_to_tensorboard = True
cfg.logger.log_validation_ppl_to_tensorboard = True
cfg.logger.log_world_size_to_tensorboard = True
cfg.logger.log_timers_to_tensorboard = True
```

### Weights & Biases (W&B)

  
#### What Gets Logged

When enabled, W&B automatically mirrors the scalar metrics logged to TensorBoard.  
In addition, the full run configuration is synced at initialization, allowing for reproducibility and experiment tracking.


#### Enable W&B Logging

  1) Install W&B (if not already available):
  ```bash
  pip install wandb
  ```
  2) Authenticate with W&B using one of the following methods:
  - Set `WANDB_API_KEY` in the environment before the run, or
  - Run `wandb login` once on the machine.
  2) **Configure logging** in your training setup. In these examples, `cfg` refers to a `ConfigContainer` instance (such as one produced by a recipe), which contains a `logger` attribute representing the `LoggerConfig`:
  
  ```python
  from megatron.bridge.training.config import LoggerConfig

  cfg.logger = LoggerConfig(
      tensorboard_dir="./runs/tensorboard",   # recommended: enables shared logging gate
      wandb_project="my_project",
      wandb_exp_name="my_experiment",
      wandb_entity="my_team",                 # optional
      wandb_save_dir="./runs/wandb",          # optional
  )
  ```
  
```{note}
W&B is initialized lazily on the last rank when `wandb_project` is set and `wandb_exp_name` is non-empty.
```  

#### W&B Configuration with NeMo Run Launching

For users launching training scripts with NeMo Run, W&B can be optionally configured using the {py:class}`bridge.recipes.run_plugins.WandbPlugin`.

The plugin automatically forwards the `WANDB_API_KEY` and by default injects CLI overrides for the following logger parameters:

- `logger.wandb_project`  
- `logger.wandb_entity`  
- `logger.wandb_exp_name`  
- `logger.wandb_save_dir`

This allows seamless integration of W&B logging into your training workflow without manual configuration.


#### Progress Log

When `logger.log_progress` is enabled, the framework generates a `progress.txt` file in the checkpoint save directory.

This file includes:
- **Job-level metadata**, such as timestamp and GPU count
- **Periodic progress entries** throughout training

At each checkpoint boundary, the log is updated with:
- **Job throughput** (TFLOP/s/GPU)
- **Cumulative throughput**
- **Total floating-point operations**
- **Tokens processed**

This provides a lightweight, text-based audit trail of training progress, useful for tracking performance across restarts.


## Tensor Inspection

Megatron Bridge integrates with TransformerEngine's tensor inspection features via NVIDIA DLFW Inspect. This integration, controlled by {py:class}`~bridge.training.config.TensorInspectConfig`, enables advanced debugging and analysis of tensor statistics during training. When enabled, the framework handles initialization, step tracking, and cleanup automatically.

```{note}
**Current limitations:** Tensor inspection is currently supported only for linear modules in TransformerEngine (e.g., `fc1`, `fc2`, `layernorm_linear`). Operations like attention are not supported.
```

```{note}
This section covers Megatron Bridge configuration. For comprehensive documentation on features, configuration syntax, and advanced usage, see:

- [TransformerEngine Debug Documentation](https://github.com/NVIDIA/TransformerEngine/tree/af2a0c16ec11363c0af84690cd877a59f898820e/docs/debug)
- [NVIDIA DLFW Inspect Documentation](https://github.com/NVIDIA/nvidia-dlfw-inspect/tree/4118044cc84f0183714a2ab1bc215fa49f9aaa82/docs)
```

### Installation

Install NVIDIA DLFW Inspect if not already available:
```bash
pip install nvdlfw-inspect
```

### Available Features

TransformerEngine provides the following debug features:

- **LogTensorStats** – Logs high-precision tensor statistics: `min`, `max`, `mean`, `std`, `l1_norm`, `l2_norm`, `cur_amax`, `dynamic_range`.
- **LogFp8TensorStats** – Logs quantized tensor statistics for FP8 recipes: `underflows%`, `scale_inv_min`, `scale_inv_max`, `mse`. Supports simulating alternative recipes (e.g., tracking `mxfp8_underflows%` during per-tensor current-scaling training)
- **DisableFP8GEMM** – Runs specific GEMM operations in high precision
- **DisableFP8Layer** – Disables FP8 for entire layers
- **PerTensorScaling** – Enables per-tensor current scaling for specific tensors
- **FakeQuant** – Experimental quantization testing

See [TransformerEngine debug features](https://github.com/NVIDIA/TransformerEngine/tree/af2a0c16ec11363c0af84690cd877a59f898820e/transformer_engine/debug/features) for complete parameter lists and usage details.

### Configuration

Configure tensor inspection using {py:class}`~bridge.training.config.TensorInspectConfig` with either a YAML file or inline dictionary.

#### YAML Configuration

```yaml
tensor_inspect:
  enabled: true
  features: ./conf/fp8_tensor_stats.yaml
  log_dir: ./logs/tensor_inspect
```

**Example feature configuration file:**

```yaml
fp8_tensor_stats:
  enabled: true
  layers:
    layer_name_regex_pattern: ".*(fc2)"
  transformer_engine:
    LogFp8TensorStats:
      enabled: true
      tensors: [weight,activation,gradient]
      stats: ["underflows%", "mse"]
      freq: 5
      start_step: 0
      end_step: 100
```

#### Python Configuration

```python
from bridge.training.config import TensorInspectConfig

# Option 1: inline python dict
cfg.tensor_inspect = TensorInspectConfig(
    enabled=True,
    features={
        "fp8_gradient_stats": {
            "enabled": True,
            "layers": {"layer_name_regex_pattern": ".*(fc1|fc2)"},
            "transformer_engine": {
                "LogFp8TensorStats": {
                    "enabled": True,
                    "tensors": ["weight","activation","gradient"],
                    "stats": ["underflows%", "mse"],
                    "freq": 5,
                    "start_step": 0,
                    "end_step": 100,
                },
            },
        }
    },
    log_dir="./logs/tensor_inspect",
)

# Option 2: reference external YAML
cfg.tensor_inspect = TensorInspectConfig(
    enabled=True,
    features="./conf/fp8_inspect.yaml",
    log_dir="./logs/tensor_inspect",
)

```

#### Layer Selection

Features apply to linear modules matched by selectors in the `layers` section:

- `layer_name_regex_pattern: .*` – All supported linear layers
- `layer_name_regex_pattern: .*layers\.(0|1|2).*(fc1|fc2|layernorm_linear)` – Linear modules in first three transformer layers
- `layer_name_regex_pattern: .*(fc1|fc2)` – MLP projections only
- `layer_types: [layernorm_linear, fc1]` – String matching (alternative to regex)

Tensor-level selectors (`tensors`, `tensors_struct`) control which tensor roles are logged: `activation`, `gradient`, `weight`, `output`, `wgrad`, `dgrad`.

### Output and Monitoring

Tensor statistics are written to `tensor_inspect.log_dir` and forwarded to TensorBoard/W&B when enabled.

**Log locations:**
- Text logs: `<log_dir>/nvdlfw_inspect_statistics_logs/`
- TensorBoard
- W&B

### Performance Considerations

- Use `freq > 1` to reduce overhead. Statistics collection is expensive for large models.
- Narrow layer selection with specific regex patterns rather than `.*`


## Console Logging

Megatron Bridge uses the standard Python logging subsystem for console output. 

### Configure Console Logging

To control console logging behavior, use the following configuration options:

- `logging_level` sets the default verbosity level. It can be overridden via the `MEGATRON_BRIDGE_LOGGING_LEVEL` environment variable.
- `filter_warnings` suppresses messages at the WARNING level.
- `modules_to_filter` specifies logger name prefixes to exclude from output.
- `set_level_for_all_loggers` determines whether the logging level is applied to all loggers or only a subset, depending on the current implementation.


### Monitor Logging Cadence and Content

To monitor training progress at regular intervals, the framework prints a summary line every `log_interval` iterations.

Each summary includes:
- **Timestamp**
- **Iteration counters**
- **Consumed and skipped samples**
- **Iteration time (ms)**
- **Learning rates**
- **Global batch size**
- **Per-loss averages**
- **Loss scale**

When enabled, additional metrics are printed:
- **Gradient norm**
- **Zeros in gradients**
- **Parameter norm**
- **Energy and power per GPU**

Straggler timing reports follow the same `log_interval` cadence, helping identify performance bottlenecks across ranks.


### Minimize Timing Overhead

To reduce performance impact, set `timing_log_level` to `0`.  
Increase to `1` or `2` only when more detailed timing metrics are required, as higher levels introduce additional logging overhead.

