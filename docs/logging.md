# Logging and Monitoring

This guide describes how to configure logging in Megatron Bridge. It introduces the high-level `LoggerConfig`, explains experiment logging to TensorBoard and Weights & Biases (W&B), and documents console logging behavior.

## LoggerConfig Overview

`LoggerConfig` is a dataclass that encapsulates logging‑related settings for training. It resides inside the overall `ConfigContainer`, which represents the complete configuration for a training run. For the complete API and all available fields, see {py:class}`~megatron.bridge.training.config.LoggerConfig`.

### Timing and timers

`timing_log_level` controls which timers are measured. Level 0 measures only overall iteration time. Level 1 adds once‑per‑iteration operations such as gradient all‑reduce. Level 2 also includes frequently executed operations and therefore has higher overhead. `timing_log_option` controls how timer values are aggregated across ranks and can be set to `"max"`, `"minmax"`, or `"all"`. When `log_timers_to_tensorboard` is enabled, the framework records timer metrics to supported backends.

### Additional Features

Additional toggles include loss scale, validation perplexity, CUDA memory statistics, and world size. `log_params_norm` enables computing and logging the model parameter L2 norm (and, when available, gradient norm). `log_energy` enables the energy monitor, which records per‑GPU energy and instantaneous power.

## Experiment Logging
Both TensorBoard and Weights & Biases are supported. Enabling TensorBoard is recommended when using Weights & Biases to ensure all scalar metrics are emitted consistently.

### TensorBoard

**What gets logged**

Learning rate (and decoupled LR when used), per-loss scalars, batch size, loss scale, optional CUDA memory/world size, validation loss and optional perplexity, timers if enabled, and optional energy/power.

**How to enable**
  1) Install TensorBoard (if not already available):
  ```bash
  pip install tensorboard
  ```
  2) Provide the following in your configuration. In these examples, `cfg` refers to the `ConfigContainer` instance (for example, one produced by a recipe), which contains a `logger` attribute representing the `LoggerConfig`:
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

**Directory**

Events are written under the provided `tensorboard_dir` directory.

Additional example enabling more metrics:
```python
cfg.logger.tensorboard_dir = "./logs/tb"
cfg.logger.tensorboard_log_interval = 5
cfg.logger.log_loss_scale_to_tensorboard = True
cfg.logger.log_validation_ppl_to_tensorboard = True
cfg.logger.log_world_size_to_tensorboard = True
cfg.logger.log_timers_to_tensorboard = True
```

### Weights & Biases (W&B)

**What gets logged**

Data logged to W&B mirrors the TensorBoard scalar metrics when the TensorBoard. The full run configuration is also synced at initialization.

**How to enable**

  1) Install W&B (if not already available):
  ```bash
  pip install wandb
  ```
  2) Authenticate with W&B (one of):
  - Set `WANDB_API_KEY` in the environment before the run, or
  - Run `wandb login` once on the machine.
  3) Provide the following in your configuration. In these examples, `cfg` refers to your `ConfigContainer` instance which contains a `logger` attribute representing the `LoggerConfig`:
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

**Recipe users with NeMo-Run**

You can optionally configure W&B via the `WandbPlugin`. The plugin forwards `WANDB_API_KEY` and injects CLI overrides such as `logger.wandb_project`, `logger.wandb_entity`, `logger.wandb_exp_name`, and `logger.wandb_save_dir`.

### Progress log

When `logger.log_progress` is enabled, the framework writes a `progress.txt` file under the checkpoint save directory. The file records job-level metadata (such as timestamp and GPU count) and periodic progress entries. At checkpoint boundaries, the framework appends entries with job throughput (TFLOP/s/GPU), cumulative throughput, total floating‑point operations, and tokens processed. This provides a lightweight, text‑based audit of training progress across restarts.

## Console logging

Megatron Bridge uses the standard Python logging subsystem for console output. 

### Configuration

The `logging_level` controls the default level and can also be overridden via the `MEGATRON_BRIDGE_LOGGING_LEVEL` environment variable. `filter_warnings` suppresses WARNING messages. `modules_to_filter` specifies logger name prefixes to filter out. `set_level_for_all_loggers` controls whether the level is applied to all loggers or only a subset, depending on the current implementation.

### Cadence and content

Every `log_interval` iterations, the framework prints a consolidated summary line that includes a timestamp, iteration counters, consumed and skipped samples, iteration time (ms), learning rates, global batch size, per‑loss averages, and loss scale. When enabled, it also prints gradient norm, zeros in gradients, parameter norm, and energy and power per GPU. Straggler timing reports follow the same `log_interval` cadence.

### Overhead considerations

For minimal overhead, keep `timing_log_level` at 0. Increase to 1 or 2 only when more detailed timing is needed.
