# Profiling

Megatron Bridge provides built-in support for profiling your training jobs using various performance analysis tools, including NVIDIA Nsight Systems (Nsys) for workflow optimization and PyTorch-based profilers and memory trackers for tracking performance and memory usage patterns during training.

## ProfilingConfig Overview

`ProfilingConfig` is a dataclass that encapsulates profiling-related settings for training. It resides inside the overall `ConfigContainer`, which represents the complete configuration for a training run. For the complete API and all available fields, see {py:class}`~megatron.bridge.training.config.ProfilingConfig`.

### Profiling Selection

The configuration provides mutually exclusive profiling options. You can enable either NSys profiling (`use_nsys_profiler`) or PyTorch profiling (`use_pytorch_profiler`), but not both simultaneously.

### Step Range and Ranks

All profiling modes support configuring the step range (`profile_step_start` and `profile_step_end`) and target ranks (`profile_ranks`). By default, profiling targets rank 0, but you can specify multiple ranks to profile different parts of your distributed training setup.

### Additional Features

The configuration includes options for recording tensor shapes (`record_shapes`) and memory profiling (`record_memory_history` with configurable `memory_snapshot_path`). These features provide deeper insights into your model's memory usage patterns and tensor operations.

## NSys Profiling

NVIDIA Nsys is a system-wide performance analysis tool designed to help you tune and optimize CUDA applications. Megatron Bridge integrates with Nsys to enable profiling specific steps of your training job, making it easy to collect detailed performance data without manual instrumentation.

```{note}
NSys profiling cannot be used with the `FaultTolerancePlugin` due to implementation conflicts. If both are enabled, the framework will automatically disable NSys profiling and emit a warning.
```

### Configuration Options

Enable NSys profiling by setting `use_nsys_profiler=True` in your `ProfilingConfig`. The key configuration options include:

```python
from megatron.bridge.training.config import ProfilingConfig

# In your ConfigContainer setup, cfg is a ConfigContainer instance
cfg.profiling = ProfilingConfig(
    use_nsys_profiler=True,
    profile_step_start=10,
    profile_step_end=15,
    profile_ranks=[0, 1],  # Profile first two ranks
    record_shapes=False,   # Optional: record tensor shapes
)
```

### Launching with NSys

When using NSys profiling, launch your training script with the NSys command wrapper:

```bash
nsys profile -s none -o <profile_filepath> -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python <path_to_script>
```

Replace `<profile_filepath>` with your desired output path and `<path_to_script>` with your training script. The `--capture-range=cudaProfilerApi` option ensures profiling is controlled by the framework's step range configuration.

### NSys Plugin for NeMo-Run Users

Recipe users can leverage the `NsysPlugin` to configure NSys profiling through NeMo-Run executors. The plugin provides a convenient interface for setting up profiling without manually configuring the underlying NSys command.

```python
import nemo_run as run
from megatron.bridge.recipes.run_plugins import NsysPlugin

# Create your recipe and executor
recipe = your_recipe_function()
executor = run.SlurmExecutor(...)

# Configure NSys profiling via plugin
plugins = [
    NsysPlugin(
        profile_step_start=10,
        profile_step_end=15,
        profile_ranks=[0, 1],
        nsys_trace=["nvtx", "cuda"],  # Optional: specify trace events
        record_shapes=False,
        nsys_gpu_metrics=False,
    )
]

# Run with profiling enabled
with run.Experiment("nsys_profiling_experiment") as exp:
    exp.add(recipe, executor=executor, plugins=plugins)
    exp.run()
```

The plugin automatically configures the NSys command line options and sets up the profiling configuration in your training job.

### Analyzing Results

Once your profiling run is complete, you'll have generated NSys profile files that can be opened with the NSys GUI. Install [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) from the NVIDIA Developer website, open the generated `.nsys-rep` files, and use the timeline view to examine your training job's performance characteristics.

## PyTorch Profiler

Megatron Bridge supports the built-in PyTorch profiler, which is useful for viewing profiles in TensorBoard and understanding PyTorch-level performance characteristics.

### Configuration

Enable PyTorch profiling by setting `use_pytorch_profiler=True` in your `ProfilingConfig`:

```python
from megatron.bridge.training.config import ProfilingConfig

cfg.profiling = ProfilingConfig(
    use_pytorch_profiler=True,
    profile_step_start=10,
    profile_step_end=15,
    profile_ranks=[0],
    record_shapes=True,    # Record tensor shapes for detailed analysis
)
```

### PyTorch Profiler Plugin

Similar to NSys, recipe users can use the `PyTorchProfilerPlugin` for convenient configuration:

```python
from megatron.bridge.recipes.run_plugins import PyTorchProfilerPlugin

plugins = [
    PyTorchProfilerPlugin(
        profile_step_start=10,
        profile_step_end=15,
        profile_ranks=[0],
        record_memory_history=True,
        memory_snapshot_path="memory_snapshot.pickle",
        record_shapes=True,
    )
]
```

## Memory Profiling

Megatron Bridge provides built-in support for CUDA memory profiling to track and analyze memory usage patterns during training, including GPU memory allocation and consumption tracking.

More information about the generated memory profiles can be found [here](https://pytorch.org/blog/understanding-gpu-memory-1/).

### Configuration

Enable memory profiling by setting `record_memory_history=True` in your `ProfilingConfig`. This can be used with either profiling mode:

```python
from megatron.bridge.training.config import ProfilingConfig

cfg.profiling = ProfilingConfig(
    use_pytorch_profiler=True,  # or use_nsys_profiler=True
    profile_step_start=10,
    profile_step_end=15,
    profile_ranks=[0],
    record_memory_history=True,
    memory_snapshot_path="memory_trace.pickle",  # Customize output path
)
```

### Memory Analysis

Once the run completes, the specified path will contain memory snapshots for each specified rank. These traces can be loaded with the PyTorch Memory Viz tool to plot memory usage over time and identify memory bottlenecks or leaks in your training pipeline.

## Performance Considerations

Profiling adds overhead to your training job, so measured timings may be slightly higher than normal operation. For accurate profiling results, disable other intensive operations like frequent checkpointing during the profiled step range. Choose your profiling step range carefully to capture representative training behavior while minimizing the performance impact on the overall job.
