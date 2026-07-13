# Profiling

Megatron Bridge provides built-in support for profiling training jobs using a range of performance analysis tools. These include NVIDIA Nsight Systems (Nsys) for workflow optimization, as well as PyTorch-based profilers and memory trackers to monitor performance and memory usage patterns during training.

## ProfilingConfig Overview

{py:class}`bridge.training.config.ProfilingConfig` is a dataclass that encapsulates profiling-related settings for training. It resides inside the overall {py:class}`bridge.training.config.ConfigContainer`, which represents the complete configuration for a training run.


### Profiling Options

The configuration supports two mutually exclusive profiling options:

- **NSys profiling** (`use_nsys_profiler`)
- **PyTorch profiling** (`use_pytorch_profiler`)

You can enable one or the other, but not both at the same time.


### Step Range and Target Ranks

All profiling modes allow you to configure:

- **Step range**: `profile_step_start` and `profile_step_end`
- **Target ranks**: `profile_ranks`

By default, profiling targets rank 0. You can specify multiple ranks to analyze different parts of your distributed training setup.


### Advanced Profiling Features

The configuration includes options for recording tensor shapes (`record_shapes`) and enabling memory profiling (`record_memory_history`) with a customizable output path (`memory_snapshot_path`). These features offer deeper visibility into your model’s memory consumption and tensor-level operations during training.


## NSys Profiling

NVIDIA Nsys is a system-wide performance analysis tool designed to help you tune and optimize CUDA applications. Megatron Bridge integrates with Nsys to enable profiling specific steps of your training job, making it easy to collect detailed performance data without manual instrumentation.

```{note}
NSys profiling cannot be used with the `FaultTolerancePlugin` due to implementation conflicts. If both are enabled, the framework will automatically disable NSys profiling and emit a warning.
```

### Configure NSys Profiling

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

### Launch with NSys

When using NSys profiling, launch your training script with the NSys command wrapper:

```bash
nsys profile -s none -o <profile_filepath> -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python <path_to_script>
```

Replace `<profile_filepath>` with your desired output path and `<path_to_script>` with your training script. The `--capture-range=cudaProfilerApi` option ensures profiling is controlled by the framework's step range configuration.

### Configure Profiling with the NeMo Run NSys Plugin

Recipe users can leverage the {py:class}`bridge.recipes.run_plugins.NsysPlugin` to configure NSys profiling through NeMo Run executors. The plugin provides a convenient interface for setting up profiling without manually configuring the underlying NSys command.

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

### Analyze Results

After your profiling run completes, the NSys profile files (`.nsys-rep`) will be generated. To analyze them, install [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) from the NVIDIA Developer website, open the files in the NSys GUI, and use the timeline view to explore the performance characteristics of your training job.

## PyTorch Profiler

Megatron Bridge supports the built-in PyTorch profiler, which is useful for viewing profiles in TensorBoard and understanding PyTorch-level performance characteristics.

### Configure PyTorch Profiler

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

### Configure Profiling with the PyTorch Profiler Plugin

Similar to NSys, recipe users can use the {py:class}`bridge.recipes.run_plugins.PyTorchProfilerPlugin` for convenient configuration:

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

### Configure Memory Profiling

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

### Analyze Memory Usage

After the run completes, memory snapshots for each specified rank are saved to the designated path. Load these traces using the PyTorch Memory Viz tool to plot memory usage over time and detect bottlenecks or leaks in your training pipeline.

## Optimize Profiling Accuracy

Profiling adds overhead to your training job, so measured timings may be slightly higher than normal operation. For accurate profiling results, disable other intensive operations like frequent checkpointing during the profiled step range. Choose your profiling step range carefully to capture representative training behavior while minimizing the performance impact on the overall job.
