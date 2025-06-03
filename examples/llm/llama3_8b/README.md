# Llama3 8B Pretraining with NeMo-LM

This directory provides multiple execution paths for pretraining Llama3 8B models using NeMo-LM. Choose the approach that best fits your workflow and infrastructure.

## Quick Start

```bash
# Simple local run with default settings
python pretrain_llama3_8b.py

# With custom overrides
python pretrain_llama3_8b.py model_config.hidden_size=1024 train_config.train_iters=1000

# With YAML config
python pretrain_llama3_8b.py --config-file my_config.yaml
```

## Execution Paths Overview

| Method | Best For | Key Benefits |
|--------|----------|--------------|
| **[Plain Python Script](#plain-python-script)** | Development, debugging, custom launchers | Direct control, easy debugging |
| **[NeMo Run + Script](#nemo-run--script)** | Production runs, experiment tracking | Built-in launchers, experiment management |
| **[NeMo Run + Partial](#nemo-run--partial)** | Complex configs, config composition | Advanced config features, modularity |

---

## Plain Python Script

**File:** `pretrain_llama3_8b.py`

Direct execution of the pretraining script with standard Python tooling.

### Local Development

```bash
# Basic run
python pretrain_llama3_8b.py

# Enable debug logging
python pretrain_llama3_8b.py --debug

# With configuration overrides
python pretrain_llama3_8b.py \
    model_config.hidden_size=2048 \
    train_config.train_iters=5000 \
    optimizer_config.lr=1e-4
```

### Multi-GPU with torchrun

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 pretrain_llama3_8b.py \
    model_config.tensor_model_parallel_size=2 \
    model_config.pipeline_model_parallel_size=2

# Multi-node setup
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_llama3_8b.py \
        model_config.tensor_model_parallel_size=4 \
        model_config.pipeline_model_parallel_size=2
```

### SLURM Integration

```bash
#!/bin/bash
#SBATCH --job-name=llama3_8b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00

srun python pretrain_llama3_8b.py \
    model_config.tensor_model_parallel_size=8 \
    model_config.pipeline_model_parallel_size=4 \
    train_config.global_batch_size=512
```

### Configuration Options

#### YAML Configuration Files

Create `conf/my_config.yaml`:
```yaml
model_config:
  hidden_size: 4096
  num_layers: 32
  
optimizer_config:
  lr: 1e-4
  weight_decay: 0.1
  
scheduler_config:
  lr_decay_style: cosine
  lr_warmup_iters: 2000

train_config:
  train_iters: 10000
  global_batch_size: 512
  micro_batch_size: 2
  eval_interval: 1000

logger_config:
  log_interval: 100
  tensorboard_dir: ./tensorboard_logs

checkpoint_config:
  save_interval: 1000
  save: ./checkpoints
```

Run with:
```bash
python pretrain_llama3_8b.py --config-file conf/my_config.yaml
```

#### Hydra-Style CLI Overrides

The script supports powerful Hydra override syntax:

```bash
# Basic assignment
python pretrain_llama3_8b.py model_config.hidden_size=4096

# Addition (new parameters)
python pretrain_llama3_8b.py +logger_config.custom_param=value

# Deletion
python pretrain_llama3_8b.py ~model_config.unwanted_param

# List operations
python pretrain_llama3_8b.py +profiling_config.profile_ranks=[0,1,2,3]

# Nested operations
python pretrain_llama3_8b.py scheduler_config.lr_warmup_iters=1000

# Complex combinations
python pretrain_llama3_8b.py \
    --config-file base_config.yaml \
    model_config.hidden_size=4096 \
    +profiling_config.use_pytorch_profiler=true \
    train_config.train_iters=5000
```

---

## NeMo Run + Script

**Integration:** Uses NeMo Run framework to execute the plain Python script

Provides built-in launcher support and experiment management while using the same script.

### Local Executor

```python
import nemo_run as run

# Create partial for the script
script_partial = run.Script(
    path="examples/llm/llama3_8b/pretrain_llama3_8b.py",
    args=[
        "model_config.hidden_size=4096",
        "train_config.train_iters=1000",
        "--config-file", "conf/my_config.yaml"
    ]
)

# Run locally
with run.Executor(folder="./results") as executor:
    run_context = executor.submit(script_partial)
    print(f"Job submitted: {run_context}")
```

### SLURM Executor

```python
import nemo_run as run

# Configure SLURM executor
slurm_executor = run.SlurmExecutor(
    folder="./slurm_results",
    account="your_account",
    partition="gpu",
    nodes=4,
    ntasks_per_node=8,
    gpus_per_task=1,
    time="24:00:00",
)

# Submit job
script_partial = run.Script(
    path="examples/llm/llama3_8b/pretrain_llama3_8b.py",
    args=[
        "model_config.tensor_model_parallel_size=8",
        "model_config.pipeline_model_parallel_size=4",
        "train_config.global_batch_size=512"
    ]
)

with slurm_executor:
    run_context = slurm_executor.submit(script_partial)
    print(f"SLURM job submitted: {run_context.job_id}")
```

### Docker/Container Executor

```python
import nemo_run as run

# Use containerized execution
container_executor = run.DockerExecutor(
    folder="./container_results",
    container_image="nemo:24.09",
    container_mounts=[
        ("/data", "/workspace/data"),
        ("/results", "/workspace/results")
    ]
)

script_partial = run.Script(
    path="examples/llm/llama3_8b/pretrain_llama3_8b.py",
    args=["--config-file", "/workspace/conf/production.yaml"]
)

with container_executor:
    run_context = container_executor.submit(script_partial)
```

---

## NeMo Run + Partial

**Integration:** Uses NeMo's run.Partial system for advanced configuration management

Leverages the recipe's `pretrain_config()` function directly with NeMo Run's configuration system.

### Basic Partial Usage

```python
import nemo_run as run
from nemo_lm.recipes.llm.llama3_8b import pretrain_config

# Get the base configuration
config = pretrain_config()

# Create a partial for pretraining
pretrain_partial = run.Partial(
    func=run.Config(nemo_lm.training.pretrain.megatron_pretrain),
    config=config,
    forward_step_func=run.Config(nemo_lm.models.utils.forward_step)
)

# Execute locally
with run.Executor(folder="./partial_results") as executor:
    run_context = executor.submit(pretrain_partial)
```

### Advanced Configuration Composition

```python
import nemo_run as run
from nemo_lm.recipes.llm.llama3_8b import pretrain_config, get_llama3_8b_model_config

# Compose configuration from parts
model_config = get_llama3_8b_model_config()
base_config = pretrain_config()

# Override specific components
model_config.hidden_size = 4096
model_config.num_layers = 32
base_config.model_config = model_config

# Override training settings
base_config.train_config.train_iters = 10000
base_config.optimizer_config.lr = 1e-4

# Create partial with custom config
pretrain_partial = run.Partial(
    func=run.Config(nemo_lm.training.pretrain.megatron_pretrain),
    config=base_config,
    forward_step_func=run.Config(nemo_lm.models.utils.forward_step)
)
```

### Config Templates and Inheritance

```python
import nemo_run as run

# Create reusable config templates
@run.autoconvert
def create_large_model_config():
    config = pretrain_config()
    config.model_config.hidden_size = 8192
    config.model_config.num_layers = 48
    config.optimizer_config.clip_grad = 1.0
    return config

@run.autoconvert  
def create_debug_config():
    config = pretrain_config()
    config.train_config.train_iters = 100
    config.train_config.eval_interval = 10
    config.train_config.micro_batch_size = 1
    return config

# Use templates
large_config = create_large_model_config()
debug_config = create_debug_config()

# Submit multiple experiments
configs = [
    ("large_model", large_config),
    ("debug_run", debug_config)
]

with run.Executor(folder="./experiments") as executor:
    for name, config in configs:
        partial = run.Partial(
            func=run.Config(nemo_lm.training.pretrain.megatron_pretrain),
            config=config,
            forward_step_func=run.Config(nemo_lm.models.utils.forward_step)
        )
        executor.submit(partial, name=name)
```

### Multi-Executor Experiments

```python
import nemo_run as run

# Define multiple executors for different experiments
executors = {
    "local": run.LocalExecutor(folder="./local_runs"),
    "slurm": run.SlurmExecutor(
        folder="./slurm_runs",
        account="research",
        partition="gpu",
        nodes=2,
        time="12:00:00"
    ),
    "cluster": run.SlurmExecutor(
        folder="./cluster_runs", 
        account="production",
        partition="large_gpu",
        nodes=8,
        time="48:00:00"
    )
}

# Submit to appropriate executor based on experiment size
experiments = [
    ("small_test", "local", {"train_config.train_iters": 100}),
    ("medium_run", "slurm", {"train_config.train_iters": 1000}),
    ("full_train", "cluster", {"train_config.train_iters": 100000})
]

for name, executor_name, overrides in experiments:
    config = pretrain_config()
    
    # Apply overrides
    for key, value in overrides.items():
        parts = key.split('.')
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    
    # Submit to appropriate executor
    partial = run.Partial(
        func=run.Config(nemo_lm.training.pretrain.megatron_pretrain),
        config=config,
        forward_step_func=run.Config(nemo_lm.models.utils.forward_step)
    )
    
    with executors[executor_name]:
        executors[executor_name].submit(partial, name=name)
```

---

## Configuration Management

### Hierarchical Configuration

All execution paths support hierarchical configuration with the following precedence (highest to lowest):

1. **CLI overrides** (e.g., `model_config.hidden_size=4096`)
2. **YAML config file** (e.g., `--config-file my_config.yaml`)  
3. **Recipe defaults** (from `pretrain_config()`)

### Environment Variables

Set environment variables for common configurations:

```bash
export NEMO_LOG_LEVEL=DEBUG
export NEMO_CACHE_DIR=/shared/cache
export CUDA_VISIBLE_DEVICES=0,1,2,3

python pretrain_llama3_8b.py
```

### Configuration Structure

The configuration follows the `ConfigContainer` structure defined in `config.py`:

- **`model_config`** - Model architecture settings (GPTConfig)
- **`train_config`** - Training loop settings (TrainingConfig) 
- **`optimizer_config`** - Optimizer configuration (OptimizerConfig)
- **`scheduler_config`** - Learning rate scheduler (SchedulerConfig)
- **`dataset_config`** - Data loading configuration (GPTDatasetConfig)
- **`logger_config`** - Logging and monitoring (LoggerConfig)
- **`tokenizer_config`** - Tokenizer settings (TokenizerConfig)
- **`checkpoint_config`** - Checkpointing configuration (CheckpointConfig)
- **`dist_config`** - Distributed training setup (DistributedInitConfig)
- **`profiling_config`** - Performance profiling (ProfilingConfig)

### Configuration Validation

All paths include automatic validation:
- **Type checking** for configuration parameters
- **Value range validation** for numeric parameters
- **Dependency checking** between related parameters
- **Resource availability** validation

---

## Examples Gallery

### Development Workflow

```bash
# 1. Quick smoke test
python pretrain_llama3_8b.py train_config.train_iters=10 train_config.eval_interval=5

# 2. Debug with small model
python pretrain_llama3_8b.py --debug \
    model_config.hidden_size=512 \
    model_config.num_layers=2 \
    train_config.train_iters=100

# 3. Profile performance
python pretrain_llama3_8b.py \
    +profiling_config.use_pytorch_profiler=true \
    train_config.train_iters=50
```

### Production Training

```bash
# Full-scale multi-node training
torchrun --nnodes=8 --nproc_per_node=8 pretrain_llama3_8b.py \
    --config-file conf/production.yaml \
    model_config.tensor_model_parallel_size=8 \
    model_config.pipeline_model_parallel_size=4 \
    train_config.global_batch_size=2048 \
    model_config.hidden_size=4096 \
    train_config.train_iters=500000
```

### Hyperparameter Sweeps

```python
# Using NeMo Run for systematic sweeps
import nemo_run as run

sweep_params = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "hidden_size": [2048, 4096, 8192],
    "num_layers": [16, 24, 32]
}

with run.Executor(folder="./sweep_results") as executor:
    for lr in sweep_params["learning_rate"]:
        for hidden_size in sweep_params["hidden_size"]:
            for num_layers in sweep_params["num_layers"]:
                
                name = f"lr_{lr}_hs_{hidden_size}_nl_{num_layers}"
                
                script_partial = run.Script(
                    path="examples/llm/llama3_8b/pretrain_llama3_8b.py",
                    args=[
                        f"optimizer_config.lr={lr}",
                        f"model_config.hidden_size={hidden_size}",
                        f"model_config.num_layers={num_layers}"
                    ]
                )
                
                executor.submit(script_partial, name=name)
```

---

## Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce batch size
python pretrain_llama3_8b.py train_config.micro_batch_size=1

# Enable gradient checkpointing  
python pretrain_llama3_8b.py +model_config.activations_checkpoint_granularity=full
```

**Slow Training:**
```bash
# Check data loading
python pretrain_llama3_8b.py +profiling_config.use_pytorch_profiler=true

# Optimize mixed precision
python pretrain_llama3_8b.py model_config.fp16=true
```

**Configuration Errors:**
```bash
# Validate config without training
python pretrain_llama3_8b.py --debug train_config.train_iters=0
```

### Debug Modes

```bash
# Full debug logging
python pretrain_llama3_8b.py --debug

# NeMo debug mode
NEMO_DEBUG=1 python pretrain_llama3_8b.py

# PyTorch debug mode  
TORCH_SHOW_CPP_STACKTRACES=1 python pretrain_llama3_8b.py
```

---

## Performance Tips

1. **Use appropriate precision:** `model_config.fp16=true` or `model_config.bf16=true` for modern GPUs
2. **Optimize batch sizes:** Balance `train_config.micro_batch_size` and `train_config.global_batch_size`
3. **Enable gradient checkpointing:** `model_config.activations_checkpoint_granularity=full`
4. **Profile regularly:** Use `profiling_config.use_pytorch_profiler=true` to identify bottlenecks
5. **Check GPU utilization:** Monitor with `nvidia-smi` or `torch.profiler`

---

## Next Steps

- 📚 **[Configuration Reference](conf/README.md)** - Detailed parameter documentation
- 🔬 **[Experiment Tracking](../../../docs/experiments.md)** - Setting up monitoring and logging  
- 🚀 **[Advanced Features](../../../docs/advanced.md)** - Custom models, optimizers, and strategies
- 🔧 **[Troubleshooting Guide](../../../docs/troubleshooting.md)** - Common issues and solutions 