# Training Scripts

Generic launcher and training scripts that work with any GPT-based model family (e.g. Deepseek, Llama, Gemma, Qwen, GPT, etc.).

## Overview

These scripts provide a generic interface for training GPT-based models in Megatron Bridge:

- `pretrain_gpt.py` - Generic pretraining for GPT-based models.
- `finetune_gpt.py` - Generic finetuning for GPT-based models.
- `launch_with_nemo_run.py` - NeMo-Run launcher (local or Slurm)
- `launch_with_sbatch.sh` - Direct sbatch launcher
- `conf/template_overrides.yaml` - Template for YAML overrides

All scripts dynamically import recipes from `megatron.bridge.recipes`, apply user-provided overrides to the configuration, then begin training.

## Quick Start

### Pretrain

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py --recipe llama32_1b_pretrain_config
```

### Finetune

```bash
torchrun --nproc_per_node=8 finetune_gpt.py --recipe llama32_1b_finetune_config
```

## Usage with Different Models

Same scripts work across all model families:

```bash
# Llama
torchrun --nproc_per_node=8 pretrain_gpt.py --recipe llama32_1b_pretrain_config

# Gemma
torchrun --nproc_per_node=8 pretrain_gpt.py --recipe gemma3_1b_pretrain_config

# Qwen
torchrun --nproc_per_node=8 pretrain_gpt.py --recipe qwen3_8b_pretrain_config

# GPT
torchrun --nproc_per_node=8 pretrain_gpt.py --recipe gpt_126m_pretrain_config
```

## Configuration with YAML

Use YAML files for complex configurations:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --recipe llama3_8b_pretrain_config \
    --config-file conf/my_config.yaml
```

See `conf/template_overrides.yaml` for a complete template showing all available sections.

YAML structure mirrors ConfigContainer:

```yaml
data:
  data_path: /path/to/dataset
  sequence_length: 4096

train:
  train_iters: 1000
  global_batch_size: 256

model:
  seq_length: 4096  # Must match data.sequence_length
  tensor_model_parallel_size: 2

optimizer:
  lr: 0.0003

checkpoint:
  save: ./checkpoints/my_model
  save_interval: 100

# For finetuning with LoRA (requires _target_ for instantiation)
peft:
  _target_: megatron.bridge.peft.lora.LoRA
  dim: 8
  alpha: 16
```

## CLI Overrides

Override any config field using dot notation:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    train.train_iters=5000 \
    optimizer.lr=0.0002 \
    model.tensor_model_parallel_size=2
```

The first part before the dot specifies which ConfigContainer subconfig to override (e.g., `train`, `model`, `optimizer`), and the part after specifies the field.

Configuration priority:
1. CLI overrides (highest)
2. YAML config file
3. Recipe defaults (lowest)

## Multi-Node and Distributed Training

### Option 1: NeMo-Run

Prerequisites:

```bash
pip install nemo-run
```

#### Test Locally First

Before launching on Slurm, test your configuration locally:

```bash
python launch_with_nemo_run.py \
    --local \
    --script pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    --devices 2 \
    train.train_iters=10
```

This uses `LocalExecutor` with torchrun for single-node testing.

#### Launch on Slurm

Once tested, scale to Slurm by removing `--local` and adding Slurm parameters:

```bash
# From the cluster (LocalTunnel)
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account

# From your local machine (SSHTunnel)
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --ssh-tunnel \
    --host my-cluster.example.com \
    --user myusername \
    --remote-job-dir /home/myusername/nemo-runs
```

#### With Containers

When using containers, scripts are automatically packaged using `PatternPackager`:

```bash
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe qwen3_8b_pretrain_config \
    --nodes 4 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --mount /data:/data
```

For git-based packaging:

```bash
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe llama3_8b_pretrain_config \
    --nodes 2 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --packager git
```

#### Fault-Tolerant Training

Use the fault-tolerant launcher for better resiliency:

```bash
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    --launcher ft \
    --nodes 2 \
    --partition gpu \
    --account my_account
```

### Option 2: Direct sbatch

For traditional HPC workflows without NeMo-Run, use the `launch_with_sbatch.sh` script.

Edit the configuration section in `launch_with_sbatch.sh`:

```bash
# Training script to run
TRAINING_SCRIPT="pretrain_gpt.py"

# Recipe name
RECIPE="llama32_1b_pretrain_config"

# Optional: YAML config file
CONFIG_FILE="conf/my_config.yaml"

# Optional: CLI overrides
CLI_OVERRIDES="train.train_iters=5000 optimizer.lr=0.0003"

# Optional: Container settings
CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS="/data:/data /model:/model"
```

Also configure the SBATCH directives at the top of the file:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --time=04:00:00
```

Then submit:

```bash
sbatch launch_with_sbatch.sh
```

The script automatically:
- Sets up multi-node torchrun with correct SLURM environment variables
- Passes recipe and config arguments to the training script
- Handles container execution (if specified)
- Applies container mounts

## Recipe Arguments

Generic scripts call recipes with **no arguments**: `recipe()`.

All customization happens through YAML and CLI overrides after the config is built. This covers 99% of use cases.

If you need to pass arguments to the recipe constructor itself (e.g., custom parallelism at recipe build time), use model-specific examples:
- See `examples/recipes/llama/` for Llama-specific examples
- See `examples/recipes/<family>/` for other model families
- Or create a custom script

## Available Recipes

To see all available recipes:

```python
from megatron.bridge.recipes import *
import megatron.bridge.recipes as recipes

# List all pretrain configs
pretrain_recipes = [x for x in dir(recipes) if 'pretrain_config' in x]
print(pretrain_recipes)

# List all finetune configs
finetune_recipes = [x for x in dir(recipes) if 'finetune_config' in x]
print(finetune_recipes)
```

Or check the recipe source code:
- `src/megatron/bridge/recipes/llama/`
- `src/megatron/bridge/recipes/gemma/`
- `src/megatron/bridge/recipes/qwen/`
- etc.

## Example Workflows

### Workflow 1: Test Locally, Then Scale to Slurm

```bash
# Step 1: Test locally
python launch_with_nemo_run.py \
    --local \
    --script pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    --devices 2 \
    train.train_iters=10

# Step 2: Scale to Slurm
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    train.train_iters=10000
```

### Workflow 2: Multi-Node with Container

```bash
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe qwen3_8b_pretrain_config \
    --nodes 4 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/nemo-container.sqsh \
    --mount /data:/data \
    train.train_iters=100000
```

**Important for containers:**
- By default, PatternPackager only packages `scripts/training/*.py` files
- Your local changes in `src/megatron/bridge/` are **NOT packaged**
- The container uses its installed version at `/opt/Megatron-Bridge`

**To use your local changes in a container:**

Mount your local repo over the container's installation path. This shadows the built-in version:

```bash
python launch_with_nemo_run.py \
    --script pretrain_gpt.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --mount /home/ansubramania/dev/Megatron-Bridge:/opt/Megatron-Bridge \
    train.train_iters=10
```

Notes:
- Mounting to `/opt/Megatron-Bridge` replaces the container's built-in version
- The launcher **automatically detects** when you mount `Megatron-Bridge` and switches to passthrough packager
- No need to manually specify `--packager none`

### Workflow 3: Finetune with Custom YAML

```bash
# Create config
cat > conf/my_finetune.yaml << EOF
checkpoint:
  pretrained_checkpoint: ./checkpoints/gemma3_1b

train:
  train_iters: 1000

data:
  data_path: /path/to/my/dataset.jsonl

peft:
  _target_: megatron.bridge.peft.lora.LoRA
  dim: 16
  alpha: 32
EOF

# Run finetuning
torchrun --nproc_per_node=2 finetune_gpt.py \
    --recipe gemma3_1b_finetune_config \
    --config-file conf/my_finetune.yaml
```
