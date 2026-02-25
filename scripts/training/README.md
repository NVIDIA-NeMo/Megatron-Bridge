# Training Scripts

Generic launcher and training scripts that work with any GPT-based model family (e.g. Deepseek, Llama, Gemma, Qwen, GPT, etc.).

## Overview

These scripts provide a generic interface for training GPT-based models in Megatron Bridge:

- `run_recipe.py` - Generic pretraining/finetuning for GPT- and Mamba-based models.
- `launch_with_nemo_run.py` - NeMo-Run launcher (local or Slurm)
- `launch_with_sbatch.sh` - Direct sbatch launcher

All scripts dynamically import recipes from `megatron.bridge.recipes`, apply user-provided overrides to the configuration, then begin training.

## Quick Start

For the end-to-end overview of how recipes are structured, overridden, and launched with either `torchrun` or NeMo-Run, see the official [Using Recipes guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html).

### Pretrain

```bash
torchrun --nproc_per_node=8 run_recipe.py --recipe llama32_1b_pretrain_config
```

### Finetune

```bash
torchrun --nproc_per_node=8 run_recipe.py --recipe llama32_1b_finetune_config
```

## Usage with Different Models

Same scripts work across all model families:

```bash
# Llama
torchrun --nproc_per_node=8 run_recipe.py --recipe llama32_1b_pretrain_config

# Gemma
torchrun --nproc_per_node=8 run_recipe.py --recipe gemma3_1b_pretrain_config

# Qwen
torchrun --nproc_per_node=8 run_recipe.py --recipe qwen3_8b_pretrain_config

# GPT
torchrun --nproc_per_node=8 run_recipe.py --recipe gpt_126m_pretrain_config
```

## CLI Overrides

Override any config field using dot notation:

```bash
torchrun --nproc_per_node=8 run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    train.train_iters=5000 \
    optimizer.lr=0.0002 \
    model.tensor_model_parallel_size=2
```

The first part before the dot specifies which ConfigContainer subconfig to override (e.g., `train`, `model`, `optimizer`), and the part after specifies the field.

Configuration priority:
1. CLI overrides (highest)
2. Recipe defaults (lowest)

Mode is inferred from the recipe name. If your recipe name doesn't include
`pretrain` or `finetune`, pass `--mode` explicitly.

## Step Function Selection

Use `--step_func` to control the step function used during training. Available options:

- `gpt_step` - Text-only models (default)
- `vlm_step` - Vision-language models
- `llava_step` - LLaVA models

```bash
torchrun --nproc_per_node=8 run_recipe.py \
    --recipe qwen25_vl_pretrain_config \
    --step_func vlm_step
```

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
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --devices 2 \
    --dry-run \
    train.train_iters=10
```

This uses `LocalExecutor` with torchrun for single-node testing. Include `--dry-run` to confirm the composed nemo-run command before actually launching it.

#### Launch on Slurm

Once tested, scale to Slurm by removing `--local` and adding Slurm parameters:

```bash
# From the cluster (LocalTunnel)
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account

# From your local machine (SSHTunnel)
python launch_with_nemo_run.py \
    --script run_recipe.py \
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
    --script run_recipe.py \
    --recipe qwen3_8b_pretrain_config \
    --nodes 4 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --mount /data:/data
```

> **Note:** PatternPackager only includes `scripts/training/*.py`. Local changes in
> `src/megatron/bridge/` stay on your workstation unless you mount the repo into
> the container.

```bash
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --mount /path/to/your/Megatron-Bridge:/opt/Megatron-Bridge \
    train.train_iters=10
```

Mounting onto `/opt/Megatron-Bridge` shadows the container's built-in source so
your edited `src/megatron/bridge/` files are used while packaged scripts still
run from the container workspace.

For git-based packaging:

```bash
python launch_with_nemo_run.py \
    --script run_recipe.py \
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
    --script run_recipe.py \
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
TRAINING_SCRIPT="run_recipe.py"

# Recipe name
RECIPE="llama32_1b_pretrain_config"

# Step function (controls the step function: gpt_step, vlm_step, or llava_step)
STEP_TYPE="gpt_step"

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
- Passes recipe and CLI override arguments to the training script
- Handles container execution (if specified)
- Applies container mounts

## Recipe Arguments

Generic scripts call recipes with no arguments passed to the recipe function.

All customization happens through CLI overrides after the config is built.

If you need to pass arguments to the recipe constructor itself (e.g., custom parallelism at recipe build time), use model-specific examples or create a custom script.
