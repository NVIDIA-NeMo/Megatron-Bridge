# Training Scripts

Generic launcher and training scripts that work with any GPT-based model family (e.g. Deepseek, Llama, Gemma, Qwen, GPT, etc.).

## Overview

These scripts provide a generic interface for training models in Megatron Bridge:

- `run_recipe.py` - Generic pretraining/finetuning for recipes from `megatron.bridge.recipes` and flat performance recipes from `megatron.bridge.perf_recipes`.
- `launch_with_nemo_run.py` - NeMo-Run launcher (local or Slurm)
- `launch_with_sbatch.sh` - Direct sbatch launcher

All scripts dynamically import a recipe, apply user-provided overrides to the configuration, then begin training.

## Getting Started

For the end-to-end overview of how recipes are structured, overridden, and launched, see the official [Using Recipes guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html).
For preparing and selecting pretraining, text-only SFT, direct Hugging Face SFT, or Energon data, see the [data tutorial index](../../tutorials/data/README.md).

### 1. Dry-run a library recipe

```bash
uv run python scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --dry-run \
    --save-config /tmp/llama32_1b.yaml \
    --max-steps 10 \
    --data mock
```

`--recipe` first checks `megatron.bridge.recipes`, then falls back to
`megatron.bridge.perf_recipes` when `--source auto` is used.

### 2. Run the same library recipe with torch distributed

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 \
    scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --data mock \
    --max-steps 20 \
    --global-batch-size 8 \
    --micro-batch-size 1 \
    optimizer.lr=0.0003
```

### 3. Select a flat performance recipe

```bash
uv run python scripts/training/run_recipe.py \
    --source perf_recipes \
    --model llama3_8b \
    --task pretrain \
    --gpus 8 \
    --gpu h100 \
    --dtype bf16 \
    --dry-run \
    --save-config /tmp/llama3_perf.yaml
```

The selector resolves to
`llama3_8b_pretrain_8gpu_h100_bf16_config`. You can also pass the full flat
recipe name directly:

```bash
uv run python scripts/training/run_recipe.py \
    --recipe llama3_8b_pretrain_8gpu_h100_bf16_config \
    --source perf_recipes \
    --dry-run
```

### 4. Override datasets and config fields

Use easy flags for common changes and `key=value` for any `ConfigContainer`
field:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 \
    scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --dataset llm-pretrain-mock \
    --seq-length 512 \
    --tokenizer-type NullTokenizer \
    --vocab-size 32000 \
    --set train.train_iters=20 \
    model.hidden_size=256 \
    model.num_layers=2
```

## Usage with Different Models

Same scripts work across all model families:

```bash
# Llama
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_1gpu_h100_bf16_config

# Gemma
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe gemma3_1b_pretrain_1gpu_h100_bf16_config

# Qwen
uv run python -m torch.distributed.run --nproc_per_node=4 scripts/training/run_recipe.py \
    --recipe qwen3_8b_pretrain_4gpu_h100_bf16_config

# GPT
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe vanilla_gpt_pretrain_1gpu_h100_bf16_config
```

## Loading a HuggingFace Model Directly (`--hf_path`)

Some recipes accept `--hf_path` to initialize from a HuggingFace model ID (or local
path) without a separate offline checkpoint conversion step:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe vanilla_gpt_pretrain_1gpu_h100_bf16_config \
    --hf_path meta-llama/Llama-3.1-8B
```

`--hf_path` is optional and model-specific: `run_recipe.py` only forwards it to recipes
that accept an `hf_path` argument, and ignores it otherwise.

## CLI Overrides

Override any config field using dot notation:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_1gpu_h100_bf16_config \
    train.train_iters=5000 \
    optimizer.lr=0.0002 \
    model.tensor_model_parallel_size=2
```

The first part before the dot specifies which ConfigContainer subconfig to override (e.g., `train`, `model`, `optimizer`), and the part after specifies the field.

Configuration priority:
1. CLI overrides (highest)
2. Recipe defaults (lowest)

Mode is inferred from the recipe name. If your recipe name doesn't include
`pretrain`, `finetune`, `sft`, or `peft`, pass `--dataset` so the launcher can infer the mode from the dataset type.

## Step Function Selection

Use `--step_func` to control the step function used during training. Available options:

- `gpt_step` - Text-only models (default)
- `vlm_step` - Vision-language models
- `llava_step` - LLaVA models

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 scripts/training/run_recipe.py \
    --recipe qwen25_vl_7b_sft_2gpu_h100_bf16_config \
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
uv run python launch_with_nemo_run.py \
    --local \
    --recipe llama32_1b_pretrain_1gpu_h100_bf16_config \
    --gpus-per-node 1 \
    --dry-run \
    train.train_iters=10
```

This uses `LocalExecutor` with torchrun for single-node testing. Include `--dry-run` to confirm the composed nemo-run command before actually launching it.

#### Launch on Slurm

Once tested, scale to Slurm by removing `--local` and adding Slurm parameters:

```bash
# From the cluster (LocalTunnel)
uv run python launch_with_nemo_run.py \
    --recipe qwen3_8b_pretrain_4gpu_h100_bf16_config \
    --nodes 2 \
    --gpus-per-node 8 \
    --partition gpu \
    --account my_account

# From your local machine (SSHTunnel)
uv run python launch_with_nemo_run.py \
    --recipe llama3_8b_sft_2gpu_h100_bf16_config \
    --nodes 1 \
    --gpus-per-node 8 \
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
uv run python launch_with_nemo_run.py \
    --recipe qwen3_8b_pretrain_4gpu_h100_bf16_config \
    --nodes 1 \
    --gpus-per-node 4 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --mount /data:/data
```

> **Note:** PatternPackager includes Python files under `scripts/`. Local changes in
> `src/megatron/bridge/` stay on your workstation unless you mount the repo into
> the container.

```bash
uv run python launch_with_nemo_run.py \
    --recipe llama32_1b_pretrain_1gpu_h100_bf16_config \
    --nodes 2 \
    --gpus-per-node 8 \
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
uv run python launch_with_nemo_run.py \
    --recipe llama3_8b_pretrain_8gpu_h100_bf16_config \
    --source perf_recipes \
    --nodes 1 \
    --gpus-per-node 8 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --packager git
```

#### Fault-Tolerant Training

Use the fault-tolerant launcher for better resiliency:

```bash
uv run python launch_with_nemo_run.py \
    --recipe llama32_1b_pretrain_1gpu_h100_bf16_config \
    --launcher ft \
    --nodes 2 \
    --gpus-per-node 8 \
    --partition gpu \
    --account my_account
```

### Option 2: Direct sbatch

For traditional HPC workflows without NeMo-Run, use the `launch_with_sbatch.sh` script.

Edit the configuration section in `launch_with_sbatch.sh`:

```bash
# Training script to run
TRAINING_SCRIPT="run_recipe.py"

# Full recipe function name
RECIPE="llama32_1b_pretrain_config"
SOURCE="auto"

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

`run_recipe.py` calls recipes with no constructor arguments by default. It only
forwards optional constructor shortcuts (`--hf-path`, `--seq-length`,
`--packed-sequence`, `--peft-scheme`) when the selected recipe function accepts
them.

Most customization should happen after the config is built through easy flags
or `ConfigContainer` overrides such as `model.tensor_model_parallel_size=2`.
