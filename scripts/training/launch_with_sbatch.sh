#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --job-name=megatron-bridge-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --exclusive

# ==============================================================================
# Direct Slurm Launch with sbatch (Alternative to NeMo-Run)
#
# This script demonstrates how to launch generic training scripts directly
# using sbatch without NeMo-Run. This is useful for traditional HPC workflows.
#
# Usage:
#   1. Modify the #SBATCH directives above for your cluster
#   2. Set the configuration variables below
#   3. Submit: sbatch launch_with_sbatch.sh
#
# For NeMo-Run based launching (recommended for remote management), see
# launch_with_nemo_run.py
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Modify these for your setup
# ==============================================================================

# Training script to run
TRAINING_SCRIPT="run_recipe.py"

# Full recipe function name. Use SOURCE="auto" for library recipes, or
# SOURCE="perf_recipes" when RECIPE is a flat performance recipe.
RECIPE="llama32_1b_pretrain_config"
SOURCE="auto"
# Examples:
# RECIPE="llama32_1b_sft_config"; SOURCE="auto"
# RECIPE="llama3_8b_pretrain_8gpu_h100_bf16_config"; SOURCE="perf_recipes"

# Optional selector mode. Leave RECIPE empty to use these fields instead.
MODEL_RECIPE_NAME="qwen3_30b_a3b"
TASK="pretrain"
NUM_GPUS=16
GPU="h100"
COMPUTE_DTYPE="bf16"

# Runtime defaults. Leave DATA empty to preserve the dataset owned by the recipe.
DOMAIN="llm"
DATA=""

# Optional: CLI overrides (Hydra-style dot notation)
CLI_OVERRIDES=()
# CLI_OVERRIDES=("train.train_iters=1000" "train.global_batch_size=512" "optimizer.lr=0.0002")

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional, space-separated)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /model:/model"

# ==============================================================================
# Environment Setup
# ==============================================================================

# Recipes provide process-environment defaults. Export values here only when
# intentionally overriding a recipe for this deployment.

# Authentication tokens (uncomment and set your tokens)
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"

# Optional: Uncomment if needed
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Megatron Bridge Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Script: $TRAINING_SCRIPT"
echo "Recipe: ${RECIPE:-selector:$MODEL_RECIPE_NAME}"
echo "Source: $SOURCE"
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: Set"
fi
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY: Set"
fi
echo "======================================"

# Determine script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/${TRAINING_SCRIPT}"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found: $SCRIPT_PATH"
    exit 1
fi

# Build rank-local arguments. Slurm launches one Python process per allocated
# task; Bridge derives rank and rendezvous settings from the Slurm environment.
TRAIN_ARGS=()
if [ -n "$RECIPE" ]; then
    TRAIN_ARGS+=(--recipe "$RECIPE" --source "$SOURCE")
else
    TRAIN_ARGS+=(
        --source "$SOURCE"
        --model "$MODEL_RECIPE_NAME"
        --task "$TASK"
        --gpus "$NUM_GPUS"
        --gpu "$GPU"
        --dtype "$COMPUTE_DTYPE"
    )
fi
TRAIN_ARGS+=(--domain "$DOMAIN")
if [ -n "$DATA" ]; then
    TRAIN_ARGS+=(--data "$DATA")
fi

printf "Executing: uv run python %q " "$SCRIPT_PATH"
printf "%q " "${TRAIN_ARGS[@]}" "${CLI_OVERRIDES[@]}"
printf "\n"
echo "======================================"

# Require container image
if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please use a valid container image."
    exit 1
fi

# Build srun command (always containerized)
SRUN_ARGS=(srun --mpi=pmix "--container-image=$CONTAINER_IMAGE")

# Add container mounts
if [ -n "$CONTAINER_MOUNTS" ]; then
    for mount in $CONTAINER_MOUNTS; do
        SRUN_ARGS+=("--container-mounts=$mount")
    done
fi

"${SRUN_ARGS[@]}" uv run python "$SCRIPT_PATH" "${TRAIN_ARGS[@]}" "${CLI_OVERRIDES[@]}"

echo "======================================"
echo "Job completed"
echo "======================================"
