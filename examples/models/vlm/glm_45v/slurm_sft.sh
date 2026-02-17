#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# ==============================================================================
# GLM-4.5V Full Supervised Fine-Tuning (SFT)
#
# GLM-4.5V is a large MoE model (106B parameters)
# Recommended: TP=1, PP=8, EP=16 for full SFT (512 GPUs, 64 nodes)
# For smaller setups, use LoRA/DoRA instead (see slurm_peft.sh)
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set CONTAINER_IMAGE to your container path
#   3. Submit: sbatch slurm_sft.sh
# ==============================================================================

#SBATCH --job-name=glm45v-sft
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=logs/glm45v_sft_%j.out
#SBATCH --error=logs/glm45v_sft_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Model and training configurations
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/GLM-4.5V
MODEL_NAME=glm_45v
DATASET_NAME=cord_v2
SEQ_LENGTH=8192
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
EVAL_ITERS=10
LR=0.000005
MIN_LR=0.0000005
LR_WARMUP_ITERS=10
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# Parallelism configuration
TP=1
PP=8
EP=16

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional, space-separated)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /workspace:/workspace"

# ==============================================================================
# Environment Setup
# ==============================================================================

# NCCL optimizations for large-scale training
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# UV cache on shared filesystem (recommended for multi-node setups)
# Pre-sync once before submitting jobs: UV_CACHE_DIR=/path/to/cache uv sync
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# HuggingFace cache directory (recommended for shared filesystem)
# export HF_HOME="/path/to/shared/HF_HOME"

# Authentication tokens (set these for your environment)
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"
# or disable wandb logging
# export WANDB_MODE=disabled

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "GLM-4.5V Full SFT Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Model: $MODEL_NAME"
echo "Parallelism: TP=$TP, PP=$PP, EP=$EP"
echo "======================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Build CLI overrides
CLI_OVERRIDES="
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    train.eval_iters=$EVAL_ITERS \
    optimizer.lr=$LR \
    optimizer.min_lr=$MIN_LR \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_sft_tp${TP}_pp${PP}_ep${EP} \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_sft_tp${TP}_pp${PP}_ep${EP} \
    dataset.maker_name=make_${DATASET_NAME}_dataset \
    dataset.seq_length=$SEQ_LENGTH \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.expert_model_parallel_size=$EP
"

# Build command
# Only local rank 0 on each node runs uv sync, then all ranks run with --no-sync
CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 2; fi && "
CMD="$CMD uv run --no-sync python scripts/training/run_recipe.py"
CMD="$CMD --recipe ${MODEL_NAME}_finetune_config"
CMD="$CMD --step_func vlm_step"
CMD="$CMD $CLI_OVERRIDES"

echo "Executing command..."
echo "======================================"

# Require container image
if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

# Build srun command
SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"

# Add container mounts
if [ -n "$CONTAINER_MOUNTS" ]; then
    for mount in $CONTAINER_MOUNTS; do
        SRUN_CMD="$SRUN_CMD --container-mounts=$mount"
    done
fi

$SRUN_CMD bash -c "$CMD"

echo "======================================"
echo "Job completed"
echo "======================================"
