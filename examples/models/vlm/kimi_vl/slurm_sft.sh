#!/bin/bash
set -euo pipefail
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
# Kimi-K2.5-VL Supervised Fine-Tuning (SFT)
#
# Recommended parallelism for the full model: TP=2, PP=16, EP=32 (64 GPUs, 8 nodes)
# For toy model validation, see TOY_MODEL_VALIDATION.md.
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set PRETRAINED_CHECKPOINT to your local model path
#   3. Set CONTAINER_IMAGE or adapt for bare metal
#   4. Submit: sbatch slurm_sft.sh
# ==============================================================================

#SBATCH --job-name=kimi-vl-sft
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=logs/kimi_vl_sft_%j.out
#SBATCH --error=logs/kimi_vl_sft_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}

PRETRAINED_CHECKPOINT=${WORKSPACE}/models/Kimi-K2.5
RECIPE=kimi_k25_vl_sft_config
DATASET_NAME=cord_v2
SEQ_LENGTH=2048
TRAIN_ITERS=5000
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=1
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-kimi

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional, space-separated)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /workspace:/workspace"

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# export UV_CACHE_DIR="/path/to/shared/uv_cache"
# export HF_HOME="/path/to/shared/HF_HOME"
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"
export WANDB_MODE=disabled

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Kimi-K2.5-VL SFT Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "Recipe: $RECIPE"
echo "Checkpoint: $PRETRAINED_CHECKPOINT"
echo "======================================"

# Create logs directory
mkdir -p logs

CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    model.seq_length=$SEQ_LENGTH \
    model.freeze_vision_model=true \
    model.freeze_vision_projection=true \
    model.calculate_per_token_loss=true \
    model.cross_entropy_loss_fusion=false \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=${WORKSPACE}/results/${RECIPE}_sft \
    dataset.maker_name=make_${DATASET_NAME}_dataset \
    dataset.seq_length=$SEQ_LENGTH \
    ddp.average_in_collective=false \
    logger.log_interval=$LOG_INTERVAL \
    logger.log_throughput=true \
    logger.log_params_norm=true \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${RECIPE}_${DATASET_NAME}_sft"

# For multinode runs, pass --hf_path with a local model directory
# for more reliable config loading, e.g.:
#   --hf_path ${WORKSPACE}/models/Kimi-K2.5
CMD="uv run --no-sync python scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func vlm_step \
    --hf_path moonshotai/Kimi-K2.5 \
    $CLI_OVERRIDES"

echo "Executing command..."
echo "======================================"

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"

if [ -n "$CONTAINER_MOUNTS" ]; then
    for mount in $CONTAINER_MOUNTS; do
        SRUN_CMD="$SRUN_CMD --container-mounts=$mount"
    done
fi

$SRUN_CMD bash -c "$CMD"

echo "======================================"
echo "Job completed"
echo "======================================"
