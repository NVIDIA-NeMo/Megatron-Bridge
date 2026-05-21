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

# ==============================================================================
# Qwen3.5-VL 35B-A3B Full SFT with Megatron FSDP
#
# Uses Megatron FSDP for memory-efficient training with AG/RS overlap.
# Requires fsdp_dtensor checkpoint format (convert offline with
# checkpoint_inspector.py convert-torch-dist-to-fsdp-dtensor).
#
# For standard 3D parallelism (no FSDP), use slurm_sft.sh instead.
#
# Usage:
#   sbatch slurm_sft_fsdp.sh
#   sbatch --nodes=4 slurm_sft_fsdp.sh
# ==============================================================================

#SBATCH --job-name=qwen35vl-sft-fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=qwen35vl_sft_fsdp_%j.out
#SBATCH --error=qwen35vl_sft_fsdp_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RECIPE="qwen35_vl_35b_a3b_fsdp_sft_config"
HF_MODEL_NAME="Qwen3.5-35B-A3B"

WORKSPACE=${WORKSPACE:-/workspace}

PRETRAINED_CHECKPOINT=${WORKSPACE}/models/Qwen/${HF_MODEL_NAME}-fsdp-dtensor
DATASET_NAME=cord_v2
SEQ_LENGTH=4096
TRAIN_ITERS=500
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=4  # tested on Blackwell GPUs; reduce for smaller VRAM
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

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
export NCCL_NVLS_ENABLE=1
export HTTPX_LOG_LEVEL=WARNING
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda,ignore::UserWarning:modelopt.torch"

# export UV_CACHE_DIR="/path/to/shared/uv_cache"
# export HF_HOME="/path/to/shared/HF_HOME"
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"
# export WANDB_MODE=disabled

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Qwen3.5-VL FSDP SFT Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "Model: $HF_MODEL_NAME"
echo "Recipe: $RECIPE"
echo "Checkpoint: $PRETRAINED_CHECKPOINT"
echo "======================================"

CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    checkpoint.ckpt_format=fsdp_dtensor \
    checkpoint.fully_parallel_load=true \
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=${WORKSPACE}/results/${RECIPE}_sft \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${RECIPE}_${DATASET_NAME}_sft \
    dataset.maker_name=make_${DATASET_NAME}_dataset \
    dataset.seq_length=$SEQ_LENGTH"

CMD="cd /opt/Megatron-Bridge && uv run --no-sync python scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func qwen3_vl_step \
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
