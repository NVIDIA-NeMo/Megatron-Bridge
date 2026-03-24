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
# Qwen3-600M Long-Context (128K) Supervised Fine-Tuning (SFT)
#
# Parallelism: TP=1, CP=8 (minimum 8 GPUs)
# Sequence length: 131072 (128K tokens)
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set CONTAINER_IMAGE to your container path
#   3. Submit: sbatch sft.sh
# ==============================================================================

#SBATCH --job-name=qwen3-600m-128k-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --account=my_account
#SBATCH --output=logs/qwen3_600m_128k_sft_%j.out
#SBATCH --error=logs/qwen3_600m_128k_sft_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/qwen3-0.6b}
MODEL_NAME=qwen3_600m
RECIPE_NAME=qwen3_600m_sft_128k_config
DATASET_NAME=squad
SEQ_LENGTH=131072
TRAIN_ITERS=1000
GLOBAL_BATCH_SIZE=2
MICRO_BATCH_SIZE=1
EVAL_ITERS=32
EVAL_INTERVAL=30
LR_WARMUP_ITERS=50
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional; comma-separated for srun --container-mounts)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /workspace:/workspace"

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# UV cache on shared filesystem (recommended for multi-node setups)
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# HuggingFace cache directory (recommended for shared filesystem)
# export HF_HOME="/path/to/shared/HF_HOME"

# Authentication tokens (set these for your environment)
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Qwen3-600M 128K Long-Context SFT Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Model: $MODEL_NAME"
echo "Recipe: $RECIPE_NAME"
echo "======================================"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

CLI_OVERRIDES=" \
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    validation.eval_iters=$EVAL_ITERS \
    validation.eval_interval=$EVAL_INTERVAL \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_128k_sft \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_128k_sft \
"

CMD="uv run --no-sync python /opt/Megatron-Bridge/scripts/training/run_recipe.py"
CMD="$CMD --mode finetune"
CMD="$CMD --recipe ${RECIPE_NAME}"
CMD="$CMD --peft_scheme none"
CMD="$CMD $(echo "$CLI_OVERRIDES" | tr '\n' ' ' | sed 's/  \+/ /g')"

echo "Executing command..."
echo $CMD
echo "======================================"

$SRUN_CMD bash -c "$CMD"
RUN_EXIT=$?
if [ $RUN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code $RUN_EXIT"
    exit $RUN_EXIT
fi

echo "======================================"
echo "Job completed successfully"
echo "======================================"
