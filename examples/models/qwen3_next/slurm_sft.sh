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
# Qwen3-Next 80B-A3B Full Supervised Fine-Tuning (SFT)
#
# Qwen3-Next 80B-A3B is a hybrid MoE language model with GDN linear attention
# layers. Supports multiple parallelism configs: each "TP,PP,EP,CP,SP" runs
# sequentially.
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set CONTAINER_IMAGE to your container path
#   3. Set PARALLELISM_CONFIGS (TP,PP,EP,CP,SP per entry; CP = context parallel size, 1 = disabled)
#   4. Submit: sbatch slurm_sft.sh
# ==============================================================================

#SBATCH --job-name=qwen3next-sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --output=logs/qwen3_next_sft_%j.out
#SBATCH --error=logs/qwen3_next_sft_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Base directory for container image and mounts (set if not already set, e.g. by launch_nemo.sh)
export WKDIR="${WKDIR:-}"

# Model and training configurations (use pretrain checkpoint or converted Megatron checkpoint)
# Use base dir (e.g. .../Qwen3-Next-80B-A3B-Instruct) with latest_checkpointed_iteration.txt, or Bridge dir with latest_train_state.pt
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/Qwen3-Next-80B-A3B-Instruct}
MODEL_NAME=qwen3_next_80b_a3b
DATASET_NAME=squad
SEQ_LENGTH=2048
TRAIN_ITERS=1000
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
EVAL_ITERS=32
EVAL_INTERVAL=50
LR_WARMUP_ITERS=50
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# Parallelism configs: "TP,PP,EP,CP,SP" per entry (TP*PP*EP must equal total GPUs)
PARALLELISM_CONFIGS=("1,2,8,1,True" "2,1,8,1,True")

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional; comma-separated for srun --container-mounts)
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

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Qwen3-Next 80B-A3B Full SFT Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Model: $MODEL_NAME"
echo "Parallelism configs: ${PARALLELISM_CONFIGS[*]}"
echo "======================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Require container image
if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

# Build srun command (shared across configs)
SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi
echo "SRUN base: $SRUN_CMD"
echo "======================================"

# Run each parallelism config in sequence
CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP CP SP <<< "$CONFIG"
    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    echo ""
    echo "======================================"
    echo "Config $CONFIG_INDEX/${#PARALLELISM_CONFIGS[@]}: TP=$TP, PP=$PP, EP=$EP, SP=$SP, CP=$CP"
    echo "======================================"

    # Build CLI overrides for this config (full SFT: no --peft_scheme)
    CLI_OVERRIDES=" \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        train.eval_iters=$EVAL_ITERS \
        train.eval_interval=$EVAL_INTERVAL \
        validation.eval_interval=$EVAL_INTERVAL \
        validation.eval_iters=$EVAL_ITERS \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_finetune_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_finetune_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP} \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP \
        model.expert_model_parallel_size=$EP \
        model.expert_tensor_parallel_size=1 \
        model.sequence_parallel=$SP \
        model.context_parallel_size=$CP \
        model.calculate_per_token_loss=True \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        dataset.seq_length=$SEQ_LENGTH \
        model.seq_length=$SEQ_LENGTH
    "

    CMD="uv run --no-sync python /opt/Megatron-Bridge/scripts/training/run_recipe.py"
    CMD="$CMD --recipe ${MODEL_NAME}_finetune_config"
    CMD="$CMD --peft_scheme none"
    # Collapse newlines so bash -c receives a single command
    CMD="$CMD $(echo "$CLI_OVERRIDES" | tr '\n' ' ' | sed 's/  \+/ /g')"

    echo "Executing command..."
    echo $CMD
    echo "======================================"

    $SRUN_CMD bash -c "$CMD"
    RUN_EXIT=$?
    if [ $RUN_EXIT -ne 0 ]; then
        echo "ERROR: Config TP=$TP, PP=$PP, EP=$EP, SP=$SP, CP=$CP failed with exit code $RUN_EXIT"
        exit $RUN_EXIT
    fi
done

echo "======================================"
echo "Job completed (all ${#PARALLELISM_CONFIGS[@]} configs)"
echo "======================================"
