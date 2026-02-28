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
# GPT-OSS 20B Pretraining
#
# GPT-OSS 20B is an MoE language model. Supports multiple parallelism configs:
# each "TP,PP,EP,CP,SP" runs sequentially.
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set CONTAINER_IMAGE to your container path
#   3. Set PARALLELISM_CONFIGS (TP,PP,EP,CP,SP per entry; CP = context parallel size, 1 = disabled)
#   4. Submit: sbatch slurm_pretrain.sh
# ==============================================================================

#SBATCH --job-name=gpt-oss-pretrain
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --account=my_account
#SBATCH --output=logs/gpt_oss_pretrain_%j.out
#SBATCH --error=logs/gpt_oss_pretrain_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Base directory for container image and mounts (set if not already set, e.g. by launch_nemo.sh)
export WKDIR="${WKDIR:-}"

# Model and training configurations
MODEL_NAME=gpt_oss_20b
DATASET_NAME=dclm  # set to "mock" for mock data; "dclm" uses DCLM when DCLM_DATA_DIR/DCLM_CACHE are set below
SEQ_LENGTH=4096

# When DATASET_NAME=dclm, set DCLM_DATA_DIR and DCLM_CACHE so the recipe uses DCLM; leave unset for mock
if [ "$DATASET_NAME" = "dclm" ]; then
    # export DCLM_DATA_DIR="/path/to/dclm/preprocessed"
    # export DCLM_CACHE="/path/to/cache"
    :
else
    unset DCLM_DATA_DIR
    unset DCLM_CACHE
fi

TRAIN_ITERS=1000
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
EVAL_ITERS=10
LR_WARMUP_ITERS=50
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# Parallelism configs: "TP,PP,EP,CP,SP" per entry (max(TP*CP, EP)*PP must be divisible by the total number of GPUs)
PARALLELISM_CONFIGS=("2,4,4,1,True" "4,2,4,1,True" "2,4,4,2,True")

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
echo "GPT-OSS 20B Pretraining Job"
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

# If using DCLM, pass dataset config via CLI overrides
DCLM_DATASET_OVERRIDES=""
if [ -n "${DCLM_DATA_DIR:-}" ] && [ -n "${DCLM_CACHE:-}" ]; then
    BLEND_PATHS=""
    for i in $(seq 1 10); do
        pad=$(printf "%02d" $i)
        PREFIX="${DCLM_DATA_DIR}/dclm_01_${pad}_text_document"
        if [ -f "${PREFIX}.bin" ]; then
            BLEND_PATHS="${BLEND_PATHS}\"${PREFIX}\","
        fi
    done
    BLEND_PATHS="${BLEND_PATHS%,}"
    
    if [ -n "$BLEND_PATHS" ]; then
        DCLM_DATASET_OVERRIDES="dataset.blend=[[${BLEND_PATHS}],null] dataset.split='\"9999,8,2\"' dataset.path_to_cache=${DCLM_CACHE}"
    else
        echo "WARNING: No DCLM data found in ${DCLM_DATA_DIR}!"
    fi
fi

# Run each parallelism config in sequence
export CUDA_DEVICE_MAX_CONNECTIONS=1
CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    OLD_IFS=$IFS
    IFS=',' read -r TP PP EP CP SP <<< "$CONFIG"
    IFS=$OLD_IFS

    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    
    echo ""
    echo "======================================"
    echo "Config $CONFIG_INDEX/${#PARALLELISM_CONFIGS[@]}: TP=$TP, PP=$PP, EP=$EP, SP=$SP, CP=$CP"
    echo "======================================"

    # Build CLI overrides for this config
    CLI_OVERRIDES=" \
        model.seq_length=$SEQ_LENGTH \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        train.eval_iters=$EVAL_ITERS \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_pretrain_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_pretrain_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP} \
        dataset.sequence_length=$SEQ_LENGTH \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP \
        model.expert_model_parallel_size=$EP \
        model.sequence_parallel=$SP \
        model.context_parallel_size=$CP \
    "
    if [ -n "$DCLM_DATASET_OVERRIDES" ]; then
        CLI_OVERRIDES="$CLI_OVERRIDES $DCLM_DATASET_OVERRIDES"
    fi

    CMD="uv run --no-sync python /opt/Megatron-Bridge/scripts/training/run_recipe.py"
    CMD="$CMD --recipe ${MODEL_NAME}_pretrain_config"
    CMD="$CMD $CLI_OVERRIDES"

    echo "Executing command..."
    echo "$CMD"
    echo "======================================"

    $SRUN_CMD bash -c "$CMD"
    RUN_EXIT=$?
    if [ $RUN_EXIT -ne 0 ]; then
        echo "ERROR: Config TP=$TP, PP=$PP, EP=$EP, SP=$SP, CP=$CP failed with exit code $RUN_EXIT"
        continue
    fi
done

echo "======================================"
echo "Job completed (all ${#PARALLELISM_CONFIGS[@]} configs)"
echo "======================================"
