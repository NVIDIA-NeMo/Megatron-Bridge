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
# GPT-OSS 20B Full Supervised Fine-Tuning (SFT)
#
# GPT-OSS 20B is an MoE language model. Supports multiple parallelism configs:
# each "TP,PP,EP,CP,SP" runs sequentially.
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set CONTAINER_IMAGE to your container path
#   3. Set DATASET_NAME to select dataset-specific defaults (see below)
#   4. Set PARALLELISM_CONFIGS (TP,PP,EP,CP,SP per entry; CP = context parallel size, 1 = disabled)
#   5. Submit: sbatch slurm_sft.sh
#
# Resume: if a checkpoint already exists at SAVE_DIR, training resumes automatically.
# ==============================================================================

#SBATCH --job-name=gpt-oss-sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8  # Change to 4 for GB200 (Blackwell, 4 GPUs/node)
#SBATCH --gpus-per-node=8    # Change to 4 for GB200 (Blackwell, 4 GPUs/node)
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --account=my_account
#SBATCH --output=logs/gpt_oss_sft_%j.out
#SBATCH --error=logs/gpt_oss_sft_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Base directory for container image and mounts (set if not already set, e.g. by launch_nemo.sh)
export WKDIR="${WKDIR:-}"

# Model and pretrained checkpoint
# Use base dir (e.g. .../gpt-oss-20b) with latest_checkpointed_iteration.txt, or Bridge dir with latest_train_state.pt
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/gpt-oss-20b}
MODEL_NAME=gpt_oss_20b

# Dataset — controls recipe and hyperparameter defaults below
# Supported presets: squad, openmathinstruct2_gsm8k
DATASET_NAME=${DATASET_NAME:-squad}

# Dataset-specific defaults (all overridable via env vars)
case "$DATASET_NAME" in
  openmathinstruct2*)
    # Packed sequences + analysis channel CoT; tuned for GSM8K math reasoning
    RECIPE_NAME="${RECIPE_NAME:-${MODEL_NAME}_sft_openmathinstruct2_thinking_packed_config}"
    SEQ_LENGTH=${SEQ_LENGTH:-4096}
    TRAIN_ITERS=${TRAIN_ITERS:-1000}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
    LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-250}
    LR=${LR:-5e-6}
    MIN_LR=${MIN_LR:-5e-7}
    ;;
  *)  # squad, custom datasets
    RECIPE_NAME="${RECIPE_NAME:-${MODEL_NAME}_sft_config}"
    SEQ_LENGTH=${SEQ_LENGTH:-2048}
    TRAIN_ITERS=${TRAIN_ITERS:-1000}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
    LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-50}
    ;;
esac

# Recipe overrides (optional; replaces the dataset-derived default above)
# RECIPE_NAME="${MODEL_NAME}_sft_fp8_current_scaling_config"   # Hopper FP8 current scaling
# RECIPE_NAME="${MODEL_NAME}_sft_mxfp8_config"                 # Blackwell MXFP8

MICRO_BATCH_SIZE=1
EVAL_ITERS=32
EVAL_INTERVAL=50
LOG_INTERVAL=1
# Optional suffix appended to the save directory name (e.g. "_run1", "_packed")
SAVE_SUFFIX="${SAVE_SUFFIX:-}"
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# Parallelism configs: "TP,PP,EP,CP,SP" per entry (max(TP*CP, EP)*PP must be divisible by the total number of GPUs)
PARALLELISM_CONFIGS=("2,2,4,1,True" "4,1,4,1,True")

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
# Increase heartbeat timeout for long checkpoint saves or slow nodes
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

# UV cache on shared filesystem (recommended for multi-node setups)
# Pre-sync once before submitting jobs: UV_CACHE_DIR=/path/to/cache uv sync
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# HuggingFace / NeMo cache directories (recommended for shared filesystem)
# export HF_HOME="/path/to/shared/HF_HOME"
# export NEMO_HOME="/path/to/shared/NEMO_HOME"

# Authentication tokens (set these for your environment)
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "GPT-OSS 20B Full SFT Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Recipe: $RECIPE_NAME"
echo "Train iters: $TRAIN_ITERS  LR warmup: $LR_WARMUP_ITERS"
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

    SAVE_DIR="${WORKSPACE}/results/${MODEL_NAME}_${DATASET_NAME}_finetune_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP}${SAVE_SUFFIX}"

    # Resume from existing checkpoint if available, otherwise start from pretrained
    if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "Resuming from existing checkpoint: $SAVE_DIR"
        CKPT_OVERRIDES="checkpoint.load=${SAVE_DIR} checkpoint.finetune=False"
    else
        echo "Starting fresh from pretrained checkpoint: $PRETRAINED_CHECKPOINT"
        CKPT_OVERRIDES="checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    fi

    # Build CLI overrides for this config (full SFT: no --peft_scheme)
    LR_OVERRIDES=""
    [ -n "$LR" ] && LR_OVERRIDES="$LR_OVERRIDES optimizer.lr=$LR"
    [ -n "$MIN_LR" ] && LR_OVERRIDES="$LR_OVERRIDES optimizer.min_lr=$MIN_LR"
    CLI_OVERRIDES=" \
        $CKPT_OVERRIDES \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        validation.eval_interval=$EVAL_INTERVAL \
        validation.eval_iters=$EVAL_ITERS \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        $LR_OVERRIDES \
        checkpoint.save=${SAVE_DIR} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_finetune_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP}${SAVE_SUFFIX} \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP \
        model.expert_model_parallel_size=$EP \
        model.expert_tensor_parallel_size=1 \
        model.sequence_parallel=$SP \
        model.context_parallel_size=$CP \
        model.calculate_per_token_loss=True \
        dataset.packed_sequence_specs.pad_seq_to_mult=$([ "$CP" -gt 1 ] && echo $((CP * 2)) || echo 1) \
        dataset.packed_sequence_specs.packed_sequence_size=$SEQ_LENGTH \
        dataset.seq_length=$SEQ_LENGTH \
        model.seq_length=$SEQ_LENGTH \
        dist.distributed_timeout_minutes=90
    "
    CMD="uv run --no-sync python /opt/Megatron-Bridge/scripts/training/run_recipe.py"
    CMD="$CMD --recipe ${RECIPE_NAME}"
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
