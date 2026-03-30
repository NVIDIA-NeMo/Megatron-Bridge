#!/bin/bash
set -euo pipefail
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
# Kimi-K2.5-VL Full Supervised Fine-Tuning (SFT)
#
# Full model (~1T params, 384 MoE experts, FP8 expert weights)
# Recommended parallelism: TP=2, PP=16, EP=32 (1024 GPUs, 128 nodes)
#
# Recipe: kimi_k25_vl_sft_config
#   - Muon optimizer with cosine annealing
#   - Sequence length 4096
#   - Recompute: full, uniform, 1 layer
#   - MoE: alltoall dispatcher with DeepEP backend
#   - trust_remote_code required
#
# Usage:
#   sbatch slurm_sft.sh
# ==============================================================================

#SBATCH --job-name=kimi-k25-vl-sft
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=72:00:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}

RECIPE="kimi_k25_vl_sft_config"
HF_MODEL_PATH="moonshotai/Kimi-K2.5"

PRETRAINED_CHECKPOINT=${WORKSPACE}/models/Kimi-K2.5
DATASET_NAME=cord_v2
SEQ_LENGTH=4096
TRAIN_ITERS=500
GLOBAL_BATCH_SIZE=4096
MICRO_BATCH_SIZE=1
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-kimi-k25-vl

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
echo "Kimi-K2.5-VL Full SFT Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "Recipe: $RECIPE"
echo "Checkpoint: $PRETRAINED_CHECKPOINT"
echo "======================================"

CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=${WORKSPACE}/results/${RECIPE} \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${RECIPE}_${DATASET_NAME} \
    dataset.maker_name=make_${DATASET_NAME}_dataset \
    dataset.seq_length=$SEQ_LENGTH"

CMD="uv run --no-sync python scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func vlm_step \
    --hf_path $HF_MODEL_PATH \
    --trust_remote_code \
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
