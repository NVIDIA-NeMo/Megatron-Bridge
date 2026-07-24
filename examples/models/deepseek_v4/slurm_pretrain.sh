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
# DeepSeek-V4-Flash Pretraining
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster.
#   2. Set CONTAINER_IMAGE to your container path.
#   3. Select RECIPE_NAME below: Adam MXFP8 or Muon BF16.
#   4. Submit: sbatch slurm_pretrain.sh
# ==============================================================================

#SBATCH --job-name=dsv4-pretrain
#SBATCH --account=my_account
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/dsv4_pretrain_%j.out
#SBATCH --error=logs/dsv4_pretrain_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
export WKDIR="${WKDIR:-}"

MODEL_NAME=deepseek_v4_flash
HF_CONFIG=deepseek-ai/DeepSeek-V4-Flash
RECIPE_NAME=deepseek_v4_flash_pretrain_mxfp8_config       # Adam MXFP8
# RECIPE_NAME=deepseek_v4_flash_pretrain_config           # Adam BF16
# RECIPE_NAME=deepseek_v4_flash_pretrain_muon_config      # Muon BF16

DATASET_NAME=dclm  # set to "mock" for mock data
SEQ_LENGTH=4096

# When DATASET_NAME=dclm, set DCLM_DATA_DIR and DCLM_CACHE so the recipe uses DCLM.
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
EVAL_INTERVAL=100
EVAL_ITERS=10
LR_WARMUP_ITERS=50
SAVE_INTERVAL=300
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-dsv4

# TP,PP,EP,CP. The default targets 8 GB200 nodes with 4 GPUs per node.
PARALLELISM_CONFIG=${PARALLELISM_CONFIG:-1,4,8,1}

CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data,/workspace:/workspace"

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_PXN_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Pre-sync once before submitting jobs: UV_CACHE_DIR=/path/to/cache uv sync
# export UV_CACHE_DIR="/path/to/shared/uv_cache"
# export HF_HOME="/path/to/shared/HF_HOME"
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"

# ==============================================================================
# Job Execution
# ==============================================================================

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

if [ "$DATASET_NAME" = "mock" ]; then
    DATASET_TYPE=mock
else
    DATASET_TYPE=megatron-indexed
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

DCLM_DATASET_OVERRIDES=""
if [ -n "${DCLM_DATA_DIR:-}" ] && [ -n "${DCLM_CACHE:-}" ]; then
    BLEND_PATHS=""
    for i in $(seq 1 10); do
        pad=$(printf "%02d" "$i")
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

RECIPE_OVERRIDES=""
case "$RECIPE_NAME" in
    deepseek_v4_flash_pretrain_mxfp8_config)
        # Adam MXFP8 model/train-state checkpoints are enabled; optimizer-state
        # checkpointing can OOM at this scale until the distributed optimizer save path is fixed.
        RECIPE_OVERRIDES="checkpoint.save_optim=false checkpoint.load_optim=false"
        ;;
    deepseek_v4_flash_pretrain_muon_config)
        ;;
    *)
        echo "ERROR: Unsupported RECIPE_NAME: $RECIPE_NAME"
        exit 1
        ;;
esac

OLD_IFS=$IFS
IFS=',' read -r TP PP EP CP <<< "$PARALLELISM_CONFIG"
IFS=$OLD_IFS

NNODES=${SLURM_JOB_NUM_NODES:-8}
NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29571}
MASTER_ADDR=$(python3 - <<'PY'
import os
import re

nodelist = os.environ.get("SLURM_NODELIST", "")
match = re.match(r"([\w-]+)\[(\d+)", nodelist)
print(match.group(1) + match.group(2) if match else (nodelist.split(",")[0] if nodelist else "localhost"))
PY
)

RUN_NAME=${MODEL_NAME}_${DATASET_NAME}_${RECIPE_NAME}_tp${TP}_pp${PP}_ep${EP}_cp${CP}_${SLURM_JOB_ID:-manual}
CHECKPOINT_DIR=${WORKSPACE}/results/${RUN_NAME}/checkpoints

CLI_OVERRIDES=" \
    model.seq_length=$SEQ_LENGTH \
    dataset.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    scheduler.lr_decay_iters=$TRAIN_ITERS \
    checkpoint.save=$CHECKPOINT_DIR \
    checkpoint.save_interval=$SAVE_INTERVAL \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=$RUN_NAME \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.expert_model_parallel_size=$EP \
    model.context_parallel_size=$CP \
    $RECIPE_OVERRIDES \
    $DCLM_DATASET_OVERRIDES"

CMD="uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /opt/Megatron-Bridge/scripts/training/run_recipe.py \
    --recipe $RECIPE_NAME \
    --dataset $DATASET_TYPE \
    --step_func gpt_step \
    $CLI_OVERRIDES"

echo "======================================"
echo "DeepSeek-V4-Flash Pretraining"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Recipe: $RECIPE_NAME"
echo "Hardware target: GB200/Blackwell"
echo "Dataset: $DATASET_TYPE/$DATASET_NAME"
echo "Parallelism: TP=$TP PP=$PP EP=$EP CP=$CP"
echo "Run name: $RUN_NAME"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "SRUN base: $SRUN_CMD"
echo "======================================"
echo "$CMD"
echo "======================================"

$SRUN_CMD bash -lc "
    set -euo pipefail
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export NCCL_PXN_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONPATH=/opt/Megatron-Bridge/src:/opt/megatron-lm:\${PYTHONPATH:-}
    cd /opt/Megatron-Bridge
    $CMD
"

echo "======================================"
echo "DeepSeek-V4 pretraining job completed"
echo "======================================"
