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
# This script runs one DeepSeek-V4-Flash recipe through scripts/training/run_recipe.py.
# Defaults are a release-candidate 1k pretraining run: seq4096, GBS128,
# validation every 100 iterations, 10 eval iterations, TP1/PP4/EP8/CP1.
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster.
#   2. Set CONTAINER_IMAGE to your container path.
#   3. If needed, mount datasets/dependencies through CONTAINER_MOUNTS and add
#      extra Python dependency paths through EXTRA_PYTHONPATH.
#   4. Submit: sbatch slurm_pretrain.sh
#
# Release pretrain recipes:
#   Adam MXFP8 (default):
#     sbatch --job-name=dsv4-adam-mxfp8 --export=ALL,RECIPE_NAME=deepseek_v4_flash_pretrain_mxfp8_config,CASE_NAME=dsv4_adam_mxfp8 slurm_pretrain.sh
#   Muon BF16:
#     sbatch --job-name=dsv4-muon-bf16 --export=ALL,RECIPE_NAME=deepseek_v4_flash_pretrain_muon_config,CASE_NAME=dsv4_muon_bf16 slurm_pretrain.sh
#
# Fast smoke override:
#   sbatch --export=ALL,DATASET_NAME=mock,DATASET_TYPE=mock,SEQ_LENGTH=128,TRAIN_ITERS=60,EVAL_INTERVAL=20,EVAL_ITERS=5 slurm_pretrain.sh
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

# Workspace directory for checkpoints, logs, and results.
WORKSPACE="${WORKSPACE:-/workspace}"

# Paths inside the container. The default container is expected to provide Bridge
# and Megatron-LM at these locations. Override them when mounting local checkouts.
BRIDGE_PATH="${BRIDGE_PATH:-/opt/Megatron-Bridge}"
MCORE_PATH="${MCORE_PATH:-/opt/megatron-lm}"

# Hugging Face model id or local path. Use a local path for offline clusters.
HF_CONFIG="${HF_CONFIG:-deepseek-ai/DeepSeek-V4-Flash}"
HF_HOME="${HF_HOME:-${WORKSPACE}/hf_home}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Add colon-separated paths for dependencies not included in the container, e.g.
# external Megatron-LM checkout, FlashMLA, cuDNN frontend, fast-hadamard-transform,
# transformers preview wheels, or emerging-optimizers for Muon.
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-deepseek_v4_flash}"
RECIPE_NAME="${RECIPE_NAME:-deepseek_v4_flash_pretrain_mxfp8_config}"
DATASET_NAME="${DATASET_NAME:-dclm}"  # set to "mock" for mock data
DATASET_TYPE="${DATASET_TYPE:-llm-pretrain}"
STEP_FUNC="${STEP_FUNC:-gpt_step}"
MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"

SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-50}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-true}"
FULLY_PARALLEL_SAVE="${FULLY_PARALLEL_SAVE:-}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"

TP="${TP:-1}"
PP="${PP:-4}"
CP="${CP:-1}"
EP="${EP:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
NNODES="${SLURM_JOB_NUM_NODES:-${NNODES:-8}}"
MASTER_PORT="${MASTER_PORT:-29571}"

NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
RECOMPUTE_OVERRIDES="${RECOMPUTE_OVERRIDES:-model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
RECIPE_OVERRIDES=""

# When DATASET_NAME=dclm, set these to preprocessed DCLM paths. Leave unset for
# mock data or if dataset.blend is supplied through EXTRA_OVERRIDES.
DCLM_DATA_DIR="${DCLM_DATA_DIR:-}"
DCLM_CACHE="${DCLM_CACHE:-${WORKSPACE}/cache}"

WANDB_PROJECT="${WANDB_PROJECT:-megatron-bridge-dsv4}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-}"
WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-${WORKSPACE}/wandb}"
WANDB_MODE="${WANDB_MODE:-online}"

# Container image (required).
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional, comma-separated for srun --container-mounts).
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"
# CONTAINER_MOUNTS="/data:/data,/workspace:/workspace"

CASE_NAME="${CASE_NAME:-${RECIPE_NAME}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTDIR="${OUTDIR:-${WORKSPACE}/results/${MODEL_NAME}_${CASE_NAME}_${JOB_ID}}"

# ==============================================================================
# Recipe-specific overrides
# ==============================================================================

case "$RECIPE_NAME" in
    deepseek_v4_flash_pretrain_mxfp8_config)
        # TE FusedAdam optimizer-state materialization can OOM at this scale.
        # Save model/train state by default; pass explicit checkpoint overrides
        # through EXTRA_OVERRIDES when testing full optimizer-state checkpointing.
        RECIPE_OVERRIDES="checkpoint.save_optim=false checkpoint.load_optim=false"
        ;;
    deepseek_v4_flash_pretrain_muon_config)
        RECIPE_OVERRIDES=""
        ;;
    *)
        echo "ERROR: slurm_pretrain.sh release pretrain recipes are deepseek_v4_flash_pretrain_mxfp8_config and deepseek_v4_flash_pretrain_muon_config."
        echo "       Use a custom script or explicit recipe test harness for experimental recipes."
        exit 1
        ;;
esac

if [ "$DATASET_NAME" = "mock" ]; then
    DATASET_TYPE="mock"
    unset DCLM_DATA_DIR
fi

if [ -z "$WANDB_EXP_NAME" ]; then
    WANDB_EXP_NAME="${CASE_NAME}_tp${TP}_pp${PP}_ep${EP}_cp${CP}_seq${SEQ_LENGTH}_gbs${GLOBAL_BATCH_SIZE}_${JOB_ID}"
fi

# ==============================================================================
# Environment setup and validation
# ==============================================================================

mkdir -p logs "$OUTDIR" "$WANDB_SAVE_DIR" "$DCLM_CACHE"

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

if [ -n "$LOAD_CHECKPOINT" ] && [ ! -e "$LOAD_CHECKPOINT" ]; then
    echo "ERROR: LOAD_CHECKPOINT does not exist: $LOAD_CHECKPOINT"
    exit 1
fi

MASTER_ADDR=$(python3 - <<'PY'
import os
import re

s = os.environ.get("SLURM_NODELIST", "")
m = re.match(r"([\w-]+)\[(\d+)", s)
print(m.group(1) + m.group(2) if m else (s.split(",")[0] if s else "localhost"))
PY
)

DATASET_OVERRIDES=""
if [ "$DATASET_TYPE" = "llm-pretrain" ] && [ "$DATASET_NAME" = "dclm" ]; then
    if [ -n "$DCLM_DATA_DIR" ]; then
        BLEND_PATHS=""
        for i in $(seq 1 10); do
            pad=$(printf "%02d" "$i")
            prefix="${DCLM_DATA_DIR}/dclm_01_${pad}_text_document"
            if [ -f "${prefix}.bin" ]; then
                BLEND_PATHS="${BLEND_PATHS}\"${prefix}\","
            fi
        done
        BLEND_PATHS="${BLEND_PATHS%,}"
        if [ -n "$BLEND_PATHS" ]; then
            DATASET_OVERRIDES="dataset.blend=[[${BLEND_PATHS}],null] dataset.split='\"9999,8,2\"' dataset.path_to_cache=${DCLM_CACHE}"
        else
            echo "WARNING: No DCLM data found in ${DCLM_DATA_DIR}."
        fi
    else
        echo "WARNING: DCLM_DATA_DIR is not set. Set DATASET_NAME=mock for mock data or pass dataset.blend through EXTRA_OVERRIDES."
    fi
fi

CHECKPOINT_OVERRIDES="checkpoint.save=null checkpoint.load=null checkpoint.save_interval=0"
if [ "$SAVE_CHECKPOINTS" = "true" ]; then
    CHECKPOINT_OVERRIDES="checkpoint.save=${OUTDIR}/checkpoints checkpoint.load=null checkpoint.save_interval=${SAVE_INTERVAL}"
fi
if [ -n "$LOAD_CHECKPOINT" ]; then
    CHECKPOINT_OVERRIDES="$CHECKPOINT_OVERRIDES checkpoint.load=${LOAD_CHECKPOINT}"
fi
if [ -n "$FULLY_PARALLEL_SAVE" ]; then
    CHECKPOINT_OVERRIDES="$CHECKPOINT_OVERRIDES checkpoint.fully_parallel_save=${FULLY_PARALLEL_SAVE}"
fi

LOGGER_OVERRIDES="logger.wandb_project=$WANDB_PROJECT logger.wandb_exp_name=$WANDB_EXP_NAME logger.wandb_save_dir=$WANDB_SAVE_DIR logger.log_interval=$LOG_INTERVAL"
if [ -n "$WANDB_ENTITY" ]; then
    LOGGER_OVERRIDES="$LOGGER_OVERRIDES logger.wandb_entity=$WANDB_ENTITY"
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

PYTHONPATH_ENTRIES="$BRIDGE_PATH/src:$MCORE_PATH"
if [ -n "$EXTRA_PYTHONPATH" ]; then
    PYTHONPATH_ENTRIES="$EXTRA_PYTHONPATH:$PYTHONPATH_ENTRIES"
fi

# ==============================================================================
# Job execution
# ==============================================================================

echo "======================================"
echo "DeepSeek-V4-Flash Pretraining"
echo "======================================"
echo "Job ID: ${JOB_ID}"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${NPROC_PER_NODE}"
echo "Recipe: ${RECIPE_NAME}"
echo "Parallelism: TP=${TP} PP=${PP} CP=${CP} EP=${EP}"
echo "Sequence length: ${SEQ_LENGTH}"
echo "Train iters: ${TRAIN_ITERS}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Dataset type/name: ${DATASET_TYPE}/${DATASET_NAME}"
echo "Bridge path: ${BRIDGE_PATH}"
echo "MCore path: ${MCORE_PATH}"
echo "HF config: ${HF_CONFIG}"
echo "Output dir: ${OUTDIR}"
echo "Checkpoint overrides: ${CHECKPOINT_OVERRIDES}"
echo "Recipe overrides: ${RECIPE_OVERRIDES:-<none>}"
echo "Extra overrides: ${EXTRA_OVERRIDES:-<none>}"
echo "Extra PYTHONPATH: ${EXTRA_PYTHONPATH:-<none>}"
echo "SRUN base: ${SRUN_CMD}"
echo "======================================"

CLI_OVERRIDES=" \
    train.train_iters=$TRAIN_ITERS \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    dataset.sequence_length=$SEQ_LENGTH \
    model.seq_length=$SEQ_LENGTH \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.context_parallel_size=$CP \
    model.expert_model_parallel_size=$EP \
    model.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER_TYPE \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    scheduler.lr_decay_iters=$LR_DECAY_ITERS \
    $CHECKPOINT_OVERRIDES \
    $LOGGER_OVERRIDES \
    $RECOMPUTE_OVERRIDES \
    $RECIPE_OVERRIDES \
    $DATASET_OVERRIDES \
    $EXTRA_OVERRIDES"

CMD="uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $BRIDGE_PATH/scripts/training/run_recipe.py \
    --recipe $RECIPE_NAME \
    --dataset $DATASET_TYPE \
    --step_func $STEP_FUNC \
    --hf_path $HF_CONFIG \
    $CLI_OVERRIDES"

echo "Executing command..."
echo "$CMD"
echo "======================================"

$SRUN_CMD bash -lc "
    set -euo pipefail
    export HF_HOME=$HF_HOME
    export HF_HUB_OFFLINE=$HF_HUB_OFFLINE
    export WANDB_MODE=$WANDB_MODE
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export NCCL_PXN_DISABLE=$NCCL_PXN_DISABLE
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_TIMEOUT=1800000
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONPATH=$PYTHONPATH_ENTRIES:\${PYTHONPATH:-}
    cd $BRIDGE_PATH
    echo \"$CMD\"
    $CMD
"

echo "======================================"
echo "DeepSeek-V4 pretraining job completed"
echo "======================================"
