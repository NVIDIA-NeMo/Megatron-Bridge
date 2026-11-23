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
# Nemotron 3 Ultra DCLM Pretraining
#
# Usage:
#   1. Modify the #SBATCH directives for your cluster.
#   2. Set CONTAINER_IMAGE and DCLM_DATA_DIR.
#   3. Optionally set CONTAINER_MOUNTS, WORKSPACE, WORKDIR, or HF_MODEL_PATH.
#   4. Submit: sbatch slurm_pretrain.sh
# ==============================================================================

#SBATCH --job-name=nemotron-ultra-dclm-pretrain
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/nemotron_ultra_dclm_pretrain_%j.log
#SBATCH --exclusive

set -euo pipefail

# Required for most clusters:
CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
DCLM_DATA_DIR=${DCLM_DATA_DIR:-}

# Optional environment-specific paths:
WORKSPACE=${WORKSPACE:-/workspace}
WORKDIR=${WORKDIR:-/opt/Megatron-Bridge}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
HF_MODEL_PATH=${HF_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

# Starter profile:
MODEL_NAME=nemotron_3_ultra
RECIPE_NAME=nemotron_3_ultra_pretrain_config
SEQ_LENGTH=4096
DCLM_PATTERN=${DCLM_PATTERN:-dclm_01_*_text_document.bin}
DCLM_CACHE="${WORKSPACE}/data_cache/dclm"
TRAIN_ITERS=${TRAIN_ITERS:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
TP=${TP:-4}
PP=${PP:-12}
EP=${EP:-16}
ETP=${ETP:-1}
CP=${CP:-1}
SP=${SP:-True}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
if [ -z "${RECOMPUTE_MODULES+x}" ]; then
    RECOMPUTE_MODULES=""
fi
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
RECOMPUTE_TAG=${RECOMPUTE_TAG:-recompute_full_uniform1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SAVE_DIR="${WORKSPACE}/results/${MODEL_NAME}_dclm_pretrain_tp${TP}_pp${PP}_ep${EP}_${RECOMPUTE_TAG}_${SLURM_JOB_ID}"
WANDB_ENTITY=${WANDB_ENTITY:-nvidia-nemo-fw-public}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge-nemotron-ultra}
WANDB_EXP_NAME="${MODEL_NAME}_dclm_pretrain_tp${TP}_pp${PP}_ep${EP}_${RECOMPUTE_TAG}_${SLURM_JOB_ID}"
WANDB_MODE=${WANDB_MODE:-disabled}

[ -n "${HF_HOME:-}" ] && export HF_HOME
[ -n "${NEMO_HOME:-}" ] && export NEMO_HOME
[ -n "${UV_CACHE_DIR:-}" ] && export UV_CACHE_DIR
export WANDB_MODE
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}ignore:The AccumulateGrad node:UserWarning,ignore:The pad token id in the tokenizer collides:UserWarning"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: set CONTAINER_IMAGE to a valid sqsh image."
    exit 1
fi

if [ -z "$DCLM_DATA_DIR" ]; then
    echo "ERROR: set DCLM_DATA_DIR to a preprocessed DCLM directory."
    exit 2
fi

if [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY must be visible in the submit environment for online W&B logging."
    exit 3
fi

export DCLM_DATA_DIR DCLM_PATTERN DCLM_CACHE EXTRA_OVERRIDES HF_MODEL_PATH RECIPE_NAME SAVE_DIR SEQ_LENGTH
export TRAIN_ITERS SAVE_INTERVAL TP PP EP ETP CP SP RECOMPUTE_GRANULARITY RECOMPUTE_METHOD RECOMPUTE_MODULES
export RECOMPUTE_NUM_LAYERS GPUS_PER_NODE
export WANDB_ENTITY WANDB_PROJECT WANDB_EXP_NAME WORKDIR WORKSPACE

CMD='
set -euo pipefail
cd "$WORKDIR"
mkdir -p "$WORKSPACE/results" "$SAVE_DIR/wandb" "$SAVE_DIR/tb_logs" "$DCLM_CACHE"
export PYTHONPATH="$WORKDIR/src:$WORKDIR/3rdparty/Megatron-LM:${PYTHONPATH:-}"

BLEND_PATHS=""
shopt -s nullglob
for BIN_PATH in "$DCLM_DATA_DIR"/$DCLM_PATTERN; do
    PREFIX=${BIN_PATH%.bin}
    BLEND_PATHS="${BLEND_PATHS}\"${PREFIX}\","
done
shopt -u nullglob
BLEND_PATHS="${BLEND_PATHS%,}"

if [ -z "$BLEND_PATHS" ]; then
    echo "ERROR: no DCLM shards matching ${DCLM_DATA_DIR}/${DCLM_PATTERN}"
    exit 4
fi

OPTIONAL_OVERRIDES=()
if [ -n "$RECOMPUTE_METHOD" ]; then
    OPTIONAL_OVERRIDES+=(model.recompute_method="$RECOMPUTE_METHOD")
fi
if [ -n "$RECOMPUTE_MODULES" ]; then
    OPTIONAL_OVERRIDES+=(model.recompute_modules="$RECOMPUTE_MODULES")
fi
if [ -n "$RECOMPUTE_NUM_LAYERS" ]; then
    OPTIONAL_OVERRIDES+=(model.recompute_num_layers="$RECOMPUTE_NUM_LAYERS")
fi

uv run --no-sync python scripts/training/run_recipe.py \
    --recipe "$RECIPE_NAME" \
    --dataset megatron-indexed \
    dataset.seq_length="$SEQ_LENGTH" \
    checkpoint.save="$SAVE_DIR" \
    checkpoint.save_interval="$SAVE_INTERVAL" \
    train.train_iters="$TRAIN_ITERS" \
    train.global_batch_size=128 \
    train.micro_batch_size=1 \
    validation.eval_interval=100 \
    validation.eval_iters=10 \
    logger.log_interval=1 \
    logger.tensorboard_dir="$SAVE_DIR/tb_logs" \
    logger.wandb_entity="$WANDB_ENTITY" \
    logger.wandb_project="$WANDB_PROJECT" \
    logger.wandb_exp_name="$WANDB_EXP_NAME" \
    logger.wandb_save_dir="$SAVE_DIR/wandb" \
    model.tensor_model_parallel_size="$TP" \
    model.pipeline_model_parallel_size="$PP" \
    model.expert_model_parallel_size="$EP" \
    model.expert_tensor_parallel_size="$ETP" \
    model.sequence_parallel="$SP" \
    model.context_parallel_size="$CP" \
    model.seq_length="$SEQ_LENGTH" \
    dataset.seq_length="$SEQ_LENGTH" \
    model.recompute_granularity="$RECOMPUTE_GRANULARITY" \
    "${OPTIONAL_OVERRIDES[@]}" \
    dist.distributed_timeout_minutes=90 \
    "dataset.blend=[[${BLEND_PATHS}],null]" \
    dataset.split=\"9999,8,2\" \
    dataset.path_to_cache="$DCLM_CACHE" \
    ${EXTRA_OVERRIDES}
'

SRUN_CMD="srun --mpi=pmix --no-kill --container-image=${CONTAINER_IMAGE} --no-container-mount-home"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="${SRUN_CMD} --container-mounts=${CONTAINER_MOUNTS}"
fi

echo "======================================"
echo "Nemotron 3 Ultra DCLM Pretraining"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs/node: ${GPUS_PER_NODE}"
echo "Recipe: ${RECIPE_NAME}"
echo "HF model: ${HF_MODEL_PATH}"
echo "DCLM data: ${DCLM_DATA_DIR}"
echo "Parallelism: TP=${TP} PP=${PP} EP=${EP} ETP=${ETP} CP=${CP} SP=${SP}"
echo "Recompute: ${RECOMPUTE_GRANULARITY} ${RECOMPUTE_METHOD:-} ${RECOMPUTE_MODULES}"
echo "Save dir: ${SAVE_DIR}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})"
echo "======================================"

$SRUN_CMD bash -c "$CMD"

echo DCLM_PRETRAIN_DONE
