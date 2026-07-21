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
# Nemotron 3 Ultra OpenMath Full SFT
#
# Usage:
#   1. Modify the #SBATCH directives for your cluster.
#   2. Set CONTAINER_IMAGE and optional CONTAINER_MOUNTS.
#   3. Submit: sbatch slurm_sft.sh
# ==============================================================================

#SBATCH --job-name=nemotron-ultra-openmath-sft
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/nemotron_ultra_openmath_sft_%j.log
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
WORKDIR=${WORKDIR:-/opt/Megatron-Bridge}
MODEL_HOME=${MODEL_HOME:-${WORKSPACE}/models/nvidia}

HF_MODEL_PATH=${HF_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${MODEL_HOME}/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-megatron}
MODEL_NAME=nemotron_3_ultra
DATASET_NAME=openmathinstruct2
RECIPE_NAME=nemotron_3_ultra_sft_openmathinstruct2_packed_config

SEQ_LENGTH=4096
TRAIN_ITERS=${TRAIN_ITERS:-1000}
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
EVAL_ITERS=32
EVAL_INTERVAL=50
LR_WARMUP_ITERS=250
LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
LR=5e-6
MIN_LR=5e-7
SAVE_INTERVAL=${SAVE_INTERVAL:-$TRAIN_ITERS}
LOG_INTERVAL=1

TP=${TP:-2}
PP=${PP:-12}
EP=${EP:-16}
ETP=${ETP:-1}
CP=${CP:-1}
SP=${SP:-True}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PAD_SEQ_TO_MULT=${PAD_SEQ_TO_MULT:-}

if [ -z "$PAD_SEQ_TO_MULT" ]; then
    if [ "$CP" -gt 1 ]; then
        PAD_SEQ_TO_MULT=$((2 * CP))
        case "$SP" in
            True | true | TRUE | 1 | yes | YES)
                CP_SP_MULT=$((CP * TP))
                GCD_A=$PAD_SEQ_TO_MULT
                GCD_B=$CP_SP_MULT
                while [ "$GCD_B" -ne 0 ]; do
                    GCD_TMP=$((GCD_A % GCD_B))
                    GCD_A=$GCD_B
                    GCD_B=$GCD_TMP
                done
                PAD_SEQ_TO_MULT=$((PAD_SEQ_TO_MULT / GCD_A * CP_SP_MULT))
                ;;
        esac
    else
        PAD_SEQ_TO_MULT=1
    fi
fi

RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
if [ -z "${RECOMPUTE_MODULES+x}" ]; then
    RECOMPUTE_MODULES=""
fi
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
RECOMPUTE_TAG=${RECOMPUTE_TAG:-recompute_full_uniform1}

WANDB_ENTITY=${WANDB_ENTITY:-nvidia-nemo-fw-public}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge-nemotron-ultra}
WANDB_MODE=${WANDB_MODE:-disabled}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

# ==============================================================================
# Environment Setup
# ==============================================================================

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
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}ignore:The AccumulateGrad node:UserWarning"

# ==============================================================================
# Job Execution
# ==============================================================================

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

if [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY must be visible in the submit environment for online W&B logging."
    exit 2
fi

SAVE_DIR="${WORKSPACE}/results/${MODEL_NAME}_${DATASET_NAME}_sft_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP}_${RECOMPUTE_TAG}_${SLURM_JOB_ID}"
WANDB_EXP_NAME="${MODEL_NAME}_${DATASET_NAME}_sft_tp${TP}_pp${PP}_ep${EP}_${RECOMPUTE_TAG}_${SLURM_JOB_ID}"

CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT} \
    checkpoint.save=${SAVE_DIR} \
    checkpoint.save_interval=${SAVE_INTERVAL} \
    train.train_iters=${TRAIN_ITERS} \
    train.global_batch_size=${GLOBAL_BATCH_SIZE} \
    train.micro_batch_size=${MICRO_BATCH_SIZE} \
    validation.eval_interval=${EVAL_INTERVAL} \
    validation.eval_iters=${EVAL_ITERS} \
    scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
    scheduler.lr_decay_iters=${LR_DECAY_ITERS} \
    optimizer.lr=${LR} \
    optimizer.min_lr=${MIN_LR} \
    logger.log_interval=${LOG_INTERVAL} \
    logger.tensorboard_dir=${SAVE_DIR}/tb_logs \
    logger.wandb_entity=${WANDB_ENTITY} \
    logger.wandb_project=${WANDB_PROJECT} \
    logger.wandb_exp_name=${WANDB_EXP_NAME} \
    logger.wandb_save_dir=${SAVE_DIR}/wandb \
    model.tensor_model_parallel_size=${TP} \
    model.pipeline_model_parallel_size=${PP} \
    model.expert_model_parallel_size=${EP} \
    model.expert_tensor_parallel_size=${ETP} \
    model.sequence_parallel=${SP} \
    model.context_parallel_size=${CP} \
    model.seq_length=${SEQ_LENGTH} \
    dataset.seq_length=${SEQ_LENGTH} \
    dataset.offline_packing_specs.packed_sequence_size=${SEQ_LENGTH} \
    dataset.offline_packing_specs.pad_seq_to_mult=${PAD_SEQ_TO_MULT} \
    model.recompute_granularity=${RECOMPUTE_GRANULARITY} \
    dist.distributed_timeout_minutes=90"

if [ -n "$RECOMPUTE_METHOD" ]; then
    CLI_OVERRIDES="${CLI_OVERRIDES} model.recompute_method=${RECOMPUTE_METHOD}"
fi
if [ -n "$RECOMPUTE_MODULES" ]; then
    CLI_OVERRIDES="${CLI_OVERRIDES} model.recompute_modules=${RECOMPUTE_MODULES}"
fi
if [ -n "$RECOMPUTE_NUM_LAYERS" ]; then
    CLI_OVERRIDES="${CLI_OVERRIDES} model.recompute_num_layers=${RECOMPUTE_NUM_LAYERS}"
fi
CLI_OVERRIDES="${CLI_OVERRIDES} ${EXTRA_OVERRIDES}"

CMD="cd ${WORKDIR} && mkdir -p ${WORKSPACE}/results ${SAVE_DIR}/wandb ${SAVE_DIR}/tb_logs && \
export PYTHONPATH=${WORKDIR}/src:${WORKDIR}/3rdparty/Megatron-LM:\${PYTHONPATH:-} && \
uv run --no-sync python scripts/training/run_recipe.py \
--recipe ${RECIPE_NAME} \
${CLI_OVERRIDES}"

SRUN_CMD="srun --mpi=pmix --no-kill --container-image=${CONTAINER_IMAGE} --no-container-mount-home"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="${SRUN_CMD} --container-mounts=${CONTAINER_MOUNTS}"
fi

echo "======================================"
echo "Nemotron 3 Ultra OpenMath Full SFT"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs/node: ${GPUS_PER_NODE}"
echo "Recipe: ${RECIPE_NAME}"
echo "Parallelism: TP=${TP} PP=${PP} EP=${EP} ETP=${ETP} CP=${CP} SP=${SP}"
echo "Packed pad_seq_to_mult: ${PAD_SEQ_TO_MULT}"
echo "Recompute: ${RECOMPUTE_GRANULARITY} ${RECOMPUTE_METHOD:-} ${RECOMPUTE_MODULES}"
echo "Save dir: ${SAVE_DIR}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})"
echo "======================================"

$SRUN_CMD bash -c "$CMD"

LATEST_ITER_FILE="${SAVE_DIR}/latest_checkpointed_iteration.txt"
if [ -f "$LATEST_ITER_FILE" ]; then
    LATEST_ITER=$(tr -d '[:space:]' < "$LATEST_ITER_FILE")
    if [[ "$LATEST_ITER" =~ ^[0-9]+$ ]]; then
        LATEST_DIR=$(printf "iter_%07d" "$LATEST_ITER")
        find "$SAVE_DIR" -mindepth 1 -maxdepth 1 -type d -name "iter_*" ! -name "$LATEST_DIR" -exec rm -rf {} +
    else
        echo "Skipping SFT intermediate checkpoint cleanup: latest checkpoint marker is not numeric: ${LATEST_ITER}"
    fi
else
    echo "Skipping SFT intermediate checkpoint cleanup: missing ${LATEST_ITER_FILE}"
fi

echo OPENMATH_SFT_DONE
