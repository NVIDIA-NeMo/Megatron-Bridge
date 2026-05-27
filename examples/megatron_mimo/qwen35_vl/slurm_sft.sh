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
#
# Qwen3.5-VL MegatronMIMO HF CORD-v2 SFT Slurm launcher.
#
# Default layout:
#   language: TP4 PP4 DP2 on ranks 0-31
#   images:   TP1 PP1 DP1 on rank 32
#
# Request five 8-GPU nodes and launch only the 33 active ranks.

#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --nodes=5
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --job-name=qwen35_vl-mimo-sft
#SBATCH --output=qwen35_vl_mimo_sft_%j.out
#SBATCH --error=qwen35_vl_mimo_sft_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MBRIDGE_DIR="/opt/Megatron-Bridge"

WORKSPACE="${WORKSPACE:-/workspace}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORKSPACE}/qwen35_vl_mimo}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${IMAGE_PATH:-}}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${REPO_ROOT}:${MBRIDGE_DIR},${WORKSPACE}:${WORKSPACE}}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-27B}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"

LANGUAGE_TP=4
LANGUAGE_PP=4
LANGUAGE_DP=2
LANGUAGE_OFFSET=0
IMAGES_TP=1
IMAGES_PP=1
IMAGES_DP=1
IMAGES_OFFSET=32
MIMO_WORLD_SIZE=33
GPUS_PER_NODE=8

INIT_MODE="random_init"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
if [[ -n "${LOAD_CHECKPOINT}" ]]; then
    INIT_MODE="resume"
elif [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    INIT_MODE="pretrained"
fi

RUN_NAME="${RUN_NAME:-27b_cord_v2_mimo_hf_l32_v1_gbs32_lmbs1_seq${SEQ_LENGTH}_${INIT_MODE}}"
RUN_DIR="${CHECKPOINT_DIR:-${EXPERIMENT_ROOT}/results/mimo/${RUN_NAME}}"
LOG_DIR="${RUN_DIR}/logs"
RANK_LOG_DIR="${LOG_DIR}/rank_logs"
TENSORBOARD_DIR="${RUN_DIR}/tb_logs"
RUN_LOG="${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID}.log"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${RUN_NAME}}"

if [[ -z "${CONTAINER_IMAGE}" ]]; then
    echo "CONTAINER_IMAGE must point to a valid container image, for example /path/to/nemo.sqsh." >&2
    exit 2
fi

required_paths=(
    "${REPO_ROOT}"
    "${REPO_ROOT}/examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py"
    "${REPO_ROOT}/3rdparty/Megatron-LM"
)
if [[ -n "${LOAD_CHECKPOINT}" ]]; then
    required_paths+=("${LOAD_CHECKPOINT}")
elif [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    required_paths+=("${PRETRAINED_CHECKPOINT}")
fi
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

ALLOCATED_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
if (( MIMO_WORLD_SIZE > ALLOCATED_GPUS )); then
    echo "MIMO layout uses ${MIMO_WORLD_SIZE} ranks but only ${ALLOCATED_GPUS} GPUs are allocated." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}" "${RANK_LOG_DIR}" "${TENSORBOARD_DIR}"
ln -sf "${RUN_LOG}" "${LOG_DIR}/latest.log"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export HTTPX_LOG_LEVEL=WARNING
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning:torch.cuda,ignore::UserWarning:modelopt.torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE}/uv_cache}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/hf_home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE}/triton_cache}"
export NEMO_HOME="${NEMO_HOME:-${WORKSPACE}/nemo_home}"
export WANDB_MODE="${WANDB_MODE:-${WANDB_PROJECT:+online}}"
if [[ -z "${WANDB_MODE}" ]]; then
    export WANDB_MODE=disabled
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 40000))}"
export MASTER_ADDR MASTER_PORT

RUN_CMD=$(cat <<'EOF'
set -euo pipefail
cd "${MBRIDGE_DIR}"

export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${MIMO_WORLD_SIZE}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${GPUS_PER_NODE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cmd=(
    uv run --no-sync python
    examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py
    --hf-model "${HF_MODEL}"
    --experiment-root "${EXPERIMENT_ROOT}"
    --run-name "${RUN_NAME}"
    --train-iters "${TRAIN_ITERS}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --seq-length "${SEQ_LENGTH}"
    --log-dir "${RANK_LOG_DIR}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --checkpoint-dir "${RUN_DIR}"
    --component "language=tp=${LANGUAGE_TP},pp=${LANGUAGE_PP},dp=${LANGUAGE_DP},rank_offset=${LANGUAGE_OFFSET}"
    --component "images=tp=${IMAGES_TP},pp=${IMAGES_PP},dp=${IMAGES_DP},rank_offset=${IMAGES_OFFSET}"
)

if [[ -n "${LOAD_CHECKPOINT}" ]]; then
    cmd+=(--load-checkpoint "${LOAD_CHECKPOINT}")
elif [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    cmd+=(--pretrained-checkpoint "${PRETRAINED_CHECKPOINT}")
else
    cmd+=(--allow-random-init)
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
    cmd+=(--wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_EXP_NAME}")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
EOF
)

export MBRIDGE_DIR
export HF_MODEL EXPERIMENT_ROOT RUN_NAME RUN_DIR RANK_LOG_DIR TENSORBOARD_DIR
export TRAIN_ITERS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SEQ_LENGTH
export LANGUAGE_TP LANGUAGE_PP LANGUAGE_DP LANGUAGE_OFFSET
export IMAGES_TP IMAGES_PP IMAGES_DP IMAGES_OFFSET
export MIMO_WORLD_SIZE GPUS_PER_NODE
export LOAD_CHECKPOINT PRETRAINED_CHECKPOINT
export WANDB_PROJECT WANDB_EXP_NAME
export SLURM_EXPORT_ENV=ALL

echo "======================================"
echo "Qwen3.5-VL MegatronMIMO HF SFT"
echo "======================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Run name:        ${RUN_NAME}"
echo "Run dir:         ${RUN_DIR}"
echo "Run log:         ${RUN_LOG}"
echo "HF model:        ${HF_MODEL}"
echo "Active ranks:    ${MIMO_WORLD_SIZE}"
echo "Allocated GPUs:  ${ALLOCATED_GPUS}"
echo "Language:        TP=${LANGUAGE_TP} PP=${LANGUAGE_PP} DP=${LANGUAGE_DP} offset=${LANGUAGE_OFFSET}"
echo "Images:          TP=${IMAGES_TP} PP=${IMAGES_PP} DP=${IMAGES_DP} offset=${IMAGES_OFFSET}"
echo "Batch:           MIMO mbs=${MICRO_BATCH_SIZE}, gbs=${GLOBAL_BATCH_SIZE}, language-local mbs=1"
echo "Seq length:      ${SEQ_LENGTH}"
echo "Train iters:     ${TRAIN_ITERS}"
if [[ -n "${LOAD_CHECKPOINT}" ]]; then
    echo "Checkpoint load: ${LOAD_CHECKPOINT}"
elif [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "Pretrained ckpt: ${PRETRAINED_CHECKPOINT}"
else
    echo "Init:            random"
fi
echo "W&B:             ${WANDB_PROJECT:-disabled}"
echo "Container:       ${CONTAINER_IMAGE}"
echo "======================================"

mapfile -t PACKED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
REQUIRED_NODES=$(((MIMO_WORLD_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
if (( REQUIRED_NODES > ${#PACKED_NODES[@]} )); then
    echo "MIMO layout needs ${REQUIRED_NODES} packed nodes but allocation has ${#PACKED_NODES[@]}." >&2
    exit 2
fi

srun_cmd=(
    srun -l
    --kill-on-bad-exit=1
    --distribution=block:block
    --export=ALL
    --container-image "${CONTAINER_IMAGE}"
    --container-mounts "${CONTAINER_MOUNTS}"
    --output="${RUN_LOG}"
    --error="${RUN_LOG}"
)

remaining_tasks="${MIMO_WORLD_SIZE}"
for node_index in "${!PACKED_NODES[@]}"; do
    if (( remaining_tasks <= 0 )); then
        break
    fi
    tasks_on_node="${GPUS_PER_NODE}"
    if (( remaining_tasks < GPUS_PER_NODE )); then
        tasks_on_node="${remaining_tasks}"
    fi
    if (( node_index > 0 )); then
        srun_cmd+=(":")
    fi
    srun_cmd+=(
        --nodes=1
        --ntasks="${tasks_on_node}"
        --ntasks-per-node="${tasks_on_node}"
        --nodelist="${PACKED_NODES[$node_index]}"
        bash -lc "${RUN_CMD}"
    )
    remaining_tasks=$((remaining_tasks - tasks_on_node))
done

"${srun_cmd[@]}"

echo "======================================"
echo "Job completed"
echo "======================================"
