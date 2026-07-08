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
# Qwen3.5-VL 27B MegatronMIMO Supervised Fine-Tuning (non-colocated mode)
#
# Trains the language module and the image encoder on disjoint GPU sets:
#   - language: TP=4, PP=2, DP=2, rank offset=0   -> 16 ranks
#   - images:   TP=1, PP=1, DP=1, rank offset=16  ->  1 rank
#   - total active MIMO ranks: 17
#
# This layout needs 17 active ranks. The #SBATCH defaults below request 3 nodes
# of 8 GPUs (24 GPUs) and pack the 17 ranks as 8 + 8 + 1 tasks per node, which
# suits clusters that allocate whole 8-GPU nodes exclusively. This is an example,
# not a universal default: the only hard requirement is that the allocation
# provides at least 17 GPUs (the script validates this below). Adjust --nodes,
# --gpus-per-node, GPUS_PER_NODE, and --exclusive for your cluster -- e.g. on a
# cluster that allows partial-node allocation you can request exactly 17 GPUs.
# This is the non-colocated layout validated for Qwen3.5-VL 27B; see
# examples/megatron_mimo/qwen35_vl/README.md.
#
# Usage:
#   sbatch slurm_sft.sh
#
# Or override knobs at submit time without editing the file, e.g.:
#   sbatch --export=ALL,SEQ_LENGTH=2048,TRAIN_ITERS=100 slurm_sft.sh
# ==============================================================================

# ------------------------------------------------------------------------------
# Slurm resource request. These values are examples, not universal defaults --
# edit the node count, account, partition, time limit, and GPUs-per-node to match
# your cluster (--partition and --account in particular are cluster-specific).
# The defaults assume 8-GPU nodes under whole-node exclusive allocation; the
# layout only needs at least 17 GPUs total (see the validation block below).
# ------------------------------------------------------------------------------
#SBATCH --job-name=qwen35vl-mimo-sft
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=qwen35vl_mimo_sft_%j.out
#SBATCH --error=qwen35vl_mimo_sft_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
#                              USER CONFIGURATION
#
# Everything you need to configure for your runs lives in this section.
# Each knob is env-overridable, so you can change defaults inline:
#   sbatch --export=ALL,SEQ_LENGTH=2048 slurm_sft.sh
# ==============================================================================

# --- Container (REQUIRED) -----------------------------------------------------
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
# CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"
# CONTAINER_MOUNTS="/data:/data /workspace:/workspace"

# --- Tokens and shared caches (uncomment what you need) -----------------------
# export HF_TOKEN="hf_your_token_here"
# export HF_HOME="/path/to/shared/HF_HOME"
# export UV_CACHE_DIR="/path/to/shared/uv_cache"
# export WANDB_API_KEY="your_wandb_key_here"
# export WANDB_MODE=disabled   # uncomment to disable W&B logging

# --- Paths --------------------------------------------------------------------
# EXPERIMENT_ROOT matches the convention documented in README.md (conversion
# section): converted MIMO checkpoints live under ${EXPERIMENT_ROOT}/models/mimo
# and Slurm run outputs under ${EXPERIMENT_ROOT}/results/mimo.
WORKSPACE="${WORKSPACE:-/workspace}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORKSPACE}/qwen35_vl_mimo}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${EXPERIMENT_ROOT}/models/mimo/Qwen3.5-27B-mimo}"
RUN_NAME="${RUN_NAME:-qwen35-27b-mimo-cord_v2-sft}"

# --- Training hyperparameters -------------------------------------------------
DATASET_NAME="${DATASET_NAME:-cord_v2}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
# MIMO MICRO_BATCH_SIZE is the global microbatch across the language DP group.
# With language DP=2, MBS=2 gives language-local MBS=1 (matches standard 27B SFT).
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-megatron-bridge-${DATASET_NAME}-mimo}"

# --- Advanced: MIMO non-colocated parallelism layout (27B validated) ----------
# Changing these requires verifying convergence. Active ranks must fit in
# (#SBATCH --nodes) * (GPUS_PER_NODE); see validation block below.
MIMO_LANGUAGE_TP="${MIMO_LANGUAGE_TP:-4}"
MIMO_LANGUAGE_PP="${MIMO_LANGUAGE_PP:-2}"
MIMO_LANGUAGE_CP="${MIMO_LANGUAGE_CP:-1}"
MIMO_LANGUAGE_DP="${MIMO_LANGUAGE_DP:-2}"
MIMO_LANGUAGE_OFFSET="${MIMO_LANGUAGE_OFFSET:-0}"
MIMO_IMAGES_TP="${MIMO_IMAGES_TP:-1}"
MIMO_IMAGES_PP="${MIMO_IMAGES_PP:-1}"
MIMO_IMAGES_CP="${MIMO_IMAGES_CP:-1}"
MIMO_IMAGES_DP="${MIMO_IMAGES_DP:-1}"
MIMO_IMAGES_OFFSET="${MIMO_IMAGES_OFFSET:-16}"

# --- Model, cluster topology, and output directory ----------------------------
HF_MODEL_NAME="${HF_MODEL_NAME:-Qwen3.5-27B}"
HF_MODEL="${HF_MODEL:-Qwen/${HF_MODEL_NAME}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/results/mimo/${RUN_NAME}}"

# ==============================================================================
#       === Nothing below this line needs user modification ===
# ==============================================================================

# --- Derived values and layout validation -------------------------------------
MIMO_LANGUAGE_SIZE=$((MIMO_LANGUAGE_TP * MIMO_LANGUAGE_PP * MIMO_LANGUAGE_CP * MIMO_LANGUAGE_DP))
MIMO_IMAGES_SIZE=$((MIMO_IMAGES_TP * MIMO_IMAGES_PP * MIMO_IMAGES_CP * MIMO_IMAGES_DP))
MIMO_LANGUAGE_END=$((MIMO_LANGUAGE_OFFSET + MIMO_LANGUAGE_SIZE))
MIMO_IMAGES_END=$((MIMO_IMAGES_OFFSET + MIMO_IMAGES_SIZE))
MIMO_WORLD_SIZE=$((MIMO_LANGUAGE_END > MIMO_IMAGES_END ? MIMO_LANGUAGE_END : MIMO_IMAGES_END))
ALLOCATED_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))

if (( MIMO_IMAGES_OFFSET < MIMO_LANGUAGE_END && MIMO_LANGUAGE_OFFSET < MIMO_IMAGES_END )); then
    echo "ERROR: MIMO rank ranges overlap: language [${MIMO_LANGUAGE_OFFSET}, ${MIMO_LANGUAGE_END}), images [${MIMO_IMAGES_OFFSET}, ${MIMO_IMAGES_END})." >&2
    exit 2
fi
if (( MIMO_WORLD_SIZE > ALLOCATED_GPUS )); then
    echo "ERROR: MIMO layout uses ${MIMO_WORLD_SIZE} ranks but only ${ALLOCATED_GPUS} GPUs are allocated." >&2
    exit 2
fi
if (( MICRO_BATCH_SIZE % MIMO_LANGUAGE_DP != 0 )); then
    echo "ERROR: MIMO MBS (${MICRO_BATCH_SIZE}) must be divisible by language DP (${MIMO_LANGUAGE_DP})." >&2
    exit 2
fi
if (( MICRO_BATCH_SIZE % MIMO_IMAGES_DP != 0 )); then
    echo "ERROR: MIMO MBS (${MICRO_BATCH_SIZE}) must be divisible by images DP (${MIMO_IMAGES_DP})." >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
    echo "ERROR: GBS (${GLOBAL_BATCH_SIZE}) must be divisible by MIMO MBS (${MICRO_BATCH_SIZE})." >&2
    exit 2
fi

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image." >&2
    exit 1
fi

# --- Rendezvous (the finetune runner calls dist.init_process_group directly,
#     so MASTER_ADDR / MASTER_PORT must be provided explicitly) ----------------
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 40000))}"

# --- Environment exports for the per-task command -----------------------------
# NCCL_NVLS_ENABLE targets NVLink-Switch (NVLS) hardware; it is a no-op on other
# interconnects. Drop or retune these NCCL knobs for your cluster if needed.
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=1
export HTTPX_LOG_LEVEL=WARNING
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda,ignore::UserWarning:modelopt.torch"

export HF_MODEL DATASET_NAME SEQ_LENGTH TRAIN_ITERS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE
export LOG_INTERVAL WANDB_PROJECT RUN_NAME RUN_DIR PRETRAINED_CHECKPOINT
export MIMO_LANGUAGE_TP MIMO_LANGUAGE_PP MIMO_LANGUAGE_CP MIMO_LANGUAGE_DP MIMO_LANGUAGE_OFFSET
export MIMO_IMAGES_TP MIMO_IMAGES_PP MIMO_IMAGES_CP MIMO_IMAGES_DP MIMO_IMAGES_OFFSET
export MIMO_WORLD_SIZE GPUS_PER_NODE
export MASTER_ADDR MASTER_PORT
export SLURM_EXPORT_ENV=ALL

# --- Job banner ---------------------------------------------------------------
LANGUAGE_LOCAL_MBS=$((MICRO_BATCH_SIZE / MIMO_LANGUAGE_DP))
MIMO_NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))

echo "======================================"
echo "Qwen3.5-VL 27B MegatronMIMO SFT (non-colocated)"
echo "======================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Nodes:           ${SLURM_JOB_NUM_NODES}"
echo "GPUs/node:       ${GPUS_PER_NODE}"
echo "Active ranks:    ${MIMO_WORLD_SIZE}"
echo "Allocated GPUs:  ${ALLOCATED_GPUS}"
echo "Language:        TP=${MIMO_LANGUAGE_TP} PP=${MIMO_LANGUAGE_PP} CP=${MIMO_LANGUAGE_CP} DP=${MIMO_LANGUAGE_DP} offset=${MIMO_LANGUAGE_OFFSET}"
echo "Images:          TP=${MIMO_IMAGES_TP} PP=${MIMO_IMAGES_PP} CP=${MIMO_IMAGES_CP} DP=${MIMO_IMAGES_DP} offset=${MIMO_IMAGES_OFFSET}"
echo "Batch:           MBS=${MICRO_BATCH_SIZE}, GBS=${GLOBAL_BATCH_SIZE}, language-local MBS=${LANGUAGE_LOCAL_MBS}, num_microbatches=${MIMO_NUM_MICROBATCHES}"
echo "Dataset preset:  ${DATASET_NAME}"
echo "Sequence length: ${SEQ_LENGTH}"
echo "Train iters:     ${TRAIN_ITERS}"
echo "Checkpoint:      ${PRETRAINED_CHECKPOINT}"
echo "======================================"

# --- Per-task command ---------------------------------------------------------
# Each MPMD srun task runs this; SLURM_PROCID identifies the MIMO rank, and
# the language/images components are addressed by rank offset.
RUN_CMD=$(cat <<'EOF'
set -euo pipefail
cd /opt/Megatron-Bridge

export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${MIMO_WORLD_SIZE}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${GPUS_PER_NODE}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

cmd=(
    uv run --no-sync python
    examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py
    --hf-model "${HF_MODEL}"
    --dataset-name "${DATASET_NAME}"
    --seq-length "${SEQ_LENGTH}"
    --train-iters "${TRAIN_ITERS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --log-interval "${LOG_INTERVAL}"
    --experiment-root "${RUN_DIR}"
    --run-name "${RUN_NAME}"
    --checkpoint-dir "${RUN_DIR}"
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-exp-name "${RUN_NAME}"
    --component "language=tp=${MIMO_LANGUAGE_TP},pp=${MIMO_LANGUAGE_PP},cp=${MIMO_LANGUAGE_CP},dp=${MIMO_LANGUAGE_DP},rank_offset=${MIMO_LANGUAGE_OFFSET}"
    --component "images=tp=${MIMO_IMAGES_TP},pp=${MIMO_IMAGES_PP},cp=${MIMO_IMAGES_CP},dp=${MIMO_IMAGES_DP},rank_offset=${MIMO_IMAGES_OFFSET}"
)
"${cmd[@]}"
EOF
)

# --- MPMD packed srun ---------------------------------------------------------
# Spread MIMO_WORLD_SIZE active ranks across the allocated nodes,
# GPUS_PER_NODE tasks/node until exhausted. With the defaults (17 ranks, 3 nodes
# of 8 GPUs) this yields 8+8+1, leaving the last 7 GPUs unused; the packing
# adapts to whatever node count and GPUS_PER_NODE you allocate.
mapfile -t PACKED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")

srun_base=(
    srun -l
    --kill-on-bad-exit=1
    --distribution=block:block
    --export=ALL
    --container-image "${CONTAINER_IMAGE}"
)
if [ -n "$CONTAINER_MOUNTS" ]; then
    for mount in $CONTAINER_MOUNTS; do
        srun_base+=(--container-mounts="${mount}")
    done
fi

srun_cmd=("${srun_base[@]}")
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

echo "Packed MPMD srun layout:"
remaining_tasks="${MIMO_WORLD_SIZE}"
for node_index in "${!PACKED_NODES[@]}"; do
    if (( remaining_tasks <= 0 )); then
        break
    fi
    tasks_on_node="${GPUS_PER_NODE}"
    if (( remaining_tasks < GPUS_PER_NODE )); then
        tasks_on_node="${remaining_tasks}"
    fi
    echo "  ${PACKED_NODES[$node_index]}: ${tasks_on_node} task(s)"
    remaining_tasks=$((remaining_tasks - tasks_on_node))
done

"${srun_cmd[@]}"

echo "======================================"
echo "Job completed"
echo "======================================"
