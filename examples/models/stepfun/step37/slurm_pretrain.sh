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
# Step3.7 SFT (Flickr8k) — multi-node SLURM launch (PP=4, EP=8)
#
# Parallelism: TP=1 PP=4 EP=8 ETP=1 CP=1  ->  DP = world_size / 4.
# 45 decoder layers are split 12 / 12 / 12 / 9 across the 4 pipeline stages.
#
# Usage:
#   sbatch --requeue slurm_pretrain.sh
#
# Required env:
#   CONTAINER_IMAGE        enroot/squashfs container image
#
# Optional env (sensible defaults shown):
#   WORKSPACE              base dir for checkpoints / logs   (default: /workspace)
#   MEGATRON_BRIDGE_PATH   Megatron-Bridge clone             (default: /opt/Megatron-Bridge)
#   CONTAINER_MOUNTS       space-separated host:container mounts
#   HF_MODEL               HF id or local snapshot           (default: stepfun-ai/Step-3.7-Flash)
#   PRETRAINED_CHECKPOINT  converted Megatron checkpoint dir (.../iter_0000000)
#   SFT_OUTPUT             dir to save the SFT checkpoint
# ==============================================================================

#SBATCH --job-name=step37-sft-flickr8k
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --account=my_account
#SBATCH --output=step37_sft_%j.out
#SBATCH --error=step37_sft_%j.err
#SBATCH --exclusive

set -euo pipefail

# ─── Config ─────────────────────────────────────────────────────────────────
WORKSPACE=${WORKSPACE:-/workspace}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-/opt/Megatron-Bridge}
HF_MODEL=${HF_MODEL:-stepfun-ai/Step-3.7-Flash}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/checkpoints/step37_megatron/iter_0000000}
SFT_OUTPUT=${SFT_OUTPUT:-${WORKSPACE}/checkpoints/step37_sft}

TEST_NAME=${TEST_NAME:-step37_sft_flickr8k_pp4_ep8}
LOG_DIR=${LOG_DIR:-${WORKSPACE}/logs}
mkdir -p "${LOG_DIR}"

# Recipe knobs (override via env at submit time).
SEQ_LENGTH=${SEQ_LENGTH:-1024}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
TRAIN_ITERS=${TRAIN_ITERS:-50}
SAMPLE_COUNT=${SAMPLE_COUNT:-640}   # >= global_batch_size x 10 packs
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge-step37}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
if [ -z "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

# ─── Environment ──────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export MASTER_PORT=${MASTER_PORT:-29500}

NUM_NODES=${SLURM_JOB_NUM_NODES:-8}
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}

# ─── Recipe CLI overrides ─────────────────────────────────────────────────────
CLI_OVERRIDES="\
    model.seq_length=${SEQ_LENGTH} \
    model.pipeline_model_parallel_size=4 \
    model.num_layers_in_first_pipeline_stage=12 \
    model.num_layers_in_last_pipeline_stage=9 \
    model.tensor_model_parallel_size=1 \
    model.expert_model_parallel_size=8 \
    model.expert_tensor_parallel_size=1 \
    model.context_parallel_size=1 \
    model.sequence_parallel=False \
    model.variable_seq_lengths=True \
    model.moe_token_dispatcher_type=alltoall \
    model.moe_permute_fusion=True \
    model.mtp_num_layers=0 \
    model.recompute_granularity=full \
    model.recompute_method=uniform \
    model.recompute_num_layers=1 \
    model.freeze_language_model=True \
    model.freeze_vision_model=False \
    model.freeze_vision_projection=False \
    dataset.sample_count=${SAMPLE_COUNT} \
    dataset.max_packing_seqlen=${SEQ_LENGTH} \
    dataset.seq_length=${SEQ_LENGTH} \
    dataset.tokenizer_path=${HF_MODEL} \
    train.micro_batch_size=1 \
    train.global_batch_size=${GLOBAL_BATCH_SIZE} \
    train.train_iters=${TRAIN_ITERS} \
    optimizer.lr=${LR} \
    optimizer.min_lr=${MIN_LR} \
    scheduler.lr_warmup_iters=10 \
    checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT} \
    checkpoint.save=${SFT_OUTPUT} \
    logger.tensorboard_dir=${LOG_DIR}/tensorboard/${TEST_NAME}/ \
    logger.wandb_project=${WANDB_PROJECT} \
    logger.wandb_exp_name=${TEST_NAME} \
    logger.log_interval=1 \
    tokenizer.padded_vocab_size=128896"

# ─── Launch (one srun task per GPU = one rank) ────────────────────────────────
read -r -d '' INNER_SCRIPT <<'EOF' || true
set -euo pipefail
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}

cd "${MEGATRON_BRIDGE_PATH}"
uv run --no-sync python scripts/training/run_recipe.py \
    --recipe step37_flickr8k_sft_smoke_config \
    --step_func step37_flickr8k_step \
    ${CLI_OVERRIDES}
EOF

SRUN_CMD="srun --mpi=pmix --container-image=${CONTAINER_IMAGE}"
for mount in ${CONTAINER_MOUNTS}; do
    SRUN_CMD="${SRUN_CMD} --container-mounts=${mount}"
done

${SRUN_CMD} \
    --nodes=${NUM_NODES} \
    --gpus-per-node=${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --output=${LOG_DIR}/log.${TEST_NAME}_%j.out \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --no-container-mount-home \
    bash -c "export MEGATRON_BRIDGE_PATH=\"${MEGATRON_BRIDGE_PATH}\" HF_MODEL=\"${HF_MODEL}\" MASTER_PORT=\"${MASTER_PORT}\" CLI_OVERRIDES=\"${CLI_OVERRIDES}\"; ${INNER_SCRIPT}"
