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
# Step-3.5-Flash Pretrain / Resume (PP=16, 2 nodes, 16 GPUs)
#
# Launches the `step35_196b_a11b_pretrain_config` recipe with the parallelism
# layout used for the alignment / resume run: PP=16 with thinned head/tail
# stages, TP=EP=CP=1, MTP disabled, sliding-window mask matching `layer_types`.
#
# Usage:
#   sbatch --requeue --parsable slurm_pretrain.sh
#
# Required env vars (typically exported in a wrapper or your shell profile):
#   WORKSPACE          base directory for checkpoints / logs (default: /workspace)
#   CONTAINER_IMAGE    enroot/squashfs container image
#   MEGATRON_BRIDGE_PATH  path to the Megatron-Bridge clone (default: /opt/Megatron-Bridge)
#
# Optional env vars:
#   PRETRAINED_CHECKPOINT  Megatron checkpoint produced by conversion.sh
#                          (default: ${WORKSPACE}/models/stepfun-ai/Step-3.5-Flash)
#   HF_HOME / HF_TOKEN / WANDB_API_KEY
# ==============================================================================

#SBATCH --job-name=step35-pretrain
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=step35_pretrain_%j.out
#SBATCH --error=step35_pretrain_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-/opt/Megatron-Bridge}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/stepfun-ai/Step-3.5-Flash}

TEST_NAME=${TEST_NAME:-step35_pretrain_pp16}
LOG_DIR=${LOG_DIR:-${WORKSPACE}/logs}
mkdir -p "${LOG_DIR}"

# Recipe defaults; override on the sbatch command line via env to tune.
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
TRAIN_ITERS=${TRAIN_ITERS:-20}
LOG_INTERVAL=${LOG_INTERVAL:-1}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge-step35}

# Container image (required)
CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
# Space-separated container mounts (e.g. "/data:/data /workspace:/workspace")
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}

if [ -z "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

# ==============================================================================
# Environment Setup
# ==============================================================================

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET_GDR_C2C=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NVLINK_DOMAIN_SIZE=72

# Translate Slurm env -> torch.distributed env (each srun task is one rank).
export MASTER_PORT=${MASTER_PORT:-29500}

# ==============================================================================
# Job Execution
# ==============================================================================

NUM_NODES=${SLURM_JOB_NUM_NODES:-8}
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}

echo "======================================"
echo "Step-3.5-Flash Pretrain (PP=16)"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${NUM_NODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Total GPUs: $((NUM_NODES * GPUS_PER_NODE))"
echo "Checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "Log dir: ${LOG_DIR}"
echo "======================================"

# CLI overrides forwarded to run_recipe.py. Mirrors the parallelism / sliding-
# window / MTP layout of the reference resume run.
#
# window_attn_skip_freq: 48-entry mask (one per main decoder layer) that tells
# the attention kernel which layers should skip the sliding-window restriction.
# It must stay in sync with the recipe's layer_types.
CLI_OVERRIDES="\
    model.seq_length=${SEQ_LENGTH} \
    model.pipeline_model_parallel_size=8 \
    model.num_layers_in_last_pipeline_stage=3 \
    model.tensor_model_parallel_size=1 \
    model.expert_model_parallel_size=8 \
    model.expert_tensor_parallel_size=1 \
    model.context_parallel_size=1 \
    model.sequence_parallel=False \
    model.variable_seq_lengths=True \
    model.moe_token_dispatcher_type=alltoall \
    model.moe_permute_fusion=False \
    model.mtp_num_layers=0 \
    model.attention_dropout=0.0 \
    model.rotary_percent=0.5 \
    model.window_size=[512,0] \
    model.window_attn_skip_freq=[0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1] \
    train.micro_batch_size=${MICRO_BATCH_SIZE} \
    train.global_batch_size=${GLOBAL_BATCH_SIZE} \
    train.train_iters=${TRAIN_ITERS} \
    train.eval_iters=0 \
    train.manual_gc=true \
    train.manual_gc_interval=100 \
    logger.tensorboard_dir=${LOG_DIR}/tensorboard/${TEST_NAME}/ \
    logger.wandb_save_dir=${LOG_DIR}/wandb/${TEST_NAME}/ \
    logger.wandb_project=${WANDB_PROJECT} \
    logger.wandb_exp_name=${TEST_NAME} \
    logger.log_interval=${LOG_INTERVAL} \
    logger.log_memory_to_tensorboard=true \
    logger.log_throughput_to_tensorboard=true \
    checkpoint.save_interval=0 \
    checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT} \
    checkpoint.finetune=true \
    checkpoint.exit_on_missing_checkpoint=true \
    checkpoint.load_optim=false \
    checkpoint.load_rng=false \
    checkpoint.fully_parallel_load=true \
    checkpoint.ckpt_assume_constant_structure=true"

# Inline script run on every srun task. Each task is one rank.
read -r -d '' INNER_SCRIPT <<'EOF' || true
set -euo pipefail
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
echo "[rank ${RANK}] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} LOCAL_RANK=${LOCAL_RANK}"

cd "${MEGATRON_BRIDGE_PATH}"
uv run --no-sync python scripts/training/run_recipe.py \
    --recipe step35_196b_a11b_pretrain_config \
    --step_func gpt_step \
    ${CLI_OVERRIDES}
EOF

SRUN_CMD="srun --mpi=pmix --container-image=${CONTAINER_IMAGE}"
if [ -n "${CONTAINER_MOUNTS}" ]; then
    for mount in ${CONTAINER_MOUNTS}; do
        SRUN_CMD="${SRUN_CMD} --container-mounts=${mount}"
    done
fi

${SRUN_CMD} \
    --nodes=${NUM_NODES} \
    --gpus-per-node=${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --output=${LOG_DIR}/log.${TEST_NAME}_%j.out \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --no-container-mount-home \
    bash -c "export CLI_OVERRIDES=\"${CLI_OVERRIDES}\"; export MEGATRON_BRIDGE_PATH=\"${MEGATRON_BRIDGE_PATH}\"; export MASTER_PORT=\"${MASTER_PORT}\"; ${INNER_SCRIPT}"

echo "======================================"
echo "Job completed"
echo "======================================"
