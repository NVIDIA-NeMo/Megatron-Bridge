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
# Nemotron-3 Nano Omni - Full SFT on VALOR32K-AVQA (audio+video+text)
#
# Recipe: nemotron_omni_valor32k_sft_config (Energon shards)
#         Enables temporal video embedder (temporal_patch_dim=2,
#         dynamic_resolution=True, separate_video_embedder=True) + FastConformer
#         audio path.
# Default parallelism: TP=2, ETP=1, EP=8, CP=1, MBS=2, GBS=16, packed sequences,
#                      selective recompute
# Default layout:      2 nodes / 16 GPUs
#                      (min_gpus = PP * max(TP*CP, EP*ETP) = 1 * max(2, 8) = 8;
#                       16 GPUs runs DP=8 on dense / EDP=2 on MoE for throughput)
#
# Prerequisite:
#   Build the VALOR32K-AVQA Energon shards (one-time):
#     python examples/models/vlm/nemotron_3_omni/data/build_valor32k_avqa_shards.py \
#       --output_dir ${WORKSPACE}/datasets/valor32k_avqa
#
# Override TP/ETP/EP/CP/PACKED_SEQ via environment, e.g.:
#   TP=4 ETP=1 EP=4 CP=1 PACKED_SEQ=false sbatch slurm_sft_valor32k_avqa.sh
#
# Usage:
#   sbatch slurm_sft_valor32k_avqa.sh
# ==============================================================================

#SBATCH --job-name=nomni-sft-valor32k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=nomni_sft_valor32k_%j.out
#SBATCH --error=nomni_sft_valor32k_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
HF_MODEL_ID=${HF_MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}
MODEL_NAME=$(basename "$HF_MODEL_ID")

PRETRAINED_CHECKPOINT=${WORKSPACE}/models/${MODEL_NAME}
ENERGON_PATH=${ENERGON_PATH:-${WORKSPACE}/datasets/valor32k_avqa/energon}
RECIPE=nemotron_omni_valor32k_sft_config
DATASET_NAME=valor32k_avqa

# Parallelism / batching (override via env: TP=4 ETP=1 EP=4 CP=1 PACKED_SEQ=false sbatch ...)
# Note: this recipe always uses temporal_patch_dim=2 (set in the recipe itself).
TP=${TP:-2}
ETP=${ETP:-1}
EP=${EP:-8}
CP=${CP:-1}
PACKED_SEQ=${PACKED_SEQ:-true}

SEQ_LENGTH=4096
TRAIN_ITERS=${TRAIN_ITERS:-4000}
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=2
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_ITERS=${EVAL_ITERS:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# Container image (required) — use the NeMo 26.04 container or a local .sqsh copy
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="nvcr.io/nvidia/nemo:26.04"
# CONTAINER_IMAGE="/path/to/nemo_26.04.sqsh"

# Container mounts (optional, space-separated)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /workspace:/workspace"

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export HTTPX_LOG_LEVEL=WARNING
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda,ignore::UserWarning:modelopt.torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export HF_HOME="/path/to/shared/HF_HOME"
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"
# export WANDB_MODE=disabled

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Nemotron-3 Nano Omni - SFT (VALOR32K-AVQA)"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-N/A}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-N/A}"
echo "Recipe: $RECIPE"
echo "Checkpoint: $PRETRAINED_CHECKPOINT"
echo "Energon shards: $ENERGON_PATH"
echo "Parallelism: TP=$TP ETP=$ETP EP=$EP CP=$CP (packed=$PACKED_SEQ, temporal video tpd=2)"
echo "======================================"

RUN_TAG=tp${TP}etp${ETP}ep${EP}cp${CP}pack${PACKED_SEQ}_iter${TRAIN_ITERS}
OUTPUT_DIR=${WORKSPACE}/results/${RECIPE}_sft_${RUN_TAG}

CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    checkpoint.save=$OUTPUT_DIR/checkpoints \
    checkpoint.load=$OUTPUT_DIR/checkpoints \
    checkpoint.save_interval=$SAVE_INTERVAL \
    logger.tensorboard_dir=$OUTPUT_DIR/tb_logs \
    dataset.path=$ENERGON_PATH \
    dataset.seq_length=$SEQ_LENGTH \
    dataset.pack_sequences_in_batch=$PACKED_SEQ \
    model.seq_length=$SEQ_LENGTH \
    model.tensor_model_parallel_size=$TP \
    model.expert_tensor_parallel_size=$ETP \
    model.expert_model_parallel_size=$EP \
    model.context_parallel_size=$CP \
    model.sequence_parallel=True \
    model.recompute_granularity=selective \
    model.recompute_modules=[core_attn,mlp,layernorm,moe_act,moe] \
    model.freeze_language_model=False \
    model.freeze_vision_model=False \
    model.freeze_vision_projection=False \
    model.freeze_sound_encoder=False \
    model.freeze_sound_projection=False \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${RECIPE}_sft_${RUN_TAG}"

CMD="uv run --no-sync python scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --hf_path $HF_MODEL_ID \
    --step_func nemotron_omni_step \
    $CLI_OVERRIDES"

echo "Executing command..."
echo "======================================"

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"

if [ -n "$CONTAINER_MOUNTS" ]; then
    # pyxis --container-mounts is comma-separated; multiple flags would last-win
    MOUNTS_CSV=${CONTAINER_MOUNTS// /,}
    SRUN_CMD="$SRUN_CMD --container-mounts=$MOUNTS_CSV"
fi

$SRUN_CMD bash -c "$CMD"

echo "======================================"
echo "Job completed"
echo "======================================"
