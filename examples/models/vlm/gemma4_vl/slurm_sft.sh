#!/bin/bash
set -euo pipefail
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# Gemma 4 VL 26B-A4B SFT fine-tuning.
#
# Usage (single node, 8 GPUs):
#   sbatch slurm_sft.sh
#
# Usage (multi-node, e.g. 2 nodes with EP=8):
#   NUM_NODES=2 TP=2 PP=1 EP=8 sbatch --nodes=2 slurm_sft.sh

#SBATCH --job-name=gemma4vl-sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=gemma4_vl_sft_%j.out
#SBATCH --error=gemma4_vl_sft_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/gemma-4-26B-A4B}
RECIPE=${RECIPE:-gemma4_vl_26b_sft_config}
PEFT_SCHEME=${PEFT_SCHEME:-}           # Set to "lora" to run LoRA instead of full SFT
DATASET_NAME=${DATASET_NAME:-cord_v2}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-40}   # 40 iters fits 30-min wall time with checkpoint save
GBS=${GBS:-32}
MBS=${MBS:-1}
EVAL_ITERS=${EVAL_ITERS:-5}      # 5 eval iters (was 10) reduces eval cost ~36s vs 72s
LR=${LR:-0.00005}
MIN_LR=${MIN_LR:-0.000005}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-10}
TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-8}   # Max EP with 16 GPUs, TP=2, PP=1: DP=8, EP=8 (EP must divide DP)
SAVE_DIR=${SAVE_DIR:-${WORKSPACE}/results/gemma4_vl_sft_tp${TP}_pp${PP}_ep${EP}_${SLURM_JOB_ID}}

# Container image (required)
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional)
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"
# CONTAINER_MOUNTS="/data:/data,/workspace:/workspace"

# ==============================================================================
# Environment
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
# Reduce fragmentation — helps when reserved-but-unallocated memory is ~1-2 GiB
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# export UV_CACHE_DIR="/path/to/shared/uv_cache"
# export HF_HOME="/path/to/shared/HF_HOME"
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"
# export WANDB_MODE=disabled

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=${MASTER_PORT:-29501}

# ==============================================================================
# Job
# ==============================================================================

PEFT_FLAG=""
if [[ -n "${PEFT_SCHEME}" ]]; then
    PEFT_FLAG="--peft_scheme ${PEFT_SCHEME}"
fi

echo "======================================"
echo "Gemma 4 VL 26B-A4B SFT"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Recipe: $RECIPE"
echo "PEFT: ${PEFT_SCHEME:-none}"
echo "TP=$TP PP=$PP EP=$EP"
echo "Checkpoint: $PRETRAINED_CHECKPOINT"
echo "Save: $SAVE_DIR"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "======================================"

# uv.lock pins transformers==5.3.0; gemma4 requires >=5.5.0. Upgrade from shared
# cache (pre-warmed) so each node gets the right version before training.
CMD="cd /opt/Megatron-Bridge && "
CMD="${CMD}uv pip install -q --upgrade 'transformers>=5.5.0' mistral_common && "
CMD="${CMD}export RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID WORLD_SIZE=\$SLURM_NTASKS && "
CMD="${CMD}uv run --no-sync python scripts/training/run_recipe.py"
CMD="${CMD} --recipe ${RECIPE}"
CMD="${CMD} --step_func vlm_step"
CMD="${CMD} ${PEFT_FLAG}"
CMD="${CMD} checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
CMD="${CMD} model.tensor_model_parallel_size=${TP}"
CMD="${CMD} model.pipeline_model_parallel_size=${PP}"
CMD="${CMD} model.expert_model_parallel_size=${EP}"
CMD="${CMD} model.seq_length=${SEQ_LENGTH}"
CMD="${CMD} train.train_iters=${TRAIN_ITERS}"
CMD="${CMD} train.global_batch_size=${GBS}"
CMD="${CMD} train.micro_batch_size=${MBS}"
CMD="${CMD} validation.eval_iters=${EVAL_ITERS}"
CMD="${CMD} optimizer.lr=${LR}"
CMD="${CMD} optimizer.min_lr=${MIN_LR}"
CMD="${CMD} scheduler.lr_warmup_iters=${LR_WARMUP_ITERS}"
CMD="${CMD} checkpoint.save=${SAVE_DIR}"
CMD="${CMD} dataset.maker_name=make_${DATASET_NAME}_dataset"
CMD="${CMD} dataset.seq_length=${SEQ_LENGTH}"

echo "Running training..."

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=${CONTAINER_IMAGE} --no-container-mount-home"

if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="${SRUN_CMD} --container-mounts=${CONTAINER_MOUNTS}"
fi

$SRUN_CMD bash -c "$CMD"

echo "======================================"
echo "Training job finished. EXIT=$?"
echo "======================================"
