#!/bin/bash
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

# ==============================================================================
# GPT-OSS 20B SFT: Analysis+Final channel experiment with packed sequences
#
# CoT reasoning in <|channel|>analysis (thinking field), #### N in <|channel|>final.
# Uses packed sequences: ~10-15 examples per 4096-token packed sequence.
# 2500 steps with packing ≈ 4.8 epochs of 1M examples (~4.5h wall clock).
# ==============================================================================

#SBATCH --job-name=gpt-oss-sft-af
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --output=logs/gpt_oss_sft_af_%j.out
#SBATCH --error=logs/gpt_oss_sft_af_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

export WORKSPACE="${WORKSPACE:-/lustre/fsw/portfolios/coreai/users/weijiac/nemo_workspace}"
export WKDIR="${WKDIR:-/lustre/fsw/portfolios/coreai/users/weijiac}"

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/gpt-oss-20b-v2}
MODEL_NAME=gpt_oss_20b
RECIPE_NAME="gpt_oss_20b_sft_openmathinstruct2_thinking_packed_config"
DATASET_NAME=openmathinstruct2_gsm8k
SAVE_SUFFIX="${SAVE_SUFFIX:-_analysisFinalv2}"
SEQ_LENGTH=4096
TRAIN_ITERS=${TRAIN_ITERS:-1000}
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
EVAL_ITERS=32
EVAL_INTERVAL=50
LR_WARMUP_ITERS=250
SAVE_INTERVAL=${SAVE_INTERVAL:-125}
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

PARALLELISM_CONFIGS=("2,2,4,1,True")

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$WKDIR/sqsh/nemo_26.02.rc5.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre,$WKDIR/nemo_workspace/Megatron-Bridge:/opt/Megatron-Bridge,$WKDIR/nemo_workspace/Megatron-LM:/opt/megatron-lm}"

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export PYTHONUNBUFFERED=1

export HF_HOME="${WKDIR}/.cache/huggingface"
export NEMO_HOME="${WKDIR}/.cache/nemo"

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "GPT-OSS 20B SFT Analysis+Final Experiment"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Recipe: $RECIPE_NAME"
echo "Train iters: $TRAIN_ITERS  LR warmup: $LR_WARMUP_ITERS  Save every: $SAVE_INTERVAL"
echo "======================================"

mkdir -p logs

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP CP SP <<< "$CONFIG"

    SAVE_DIR="${WORKSPACE}/results/${MODEL_NAME}_${DATASET_NAME}_finetune_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP}${SAVE_SUFFIX}"
    if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "Resuming from existing checkpoint: $SAVE_DIR"
        CKPT_OVERRIDES="checkpoint.load=${SAVE_DIR} checkpoint.finetune=False"
    else
        echo "Starting fresh from pretrained checkpoint: $PRETRAINED_CHECKPOINT"
        CKPT_OVERRIDES="checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    fi

    CLI_OVERRIDES=" \
        $CKPT_OVERRIDES \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        validation.eval_interval=$EVAL_INTERVAL \
        validation.eval_iters=$EVAL_ITERS \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=${SAVE_DIR} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_entity=nvidia-nemo-fw-public \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_finetune_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP}${SAVE_SUFFIX} \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP \
        model.expert_model_parallel_size=$EP \
        model.expert_tensor_parallel_size=1 \
        model.sequence_parallel=$SP \
        model.context_parallel_size=$CP \
        model.calculate_per_token_loss=True \
        dataset.seq_length=$SEQ_LENGTH \
        model.seq_length=$SEQ_LENGTH \
        optimizer.lr=5e-6 \
        optimizer.min_lr=5e-7 \
        checkpoint.save_interval=$SAVE_INTERVAL \
        dist.distributed_timeout_minutes=90
    "

    CMD="uv run --no-sync python /opt/Megatron-Bridge/scripts/training/run_recipe.py"
    CMD="$CMD --recipe ${RECIPE_NAME}"
    CMD="$CMD --peft_scheme none"
    CMD="$CMD $(echo "$CLI_OVERRIDES" | tr '\n' ' ' | sed 's/  \+/ /g')"

    echo "Executing: $CMD"
    $SRUN_CMD bash -c "PYTHONUNBUFFERED=1 $CMD"
    RUN_EXIT=$?
    if [ $RUN_EXIT -ne 0 ]; then
        echo "ERROR: Config failed with exit code $RUN_EXIT"
        exit $RUN_EXIT
    fi
done

echo "======================================"
echo "Job completed"
echo "======================================"
