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
# DeepSeek-V4 Parameter-Efficient Fine-Tuning (PEFT / LoRA)
#
# DSv4 PEFT freezes the backbone and trains LoRA adapters; the frozen base model
# (~290 GB FP8 dequantised to bf16 for Flash) still has to live across ranks, so
# expert parallelism remains necessary even though only adapter params are
# updated. Defaults below assume 1 node of 4xB200 (4 GPUs total) for Flash;
# raise EP / nodes for Pro variants.
#
# DSv4 currently requires TP=1.
#
# Usage:
#   1. Edit #SBATCH directives for your cluster
#   2. Set CONTAINER_IMAGE to your container path
#   3. Set MODEL_VARIANT (default: DeepSeek-V4-Flash)
#   4. Set PRETRAINED_CHECKPOINT to the imported Megatron checkpoint
#   5. Submit: sbatch slurm_peft.sh
# ==============================================================================

#SBATCH --job-name=deepseek-v4-peft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --account=my_account
#SBATCH --output=logs/deepseek_v4_peft_%j.out
#SBATCH --error=logs/deepseek_v4_peft_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash}
HF_MODEL_ID="deepseek-ai/${MODEL_VARIANT}"

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/${MODEL_VARIANT}}

MODEL_NAME=deepseek_v4
RECIPE_NAME="${RECIPE_NAME:-${MODEL_NAME}_peft_config}"
PEFT_SCHEME=${PEFT_SCHEME:-lora}    # "lora" or "dora"
DATASET_NAME=${DATASET_NAME:-squad}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
TRAIN_ITERS=${TRAIN_ITERS:-1000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
EVAL_ITERS=${EVAL_ITERS:-32}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-1}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge-${DATASET_NAME}}

# Parallelism configs: "TP,PP,EP,CP,SP" per entry. DSv4 requires TP=1.
# Default for Flash on 1x4-GPU node: TP=1, PP=1, EP=4. The base model still
# needs EP > 1 even for PEFT because the frozen backbone is too large for one
# GPU. For Pro variants, raise EP and node count.
PARALLELISM_CONFIGS_DEFAULT=("1,1,4,1,False")
PARALLELISM_CONFIGS=("${PARALLELISM_CONFIGS[@]:-${PARALLELISM_CONFIGS_DEFAULT[@]}}")

CONTAINER_IMAGE=""
CONTAINER_MOUNTS=""

# ==============================================================================
# Environment
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# export UV_CACHE_DIR="/path/to/shared/uv_cache"
# export HF_HOME="/path/to/shared/HF_HOME"
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "DeepSeek-V4 PEFT (${MODEL_VARIANT}, ${PEFT_SCHEME})"
echo "======================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NUM_NODES"
echo "GPUs:     $SLURM_GPUS_PER_NODE / node"
echo "Recipe:   $RECIPE_NAME"
echo "PEFT:     $PEFT_SCHEME"
echo "Configs:  ${PARALLELISM_CONFIGS[*]}"
echo "======================================"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP CP SP <<< "$CONFIG"
    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    if [ "$TP" != "1" ]; then
        echo "ERROR: DSv4 requires TP=1; got TP=$TP."
        exit 1
    fi
    echo ""
    echo "Config $CONFIG_INDEX/${#PARALLELISM_CONFIGS[@]}: TP=$TP PP=$PP EP=$EP CP=$CP SP=$SP"

    CLI_OVERRIDES=" \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        validation.eval_interval=$EVAL_INTERVAL \
        validation.eval_iters=$EVAL_ITERS \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_${MODEL_VARIANT}_peft_${PEFT_SCHEME}_pp${PP}_ep${EP}_cp${CP} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_peft_${PEFT_SCHEME}_pp${PP}_ep${EP}_cp${CP} \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP \
        model.expert_model_parallel_size=$EP \
        model.expert_tensor_parallel_size=1 \
        model.sequence_parallel=$SP \
        model.context_parallel_size=$CP \
        model.calculate_per_token_loss=True \
        dataset.packed_sequence_specs.pad_seq_to_mult=$([ "$CP" -gt 1 ] && echo $((CP * 2)) || echo 1) \
        dataset.packed_sequence_specs.packed_sequence_size=$SEQ_LENGTH \
        dataset.seq_length=$SEQ_LENGTH \
        model.seq_length=$SEQ_LENGTH \
    "
    CMD="uv run --no-sync python /opt/Megatron-Bridge/scripts/training/run_recipe.py"
    CMD="$CMD --mode finetune"
    CMD="$CMD --recipe ${RECIPE_NAME}"
    CMD="$CMD --peft_scheme ${PEFT_SCHEME}"
    CMD="$CMD $(echo "$CLI_OVERRIDES" | tr '\n' ' ' | sed 's/  \+/ /g')"

    echo "$CMD"
    $SRUN_CMD bash -c "$CMD"
    RUN_EXIT=$?
    if [ $RUN_EXIT -ne 0 ]; then
        echo "ERROR: config ($CONFIG) failed with exit $RUN_EXIT"
        exit $RUN_EXIT
    fi
done

echo "======================================"
echo "Done: ${#PARALLELISM_CONFIGS[@]} config(s) completed"
echo "======================================"
