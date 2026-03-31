#!/usr/bin/env bash
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
# Qwen2-Audio 7B SFT (Supervised Fine-Tuning) Script
#
# Usage:
#   Method 1 — via run_recipe.py (generic entry point):
#     bash sft.sh
#
#   Method 2 — via finetune_qwen2_audio.py (model-specific, YAML overrides):
#     bash sft.sh --use-finetune-script
#
# Environment variables:
#   WORKSPACE    — root dir for models/results (default: /workspace)
#   NPROC        — number of GPUs per node (default: 8)
#   HF_MODEL     — HuggingFace model path (default: Qwen/Qwen2-Audio-7B-Instruct)
# ==============================================================================
export PYTHONPATH=/workspace_yuekai/asr/Megatron-Bridge:$PYTHONPATH
export TORCHDYNAMO_DISABLE=1

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace_yuekai/asr/Megatron-Bridge/examples/models/audio_lm/qwen2_audio}
NPROC=${NPROC:-8}
HF_MODEL=${HF_MODEL:-/workspace_yuekai/HF/Qwen2-Audio-7B}

# Before training, set WANDB_API_KEY or disable wandb logging
# export WANDB_API_KEY=<your_wandb_api_key>
# export WANDB_MODE=disabled

# Common configurations
MEGATRON_CKPT_DIR=${WORKSPACE}/megatron_ckpts/${MODEL_NAME:-qwen2_audio_7b}
MODEL_NAME=qwen2_audio_7b

# Convert HF checkpoint to Megatron format if not already done
if [ ! -d "${MEGATRON_CKPT_DIR}/iter_0000000" ]; then
    echo "Converting HF model to Megatron format..."
    uv run --no-sync python examples/conversion/convert_checkpoints.py import \
        --hf-model ${HF_MODEL} \
        --megatron-path ${MEGATRON_CKPT_DIR}
fi
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${MEGATRON_CKPT_DIR}}
DATASET_NAME=default_audio
SEQ_LENGTH=4096
TRAIN_ITERS=11250
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=4
EVAL_ITERS=0
LR=0.00002
MIN_LR=0.000002
LR_WARMUP_ITERS=5
LOG_INTERVAL=1
SAVE_INTERVAL=200
WANDB_PROJECT=megatron-bridge-${MODEL_NAME}

# Dataset maker kwargs
MAKER_NAME=make_default_audio_dataset
MAKER_DATASET=yuekai/aishell
MAKER_SPLIT=test
MAKER_PROMPT="Detect the language and recognize the speech: <|zh|>"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Method 1: via run_recipe.py (generic entry point, CLI overrides only)
# --------------------------------------------------------------------------
run_via_recipe() {
    local TP=$1
    local PP=$2

    echo "============================================================"
    echo "  run_recipe.py | TP=${TP}, PP=${PP}"
    echo "============================================================"

    uv run --no-sync torchrun --nproc_per_node=${NPROC} scripts/training/run_recipe.py \
        --recipe qwen2_audio_7b_finetune_config \
        --step_func audio_lm_step \
        checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT} \
        model.seq_length=${SEQ_LENGTH} \
        model.tensor_model_parallel_size=${TP} \
        model.pipeline_model_parallel_size=${PP} \
        model.freeze_language_model=false \
        model.freeze_audio_model=false \
        model.freeze_audio_projection=false \
        train.train_iters=${TRAIN_ITERS} \
        train.global_batch_size=${GLOBAL_BATCH_SIZE} \
        train.micro_batch_size=${MICRO_BATCH_SIZE} \
        train.eval_iters=${EVAL_ITERS} \
        optimizer.lr=${LR} \
        optimizer.min_lr=${MIN_LR} \
        scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
        checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_sft_tp${TP}_pp${PP} \
        checkpoint.save_interval=${SAVE_INTERVAL} \
        logger.log_interval=${LOG_INTERVAL} \
        logger.wandb_project=${WANDB_PROJECT} \
        logger.wandb_exp_name=${MODEL_NAME}_sft_tp${TP}_pp${PP} \
        dataset.maker_name=${MAKER_NAME} \
        dataset.seq_length=${SEQ_LENGTH}
}

# --------------------------------------------------------------------------
# Method 2: via finetune_qwen2_audio.py (YAML + CLI overrides)
# --------------------------------------------------------------------------
run_via_finetune_script() {
    local TP=$1
    local PP=$2

    echo "============================================================"
    echo "  finetune_qwen2_audio.py | TP=${TP}, PP=${PP}"
    echo "============================================================"

    uv run --no-sync torchrun --nproc_per_node=${NPROC} \
        ${SCRIPT_DIR}/finetune_qwen2_audio.py \
        --hf-model-path ${HF_MODEL} \
        --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} \
        --config-file ${SCRIPT_DIR}/conf/qwen2_audio_override_example.yaml \
        model.tensor_model_parallel_size=${TP} \
        model.pipeline_model_parallel_size=${PP} \
        checkpoint.save=${WORKSPACE}/exp/${MODEL_NAME}_sft_tp${TP}_pp${PP} \
        logger.wandb_project=${WANDB_PROJECT} \
        logger.wandb_exp_name=${MODEL_NAME}_sft_tp${TP}_pp${PP}
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
# USE_FINETUNE_SCRIPT=false
USE_FINETUNE_SCRIPT=true
for arg in "$@"; do
    case $arg in
        --use-finetune-script) USE_FINETUNE_SCRIPT=true ;;
    esac
done

# TP/PP combinations to test: "TP,PP"
PARALLELISM_CONFIGS=("1,1")

for config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP <<< "$config"

    if [ "$USE_FINETUNE_SCRIPT" = true ]; then
        run_via_finetune_script "$TP" "$PP"
    else
        run_via_recipe "$TP" "$PP"
    fi
done
