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
#   bash sft.sh
#
# Environment variables:
#   WORKSPACE    — root dir for models/results (default: /workspace)
#   NPROC        — number of GPUs per node (default: 8)
#   HF_MODEL     — HuggingFace model path (default: Qwen/Qwen2-Audio-7B)
# ==============================================================================

LOG_FILE=./qwen2_audio_7b_asr.log
exec > >(tee "${LOG_FILE}") 2>&1


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
WANDB_PROJECT=megatron-bridge-${MODEL_NAME}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Run via finetune_qwen2_audio.py (YAML + CLI overrides)
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
        logger.wandb_exp_name=${MODEL_NAME}_asr_tp${TP}_pp${PP}
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
# TP/PP combinations to test: "TP,PP"
PARALLELISM_CONFIGS=("1,1")

for config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP <<< "$config"
    run_via_finetune_script "$TP" "$PP"
done
