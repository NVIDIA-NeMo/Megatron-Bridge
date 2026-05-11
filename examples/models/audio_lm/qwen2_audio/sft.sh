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
#   WORKSPACE    — root dir for models/results (default: /workspace/Megatron-Bridge/examples/models/audio_lm/qwen2_audio)
LOG_FILE=./qwen2_audio_7b_asr.log
exec > >(tee "${LOG_FILE}") 2>&1

export TORCHDYNAMO_DISABLE=1

set -euo pipefail

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace/Megatron-Bridge/examples/models/audio_lm/qwen2_audio}
NPROC=${NPROC:-8}
HF_MODEL=${HF_MODEL:-Qwen/Qwen2-Audio-7B}

# Before training, make sure to set WANDB_API_KEY or disable wandb logging
# export WANDB_API_KEY=<your_wandb_api_key>
# export WANDB_MODE=disabled

# Common configurations
MODEL_NAME=qwen2_audio_7b
MEGATRON_CKPT_DIR=${WORKSPACE}/megatron_ckpts/${MODEL_NAME}

# Convert HF checkpoint to Megatron format if not already done
if [ ! -d "${MEGATRON_CKPT_DIR}/iter_0000000" ]; then
    echo "Converting HF model to Megatron format..."
    uv run --no-sync python examples/conversion/convert_checkpoints.py import \
        --hf-model ${HF_MODEL} \
        --megatron-path ${MEGATRON_CKPT_DIR}
fi
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${MEGATRON_CKPT_DIR}}
WANDB_PROJECT=megatron-bridge-${MODEL_NAME}

# Training hyperparameters
SEQ_LENGTH=16384
TRAIN_ITERS=11250
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=4
EVAL_INTERVAL=1000
EVAL_ITERS=10
LR=2e-5
MIN_LR=2e-6
LR_WARMUP_ITERS=5
SAVE_INTERVAL=1000
LOG_INTERVAL=1

# TP/PP combinations: "TP,PP"
PARALLELISM_CONFIGS=("1,1")

for par_config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP <<< "$par_config"
    echo "============================================================"
    echo "  run_recipe.py | TP=${TP}, PP=${PP}"
    echo "============================================================"
    uv run --no-sync python -m torch.distributed.run --nproc_per_node=${NPROC} scripts/training/run_recipe.py \
        --recipe qwen2_audio_7b_finetune_config \
        --step_func audio_lm_step \
        --hf_path ${HF_MODEL} \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        checkpoint.save=${WORKSPACE}/exp/${MODEL_NAME}_sft_tp${TP}_pp${PP} \
        checkpoint.save_interval=$SAVE_INTERVAL \
        checkpoint.save_optim=False \
        model.seq_length=$SEQ_LENGTH \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP \
        model.freeze_language_model=false \
        model.freeze_audio_model=false \
        model.freeze_audio_projection=false \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        validation.eval_interval=$EVAL_INTERVAL \
        validation.eval_iters=$EVAL_ITERS \
        optimizer.lr=$LR \
        optimizer.min_lr=$MIN_LR \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_asr_tp${TP}_pp${PP} \
        dataset.maker_name=make_default_audio_dataset \
        "dataset.maker_kwargs.path_or_dataset=yuekai/aishell" \
        "dataset.maker_kwargs.subset=train" \
        "dataset.maker_kwargs.split=test" \
        "+dataset.maker_kwargs.prompt='Detect the language and recognize the speech: <|zh|>'" \
        "dataset.val_maker_kwargs.subset=dev" \
        "dataset.val_maker_kwargs.split=test" \
        dataset.skip_test=true \
        dataset.pack_sequences_in_batch=true \
        rng.seed=42 \
        ddp.grad_reduce_in_fp32=false
done
