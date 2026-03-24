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

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Before training, make sure to set WANDB_API_KEY or disable wandb logging
# export WANDB_API_KEY=<your_wandb_api_key>
# export WANDB_MODE=disabled

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/Qwen3-0.6B}
MODEL_NAME=qwen3_600m
DATASET_NAME=squad
SEQ_LENGTH=131072
TRAIN_ITERS=100
GLOBAL_BATCH_SIZE=2
MICRO_BATCH_SIZE=1
EVAL_ITERS=10
EVAL_INTERVAL=30
LR_WARMUP_ITERS=10
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# TP=1, CP=8 — required for 128K context parallelism (8 GPUs minimum)
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe ${MODEL_NAME}_sft_128k_config \
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    validation.eval_iters=$EVAL_ITERS \
    validation.eval_interval=$EVAL_INTERVAL \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_128k_sft \
    checkpoint.load=${WORKSPACE}/results/${MODEL_NAME}_128k_sft \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_128k_sft
