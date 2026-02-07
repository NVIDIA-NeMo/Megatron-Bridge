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

set -xeuo pipefail
# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Common configurations
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
MODEL_NAME=nemotron_3_nano
DATASET_NAME=mock
SEQ_LENGTH=512
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
EVAL_ITERS=10
LR=0.00005
MIN_LR=0.000005
LR_WARMUP_ITERS=5
LOG_INTERVAL=1
# TODO(liding)
WANDB_PROJECT=liding_nano3_2602_release
# WANDB_PROJECT=megatron-bridge-${DATASET_NAME}
export NCCL_DEBUG=INFO

# TP/PP combinations: "TP,PP"
PARALLELISM_CONFIGS=("8,1" "1,8")

for config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP <<< "$config"
    
    echo "Running pretraining with TP=$TP, PP=$PP"
    # TODO(liding): add uv run back
    # uv run 
    python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
        --recipe ${MODEL_NAME}_pretrain_config \
        model.seq_length=$SEQ_LENGTH \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        train.eval_iters=$EVAL_ITERS \
        optimizer.lr=$LR \
        optimizer.min_lr=$MIN_LR \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_pretrain_tp${TP}_pp${PP} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_sft_tp${TP}_pp${PP} \
        dataset.sequence_length=$SEQ_LENGTH \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP
done


        