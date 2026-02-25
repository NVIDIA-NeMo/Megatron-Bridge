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

# Common configurations for dense model finetuning
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/Qwen3-VL-8B-Instruct
MODEL_NAME=qwen3_vl_8b
DATASET_NAME=cord_v2
SEQ_LENGTH=4096
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=1
EVAL_ITERS=10
LR=0.00005
MIN_LR=0.000005
LR_WARMUP_ITERS=10
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# TP/PP combinations: "TP,PP"
PARALLELISM_CONFIGS=("2,1" "1,2")

for config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP <<< "$config"
    
    echo "Running LoRA finetuning with TP=$TP, PP=$PP"
    uv run python -m torch.distributed.run --nproc_per_node=2 scripts/training/run_recipe.py \
        --recipe ${MODEL_NAME}_finetune_config \
        --step_func qwen3_vl_step \
        --peft_scheme lora \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        model.seq_length=$SEQ_LENGTH \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        train.eval_iters=$EVAL_ITERS \
        optimizer.lr=$LR \
        optimizer.min_lr=$MIN_LR \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_lora_tp${TP}_pp${PP} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_lora_tp${TP}_pp${PP} \
        dataset.maker_name=make_${DATASET_NAME}_dataset \
        dataset.seq_length=$SEQ_LENGTH \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP
done


# Common configurations for MoE model finetuning
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/Qwen3-VL-30B-A3B-Instruct
MODEL_NAME=qwen3_vl_30b_a3b
DATASET_NAME=cord_v2
SEQ_LENGTH=4096
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=1
EVAL_ITERS=10
LR=0.00005
MIN_LR=0.000005
LR_WARMUP_ITERS=10
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

# EP/TP/PP combinations: "EP,TP,PP" configurations
PARALLELISM_CONFIGS=("8,1,1" "4,1,1" "2,1,1")

for config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r EP TP PP <<< "$config"

    echo "Running LoRA finetuning with EP=$EP, TP=$TP, PP=$PP"
    uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
        --recipe ${MODEL_NAME}_finetune_config \
        --step_func qwen3_vl_step \
        --peft_scheme lora \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        model.seq_length=$SEQ_LENGTH \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        train.eval_iters=$EVAL_ITERS \
        optimizer.lr=$LR \
        optimizer.min_lr=$MIN_LR \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_lora_ep${EP}_tp${TP}_pp${PP}  \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_lora_ep${EP}_tp${TP}_pp${PP} \
        dataset.maker_name=make_${DATASET_NAME}_dataset \
        dataset.seq_length=$SEQ_LENGTH \
        model.expert_model_parallel_size=$EP \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP
done
