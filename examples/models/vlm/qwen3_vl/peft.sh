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

# Test Seq Packing configurations for LoRA finetuning on the dense model
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/Qwen3-VL-8B-Instruct
MODEL_NAME=qwen3_vl_8b
DATASET_NAME=cord_v2
SEQ_LENGTH=4096
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=2
EVAL_ITERS=10
LR=0.00005
MIN_LR=0.000005
LR_WARMUP_ITERS=10
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

SEQ_PACKING_CONFIGS=(True False)

# EP/TP/PP/CP combinations: "EP,TP,PP,CP" configurations
PARALLELISM_CONFIGS=("1,1,1,1" "1,1,1,2" "1,1,1,4")

for pack_config in "${SEQ_PACKING_CONFIGS[@]}"; do
    for par_config in "${PARALLELISM_CONFIGS[@]}"; do
        IFS=',' read -r EP TP PP CP <<< "$par_config"
        echo "Running LoRA finetuning pack_sequences_in_batch=$pack_config with EP=$EP TP=$TP PP=$PP CP=$CP"
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
            checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_lora_seq_pack_${pack_config}_cp${CP} \
            logger.log_interval=$LOG_INTERVAL \
            logger.wandb_project=$WANDB_PROJECT \
            logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_lora_seq_pack_${pack_config}_cp${CP} \
            dataset.maker_name=make_${DATASET_NAME}_dataset \
            dataset.seq_length=$SEQ_LENGTH \
            dataset.pack_sequences_in_batch=$pack_config \
            model.expert_model_parallel_size=$EP \
            model.tensor_model_parallel_size=$TP \
            model.pipeline_model_parallel_size=$PP \
            model.context_parallel_size=$CP \
            model.calculate_per_token_loss=True \
            ddp.average_in_collective=False \
            ddp.grad_reduce_in_fp32=True
    done
done


# Test Seq Packing configurations for LoRA finetuning on the MoE model
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/Qwen3-VL-30B-A3B-Instruct
MODEL_NAME=qwen3_vl_30b_a3b
DATASET_NAME=cord_v2
SEQ_LENGTH=4096
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=2
EVAL_ITERS=10
LR=0.00005
MIN_LR=0.000005
LR_WARMUP_ITERS=10
LOG_INTERVAL=1
WANDB_PROJECT=megatron-bridge-${DATASET_NAME}

SEQ_PACKING_CONFIGS=(True False)

# EP/TP/PP/CP combinations: "EP,TP,PP,CP" configurations
PARALLELISM_CONFIGS=("8,1,1,1" "4,1,1,2" "2,1,1,4")

for pack_config in "${SEQ_PACKING_CONFIGS[@]}"; do
    for par_config in "${PARALLELISM_CONFIGS[@]}"; do
        IFS=',' read -r EP TP PP CP <<< "$par_config"
        echo "Running LoRA finetuning pack_sequences_in_batch=$pack_config with EP=$EP TP=$TP PP=$PP CP=$CP"
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
            checkpoint.save=${WORKSPACE}/results/${MODEL_NAME}_lora_seq_pack_${pack_config}_ep${EP}_cp${CP} \
            logger.log_interval=$LOG_INTERVAL \
            logger.wandb_project=$WANDB_PROJECT \
            logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_lora_seq_pack_${pack_config}_ep${EP}_cp${CP} \
            dataset.maker_name=make_${DATASET_NAME}_dataset \
            dataset.seq_length=$SEQ_LENGTH \
            dataset.pack_sequences_in_batch=$pack_config \
            model.expert_model_parallel_size=$EP \
            model.tensor_model_parallel_size=$TP \
            model.pipeline_model_parallel_size=$PP \
            model.context_parallel_size=$CP \
            model.calculate_per_token_loss=True \
            ddp.average_in_collective=False \
            ddp.grad_reduce_in_fp32=True
    done
done
