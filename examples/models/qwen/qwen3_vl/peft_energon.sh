#!/usr/bin/env bash
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

set -euo pipefail

# Prepare ENERGON_PATH with tutorials/data/energon/README.md or the Mantis
# converter before launching this example.
: "${ENERGON_PATH:?Set ENERGON_PATH to an indexed Energon dataset root}"

WORKSPACE=${WORKSPACE:-/workspace}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/Qwen3-VL-8B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE}/results/qwen3-vl-8b-energon-lora}
NUM_GPUS=${NUM_GPUS:-1}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-100}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
EVAL_ITERS=${EVAL_ITERS:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}
ENABLE_IN_BATCH_PACKING=${ENABLE_IN_BATCH_PACKING:-False}

# Set WANDB_API_KEY to log online, or keep the default disabled mode.
export WANDB_MODE=${WANDB_MODE:-disabled}

uv run python -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_energon_config \
  --step_func qwen3_vl_step \
  --mode lora \
  checkpoint.pretrained_checkpoint="${PRETRAINED_CHECKPOINT}" \
  checkpoint.load=null \
  checkpoint.save="${OUTPUT_DIR}/checkpoints" \
  train.train_iters="${TRAIN_ITERS}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  train.micro_batch_size="${MICRO_BATCH_SIZE}" \
  validation.eval_interval="${EVAL_INTERVAL}" \
  validation.eval_iters="${EVAL_ITERS}" \
  validation.eval_micro_batch_size="${MICRO_BATCH_SIZE}" \
  model.seq_length="${SEQ_LENGTH}" \
  dataset.path="${ENERGON_PATH}" \
  dataset.seq_length="${SEQ_LENGTH}" \
  dataset.micro_batch_size="${MICRO_BATCH_SIZE}" \
  dataset.enable_in_batch_packing="${ENABLE_IN_BATCH_PACKING}" \
  logger.log_interval=1
