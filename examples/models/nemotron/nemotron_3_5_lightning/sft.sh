#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "${EXAMPLE_DIR}/_common.sh"

HARDWARE=${HARDWARE:-h100}
WORKSPACE=${WORKSPACE:-/workspace}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE}/results/nemotron-3.5-lightning-sft-${HARDWARE}}
TRAIN_ITERS=${TRAIN_ITERS:-100}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
SAVE_INTERVAL=${SAVE_INTERVAL:-${TRAIN_ITERS}}

require_path PRETRAINED_CHECKPOINT

case "${HARDWARE}" in
    h100)
        recipe=nemotron_3_5_lightning_sft_openmathinstruct2_packed_config
        ;;
    gb200)
        recipe=nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config
        ;;
    *)
        echo "ERROR: HARDWARE must be h100 or gb200, got: ${HARDWARE}" >&2
        exit 2
        ;;
esac

run_bridge_distributed scripts/training/run_recipe.py \
    --recipe "${recipe}" \
    checkpoint.pretrained_checkpoint="${PRETRAINED_CHECKPOINT}" \
    checkpoint.save="${OUTPUT_DIR}" \
    checkpoint.save_interval="${SAVE_INTERVAL}" \
    train.train_iters="${TRAIN_ITERS}" \
    train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    scheduler.lr_warmup_iters="$((TRAIN_ITERS < 10 ? TRAIN_ITERS : 10))" \
    scheduler.lr_decay_iters="${TRAIN_ITERS}" \
    logger.log_interval=1 \
    logger.tensorboard_dir=null \
    "$@"
