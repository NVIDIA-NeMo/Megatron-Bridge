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

# Full Gemma 4 26B-A4B-it VLM SFT uses the 8-GPU TP2/PP1/EP8/ETP1 recipe.
# The former single-node TP4/PP2 and TP8/PP1 examples kept all 128 experts on
# every data-parallel rank and ran out of H100 memory during the first update.
WORKSPACE=${WORKSPACE:-/workspace}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/gemma-4-26B-A4B-it}
SAVE_DIR=${SAVE_DIR:-${WORKSPACE}/results/gemma4_vl_26b_sft}

exec ./scripts/training/train.sh \
    --nodes 1 \
    --gpus-per-node 8 \
    --recipe gemma4_vl_26b_sft_config \
    --mode sft \
    --dataset cord-v2 \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --save_dir "${SAVE_DIR}"
