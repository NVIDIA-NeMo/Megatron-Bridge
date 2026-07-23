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

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_HOME=${MODEL_HOME:-${WORKSPACE}/models/nvidia}
HF_MODEL_PATH=${HF_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${MODEL_HOME}/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-megatron}

[ -n "${HF_HOME:-}" ] && export HF_HOME
[ -n "${UV_CACHE_DIR:-}" ] && export UV_CACHE_DIR

mkdir -p "$(dirname "$MEGATRON_MODEL_PATH")"

if [ -e "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt" ] || [ -e "${MEGATRON_MODEL_PATH}/latest_train_state.pt" ]; then
    echo "ERROR: target already contains a Megatron checkpoint: ${MEGATRON_MODEL_PATH}"
    exit 1
fi

echo "Nemotron 3 Ultra CPU import"
echo "HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH}"

./scripts/conversion/convert.sh import \
    --executor local \
    --device cpu \
    --hf-model "$HF_MODEL_PATH" \
    --megatron-path "$MEGATRON_MODEL_PATH" \
    --torch-dtype bfloat16
