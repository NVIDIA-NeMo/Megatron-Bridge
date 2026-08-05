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

HF_MODEL=${HF_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}
MEGATRON_PATH=${MEGATRON_PATH:-}
PROMPT=${PROMPT:-The capital of France is}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

args=(
    examples/conversion/hf_to_megatron_generate_text.py
    --hf_model_path "${HF_MODEL}"
    --prompt "${PROMPT}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --tp "${TP}"
    --pp "${PP}"
    --ep "${EP}"
    --etp "${ETP}"
)
if [[ -n "${MEGATRON_PATH}" ]]; then
    require_path MEGATRON_PATH
    args+=(--megatron_model_path "${MEGATRON_PATH}")
fi

run_bridge_distributed "${args[@]}"

