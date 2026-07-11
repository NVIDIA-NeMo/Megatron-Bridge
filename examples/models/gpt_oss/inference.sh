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
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
PROMPT=${PROMPT:-"Hello, how are you?"}
HF_MODEL_ID=${HF_MODEL_ID:-openai/gpt-oss-20b}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/models/gpt-oss-20b/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/gpt-oss-20b-hf-export}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-}

NUM_GPUS=${NUM_GPUS:-8}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

if (( TP * PP * EP != NUM_GPUS )); then
    echo "TP * PP * EP must equal NUM_GPUS for this DP=1 example." >&2
    exit 1
fi

run_generation() {
    local hf_model_path=$1
    shift
    uv run python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" \
        scripts/inference/text_generation.py \
        --hf_model_path "$hf_model_path" \
        --prompt "$PROMPT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
        --dtype bf16 \
        --seed 0 \
        --top_k 1 \
        --use-legacy-generation \
        --attention-backend unfused \
        --trust-remote-code \
        "$@"
}

# Bridge-backed generation while loading and converting the complete HF weights.
run_generation "$HF_MODEL_ID"

# Generation from the imported Megatron checkpoint.
if [ -d "$MEGATRON_MODEL_PATH" ]; then
    run_generation "$HF_MODEL_ID" --megatron_model_path "$MEGATRON_MODEL_PATH"
fi

# Bridge-backed generation from the exported, unquantized HF checkpoint.
if [ -d "$HF_EXPORT_PATH" ]; then
    run_generation "$HF_EXPORT_PATH"
fi

# Optional generation from an SFT Megatron checkpoint.
if [ -n "$SFT_CHECKPOINT" ] && [ -d "$SFT_CHECKPOINT" ]; then
    run_generation "$HF_MODEL_ID" --megatron_model_path "$SFT_CHECKPOINT"
fi
