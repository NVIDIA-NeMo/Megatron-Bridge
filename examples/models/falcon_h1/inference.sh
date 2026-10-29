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

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_NAME=${MODEL_NAME:-Falcon-H1-0.5B-Instruct}
HF_MODEL_ID=${HF_MODEL_ID:-tiiuae/${MODEL_NAME}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
PROMPT=${PROMPT:-"What is artificial intelligence?"}

MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/models/${MODEL_NAME}/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/${MODEL_NAME}-hf-export}

uv run python -m torch.distributed.run --nproc_per_node=1 \
    scripts/inference/text_generation.py \
    --hf_model_path "$HF_MODEL_ID" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp 1 --pp 1 --ep 1 --etp 1 \
    --use-legacy-generation \
    --trust-remote-code

if [ -d "$MEGATRON_MODEL_PATH" ]; then
    uv run python -m torch.distributed.run --nproc_per_node=1 \
        scripts/inference/text_generation.py \
        --hf_model_path "$HF_MODEL_ID" \
        --megatron_model_path "$MEGATRON_MODEL_PATH" \
        --prompt "$PROMPT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --tp 1 --pp 1 --ep 1 --etp 1 \
        --use-legacy-generation \
        --trust-remote-code
fi

if [ -d "$HF_EXPORT_PATH" ]; then
    uv run python -m torch.distributed.run --nproc_per_node=1 \
        scripts/inference/text_generation.py \
        --hf_model_path "$HF_EXPORT_PATH" \
        --prompt "$PROMPT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --tp 1 --pp 1 --ep 1 --etp 1 \
        --use-legacy-generation \
        --trust-remote-code
fi
