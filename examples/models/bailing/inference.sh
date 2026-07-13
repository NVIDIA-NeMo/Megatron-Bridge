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
COORDINATOR_HOST=${COORDINATOR_HOST:-127.0.0.1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
PROMPT=${PROMPT:-"Hello, how are you?"}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/models/Ling-flash-2.0/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/Ling-flash-2.0-hf-export}

# Inference with Hugging Face checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/inference/text_generation.py \
    --hf_model_path inclusionAI/Ling-flash-2.0 \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp 1 --ep 8 \
    --use-coordinator \
    --coordinator-host "${COORDINATOR_HOST}" \
    --trust-remote-code

# Inference with imported Megatron checkpoints
if [ -d "$MEGATRON_MODEL_PATH" ]; then
    uv run python -m torch.distributed.run --nproc_per_node=8 scripts/inference/text_generation.py \
        --hf_model_path inclusionAI/Ling-flash-2.0 \
        --megatron_model_path "$MEGATRON_MODEL_PATH" \
        --prompt "$PROMPT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --tp 1 --ep 8 \
        --use-coordinator \
        --coordinator-host "${COORDINATOR_HOST}" \
        --trust-remote-code
fi

# Inference with exported HF checkpoints
if [ -d "$HF_EXPORT_PATH" ]; then
    uv run python -m torch.distributed.run --nproc_per_node=8 scripts/inference/text_generation.py \
        --hf_model_path "$HF_EXPORT_PATH" \
        --prompt "$PROMPT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --tp 1 --ep 8 \
        --use-coordinator \
        --coordinator-host "${COORDINATOR_HOST}" \
        --trust-remote-code
fi
