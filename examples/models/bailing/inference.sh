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

set -euo pipefail

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}
HF_MODEL_ID=${HF_MODEL_ID:-inclusionAI/${MODEL_NAME:-Ling-mini-2.0}}
MODEL_NAME=${MODEL_NAME:-${HF_MODEL_ID##*/}}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/models/${MODEL_NAME}/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/${MODEL_NAME}-hf-export}

TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-4}
ETP=${ETP:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$((TP * PP * EP))}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
PROMPT=${PROMPT:-"The capital of France is"}

# Inference with Hugging Face checkpoints
uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "$HF_MODEL_ID" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --trust-remote-code

# Inference with imported Megatron checkpoints
uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path "$MEGATRON_MODEL_PATH" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --trust-remote-code

# Inference with exported HF checkpoints
uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "$HF_EXPORT_PATH" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --trust-remote-code
