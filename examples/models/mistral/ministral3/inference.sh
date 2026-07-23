#!/usr/bin/env bash
# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
MODEL_SIZE=${MODEL_SIZE:-3B}

case "$MODEL_SIZE" in
    3B)
        DEFAULT_MODEL_NAME=Ministral-3-3B-Base-2512
        DEFAULT_TP=2
        ;;
    8B)
        DEFAULT_MODEL_NAME=Ministral-3-8B-Base-2512
        DEFAULT_TP=2
        ;;
    14B)
        DEFAULT_MODEL_NAME=Ministral-3-14B-Base-2512
        DEFAULT_TP=4
        ;;
    *)
        echo "Unsupported MODEL_SIZE=${MODEL_SIZE}; expected one of: 3B, 8B, 14B" >&2
        exit 2
        ;;
esac

HF_MODEL_ID=${HF_MODEL_ID:-mistralai/${DEFAULT_MODEL_NAME}}
HF_MODEL_BASENAME=${HF_MODEL_ID%/}
MODEL_NAME=${MODEL_NAME:-${HF_MODEL_BASENAME##*/}}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/models/${MODEL_NAME}/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/${MODEL_NAME}-hf-export}

TP=${TP:-$DEFAULT_TP}
PP=${PP:-1}
EP=${EP:-1}
ETP=${ETP:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$((TP * PP * EP))}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
PROMPT=${PROMPT:-"The image shows"}
IMAGE_PATH=${IMAGE_PATH:-"https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png"}

# Generate after converting the original HF checkpoint in memory.
uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path "$HF_MODEL_ID" \
    --image_path "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP"

# Generate from the imported Megatron checkpoint.
uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path "$MEGATRON_MODEL_PATH" \
    --image_path "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP"

# Generate after converting the exported HF checkpoint in memory.
uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path "$HF_EXPORT_PATH" \
    --image_path "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP"
