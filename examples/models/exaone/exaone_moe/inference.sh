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

# ==============================================================================
# K-EXAONE-236B-A23B Inference (Single Node)
#
# Loads the Hugging Face checkpoint, converts it to Megatron in memory, and
# generates text across 8 local GPUs using TP=1, PP=1, EP=8.
#
# This BF16 236B model is memory intensive. H200-class GPUs are recommended;
# inference on 8x H100 80GB may OOM because parameters, workspaces, and the KV
# cache must fit together.
#
# Usage:
#   uv sync
#   examples/models/exaone/exaone_moe/inference.sh
#
# Override generation settings:
#   PROMPT="대한민국의 수도는 어디인가요?" MAX_NEW_TOKENS=64 \
#       examples/models/exaone/exaone_moe/inference.sh
# ==============================================================================

set -euo pipefail

HF_MODEL_ID="${HF_MODEL_ID:-LGAI-EXAONE/K-EXAONE-236B-A23B}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-8}"
ETP="${ETP:-1}"

PROMPT="${PROMPT:-What is artificial intelligence?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PARALLEL_SIZE=$((TP * PP * EP))
if [[ "$MODEL_PARALLEL_SIZE" -ne "$NPROC_PER_NODE" ]]; then
    echo "ERROR: TP*PP*EP=$MODEL_PARALLEL_SIZE must equal NPROC_PER_NODE=$NPROC_PER_NODE."
    exit 1
fi
if ((128 % EP != 0)); then
    echo "ERROR: EP=$EP must divide K-EXAONE's 128 routed experts."
    exit 1
fi

uv run python -m torch.distributed.run \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/inference/text_generation.py \
    --hf-model-path "$HF_MODEL_ID" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp "$TP" \
    --pp "$PP" \
    --ep "$EP" \
    --etp "$ETP" \
    --use-legacy-generation \
    --trust-remote-code \
    --distributed-timeout-minutes "$DISTRIBUTED_TIMEOUT_MINUTES"
