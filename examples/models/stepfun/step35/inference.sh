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
# Step-3.5-Flash Inference
#
# Run from an interactive one-node / 8-GPU environment. By default this loads the
# HF checkpoint and converts in memory. Set MEGATRON_MODEL_PATH to generate from a
# pre-converted Megatron checkpoint instead.
# ==============================================================================

set -xeuo pipefail

HF_MODEL=${HF_MODEL:-stepfun-ai/Step-3.5-Flash}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-}
PROMPT=${PROMPT:-"What is artificial intelligence?"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

EXTRA_ARGS=()
if [ -n "${MEGATRON_MODEL_PATH}" ]; then
    EXTRA_ARGS+=(--megatron_model_path "${MEGATRON_MODEL_PATH}")
fi

uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "${HF_MODEL}" \
    --prompt "${PROMPT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --tp "${TP}" --pp "${PP}" --ep "${EP}" --etp "${ETP}" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
