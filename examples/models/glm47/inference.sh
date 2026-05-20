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
# GLM-4.7 / GLM-4.7-Flash Inference Examples
#
# GLM-4.7-Flash (MLA+MoE, ~30B, 64 experts top-4) fits on 8 GPUs with EP=8.
# GLM-4.7 (MoE, ~358B, 160 experts top-8) requires multi-node.
# ==============================================================================

set -xeuo pipefail

# GLM-4.7-Flash

GLM47_FLASH_HF="${GLM47_FLASH_HF:-zai-org/GLM-4.7-Flash}"
PROMPT="${PROMPT:-What is artificial intelligence?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
COORDINATOR_HOST="${COORDINATOR_HOST:-127.0.0.1}"

uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/inference/text_generation.py \
    --hf_model_path "$GLM47_FLASH_HF" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tp 1 --pp 1 --ep 8 \
    --use-coordinator \
    --coordinator-host "$COORDINATOR_HOST"
