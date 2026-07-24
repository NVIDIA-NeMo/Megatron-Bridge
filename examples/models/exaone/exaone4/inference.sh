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

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-/workspace}

MODEL_NAME=exaone-4.0-1.2b
HF_MODEL_ID=LGAI-EXAONE/EXAONE-4.0-1.2B
PROMPT=${PROMPT:-"Explain what checkpoint conversion is in one paragraph."}

uv run python -m torch.distributed.run --nproc_per_node=1 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "$HF_MODEL_ID" \
    --prompt "$PROMPT" \
    --max_new_tokens 64 \
    --tp 1 --pp 1 \
    --trust-remote-code

uv run python -m torch.distributed.run --nproc_per_node=1 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path "${WORKSPACE}/models/${MODEL_NAME}/iter_0000000" \
    --prompt "$PROMPT" \
    --max_new_tokens 64 \
    --tp 1 --pp 1 \
    --trust-remote-code

uv run python -m torch.distributed.run --nproc_per_node=1 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "${WORKSPACE}/models/${MODEL_NAME}-hf-export" \
    --prompt "$PROMPT" \
    --max_new_tokens 64 \
    --tp 1 --pp 1 \
    --trust-remote-code
