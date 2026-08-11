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
MODEL_NAME=megatron-bert-uncased-345m
HF_MODEL_PATH=${HF_MODEL_PATH:-"${WORKSPACE}/models/${MODEL_NAME}-hf"}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-"${WORKSPACE}/models/${MODEL_NAME}-hf-export"}
TEXT=${TEXT:-"Paris is the [MASK] of France."}

# `MegatronBertForMaskedLM` has no `generate()` method, so inference here means
# fill-mask prediction, not text generation. This only exercises Hugging Face
# code paths, so it can be used to sanity-check both the original checkpoint
# and the round-tripped export from conversion.sh.
uv run python examples/models/bert/megatron-bert/fill_mask.py \
    --hf_model_path "$HF_MODEL_PATH" \
    --text "$TEXT"

if [ -d "$HF_EXPORT_PATH" ]; then
    uv run python examples/models/bert/megatron-bert/fill_mask.py \
        --hf_model_path "$HF_EXPORT_PATH" \
        --text "$TEXT"
fi
