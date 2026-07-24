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
# `HF_MODEL_PATH` must point at a *converted* Hugging Face checkpoint directory
# (config.json + weights), not the tokenizer-only `nvidia/megatron-bert-uncased-345m`
# Hub repo. See README.md for how to obtain one.
HF_MODEL_PATH=${HF_MODEL_PATH:-"${WORKSPACE}/models/${MODEL_NAME}-hf"}

uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model "$HF_MODEL_PATH" \
    --megatron-path "${WORKSPACE}/models/${MODEL_NAME}"

uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model "$HF_MODEL_PATH" \
    --megatron-path "${WORKSPACE}/models/${MODEL_NAME}/iter_0000000" \
    --hf-path "${WORKSPACE}/models/${MODEL_NAME}-hf-export"

uv run python -m torch.distributed.run --nproc_per_node=1 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id "$HF_MODEL_PATH" \
    --megatron-load-path "${WORKSPACE}/models/${MODEL_NAME}/iter_0000000" \
    --tp 1 --pp 1
