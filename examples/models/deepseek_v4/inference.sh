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

# DeepSeek-V4 text generation with the Bridge.
#
# Override defaults by exporting environment variables:
#   WORKSPACE: directory holding the imported Megatron checkpoint (default: /workspace)
#   MODEL_VARIANT: one of DeepSeek-V4-Flash, DeepSeek-V4-Flash-Base,
#                  DeepSeek-V4-Pro, DeepSeek-V4-Pro-Base
#   EP: expert-parallel size (default: 4 for Flash, 8 for Pro)
#   PP: pipeline-parallel size (default: 1 for Flash, 4 for Pro)
#   PROMPT: prompt string (default: "Explain hyper-connections in transformer models.")
#   NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN: optional HybridEP topology override
#   NVLINK_DOMAIN_SIZE: optional physical NVLink-domain size override
#   USE_MNNVL: optional HybridEP fabric override (0 or 1)
#
# Defaults below are for GB200 (192 GB). For H100 (80 GB) configs, see README.md.
# HybridEP auto-detects accessible ranks and fabric support when the topology
# overrides are unset. If set, the values must describe the actual deployment.

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash-Base}
HF_MODEL_ID="deepseek-ai/${MODEL_VARIANT}"
PROMPT=${PROMPT:-"Explain hyper-connections in transformer models."}

if [[ -z "${EP:-}" ]]; then
    case "${MODEL_VARIANT}" in
        DeepSeek-V4-Pro*) EP=8 ;;
        *)                EP=4 ;;
    esac
fi
if [[ -z "${PP:-}" ]]; then
    case "${MODEL_VARIANT}" in
        DeepSeek-V4-Pro*) PP=4 ;;
        *)                PP=1 ;;
    esac
fi
TP=1

# DSv4 hybrid attention does not support inference contexts yet, so use full-prefix generation.
# Inference directly from the HF checkpoint (Bridge dequantises in-flight).
uv run python -m torch.distributed.run --nproc_per_node=$((PP * EP)) \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "${HF_MODEL_ID}" \
    --prompt "${PROMPT}" \
    --max_new_tokens 100 \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --legacy-full-prefix \
    --trust-remote-code

# Inference from a previously-imported Megatron checkpoint (faster cold start).
MEGATRON_DIR="${WORKSPACE}/models/${MODEL_VARIANT}"
if [[ -d "${MEGATRON_DIR}/iter_0000000" ]]; then
    uv run python -m torch.distributed.run --nproc_per_node=$((PP * EP)) \
        examples/conversion/hf_to_megatron_generate_text.py \
        --hf_model_path "${HF_MODEL_ID}" \
        --megatron_model_path "${MEGATRON_DIR}" \
        --prompt "${PROMPT}" \
        --max_new_tokens 100 \
        --tp ${TP} --pp ${PP} --ep ${EP} \
        --legacy-full-prefix \
        --trust-remote-code
fi
