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

# DeepSeek-V4 import + export with the Bridge.
#
# DSv4 currently requires TP=1; scale via expert parallelism (EP).
# The Bridge dispatches FP8 / MXFP4 dequantisation by tensor dtype, so the
# same script works for Flash, Flash-Base, Pro, and Pro-Base.
#
# Override defaults by exporting environment variables before running:
#   WORKSPACE: directory for converted Megatron checkpoints (default: /workspace)
#   MODEL_VARIANT: one of DeepSeek-V4-Flash, DeepSeek-V4-Flash-Base,
#                  DeepSeek-V4-Pro, DeepSeek-V4-Pro-Base
#                  (default: DeepSeek-V4-Flash-Base)
#   EP: expert-parallel size (default: 4 for Flash variants, 16 for Pro variants)

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash-Base}
HF_MODEL_ID="deepseek-ai/${MODEL_VARIANT}"

if [[ -z "${EP:-}" ]]; then
    case "${MODEL_VARIANT}" in
        DeepSeek-V4-Pro*) EP=16 ;;
        *)                EP=4 ;;
    esac
fi
TP=1
PP=1

MEGATRON_DIR="${WORKSPACE}/models/${MODEL_VARIANT}"
EXPORT_DIR="${WORKSPACE}/models/${MODEL_VARIANT}-hf-export"
ITER=iter_0000000

# 1) Import HF -> Megatron (FP8 / MXFP4 dequantised to bfloat16 in-flight)
uv run python -m torch.distributed.run --nproc_per_node=$((TP * PP * EP)) \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_DIR}" \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --torch-dtype bfloat16 \
    --trust-remote-code

# 2) Compare HF and Megatron logits on a short prompt
uv run python -m torch.distributed.run --nproc_per_node=$((TP * PP * EP)) \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "${HF_MODEL_ID}" \
    --megatron_model_path "${MEGATRON_DIR}" \
    --prompt "Hello, how are you?" \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --trust-remote-code

# 3) Export Megatron -> HF (round-trip)
uv run python -m torch.distributed.run --nproc_per_node=$((TP * PP * EP)) \
    examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_DIR}/${ITER}" \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --torch-dtype bfloat16 \
    --hf-path "${EXPORT_DIR}" \
    --distributed-save \
    --trust-remote-code
