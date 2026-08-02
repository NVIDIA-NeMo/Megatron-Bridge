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
#
# Qwen3.5-VL MegatronMIMO conversion entry script.
#
# Wraps the generic, model-agnostic CLI at
#   examples/conversion/convert_megatron_mimo.py
# with Qwen3.5-VL defaults: two MIMO components (`language` + `images`)
# whose names must match the route table derived from the Qwen3.5-VL bridge's
# `mimo_source_prefixes` and the provider's `modality_keys`
# (src/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py).
#
# Usage:
#   bash examples/megatron_mimo/qwen35_vl/conversion.sh
#
# Override defaults via environment variables, e.g.:
#   MODEL_NAME=Qwen3.5-27B LANGUAGE_TP=4 VISION_TP=1 \
#     bash examples/megatron_mimo/qwen35_vl/conversion.sh
#
# MoE variants need expert parallelism, e.g.:
#   MODEL_NAME=Qwen3.5-35B-A3B LANGUAGE_DP=4 LANGUAGE_EP=4 VISION_TP=1 \
#     bash examples/megatron_mimo/qwen35_vl/conversion.sh
#
# Mirrors the non-MIMO Qwen3.5-VL entry point at
#   examples/models/qwen/qwen35_vl/conversion.sh
# so the two converters present a consistent user surface.

set -xeuo pipefail

# Workspace directory for checkpoints and results.
WORKSPACE=${WORKSPACE:-/workspace}

# Supported Qwen3.5-VL variants: dense plus the MoE variants.
MODEL_NAME=${MODEL_NAME:-Qwen3.5-0.8B}

case "${MODEL_NAME}" in
    Qwen3.5-0.8B|Qwen3.5-2B|Qwen3.5-4B|Qwen3.5-9B|Qwen3.5-27B)
        ;;
    Qwen3.5-35B-A3B|Qwen3.5-122B-A10B|Qwen3.5-397B-A17B)
        ;;
    *)
        echo "Unsupported MODEL_NAME=${MODEL_NAME}." \
             "Supported: dense Qwen3.5-{0.8B,2B,4B,9B,27B} and MoE" \
             "Qwen3.5-{35B-A3B,122B-A10B,397B-A17B}."
        exit 1
        ;;
esac

# Per-component parallelism for the MIMO model. The component names
# `language` and `images` are the canonical keys declared by the
# Qwen3.5-VL MIMO adapter's route table — deviating from them here would
# cause validate_route_table to reject the run.
LANGUAGE_TP=${LANGUAGE_TP:-1}
VISION_TP=${VISION_TP:-1}
LANGUAGE_DP=${LANGUAGE_DP:-1}
VISION_DP=${VISION_DP:-1}
# Expert parallelism for the language component. MoE variants need EP > 1:
# with EP=1 every language rank holds a full copy of the expert weights, which
# does not fit for the MoE sizes. Encoder components must stay dense (EP=1),
# and the config requires LANGUAGE_TP * LANGUAGE_DP to be divisible by
# LANGUAGE_EP, so scale DP with EP (e.g. LANGUAGE_EP=4 with LANGUAGE_DP=4).
LANGUAGE_EP=${LANGUAGE_EP:-1}
LANGUAGE_RANK_OFFSET=${LANGUAGE_RANK_OFFSET:-0}
VISION_RANK_OFFSET=${VISION_RANK_OFFSET:-$((LANGUAGE_RANK_OFFSET + LANGUAGE_TP * LANGUAGE_DP))}

# torchrun world size. By default, allocate exactly the ranks covered by the
# explicit non-colocated component layout.
NPROC_PER_NODE=${NPROC_PER_NODE:-$((VISION_RANK_OFFSET + VISION_TP * VISION_DP))}

TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}

# Import HF → MIMO. When MEGATRON_PATH is set, the MIMO dist-checkpoint
# is saved to disk (torch_dist format, one iter_0000000 subdirectory per
# run); otherwise the converted model is discarded on process exit.
MEGATRON_PATH=${MEGATRON_PATH:-"${WORKSPACE}/${MODEL_NAME}-mimo"}

uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
    examples/conversion/convert_megatron_mimo.py import \
        --hf-model "Qwen/${MODEL_NAME}" \
        --megatron-path "${MEGATRON_PATH}" \
        --component "language=tp=${LANGUAGE_TP},dp=${LANGUAGE_DP},ep=${LANGUAGE_EP},rank_offset=${LANGUAGE_RANK_OFFSET}" \
        --component "images=tp=${VISION_TP},dp=${VISION_DP},rank_offset=${VISION_RANK_OFFSET}" \
        --torch-dtype "${TORCH_DTYPE}"

# Export MIMO → HF.
HF_PATH=${HF_PATH:-"${WORKSPACE}/${MODEL_NAME}-mimo-export-hf"}

uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
    examples/conversion/convert_megatron_mimo.py export \
        --hf-model "Qwen/${MODEL_NAME}" \
        --megatron-path "${MEGATRON_PATH}" \
        --hf-path "${HF_PATH}" \
        --component "language=tp=${LANGUAGE_TP},dp=${LANGUAGE_DP},ep=${LANGUAGE_EP},rank_offset=${LANGUAGE_RANK_OFFSET}" \
        --component "images=tp=${VISION_TP},dp=${VISION_DP},rank_offset=${VISION_RANK_OFFSET}" \
        --torch-dtype "${TORCH_DTYPE}"
