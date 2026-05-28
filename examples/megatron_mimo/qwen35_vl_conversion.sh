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
# whose names must match the route table declared by the registered
# Qwen3.5-VL MIMO adapter
# (src/megatron/bridge/models/megatron_mimo/conversion/adapters/qwen35_vl.py).
#
# Usage:
#   bash examples/megatron_mimo/qwen35_vl_conversion.sh
#
# Override defaults via environment variables, e.g.:
#   MODEL_NAME=Qwen3.5-0.8B LANGUAGE_TP=2 VISION_TP=1 NPROC_PER_NODE=2 \
#     bash examples/megatron_mimo/qwen35_vl_conversion.sh
#
# Mirrors the non-MIMO Qwen3.5-VL entry point at
#   examples/models/qwen/qwen35_vl/conversion.sh
# so the two converters present a consistent user surface.

set -xeuo pipefail

# Workspace directory for checkpoints and results.
WORKSPACE=${WORKSPACE:-/workspace}

# Supported dense Qwen3.5-VL variants. MoE variants
# (Qwen3.5-35B-A3B / 122B-A10B / 397B-A17B) are out of v1 scope.
MODEL_NAME=${MODEL_NAME:-Qwen3.5-0.8B}

case "${MODEL_NAME}" in
    Qwen3.5-0.8B|Qwen3.5-2B|Qwen3.5-4B|Qwen3.5-9B|Qwen3.5-27B)
        ;;
    *)
        echo "Unsupported MODEL_NAME=${MODEL_NAME}." \
             "MIMO v1 supports dense variants only: Qwen3.5-{0.8B,2B,4B,9B,27B}."
        exit 1
        ;;
esac

# Per-component parallelism for the MIMO model. The component names
# `language` and `images` are the canonical keys declared by the
# Qwen3.5-VL MIMO adapter's route table — deviating from them here would
# cause validate_route_table to reject the run.
LANGUAGE_TP=${LANGUAGE_TP:-1}
VISION_TP=${VISION_TP:-1}

# torchrun world size. Default 2: two ranks colocated across the two
# components; minimum that exercises NCCL paths in the orchestrator.
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}

# Import HF → MIMO. When MEGATRON_PATH is set, the MIMO dist-checkpoint
# is saved to disk (torch_dist format, one iter_0000000 subdirectory per
# run); otherwise the converted model is discarded on process exit.
MEGATRON_PATH=${MEGATRON_PATH:-"${WORKSPACE}/${MODEL_NAME}-mimo"}

uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
    examples/conversion/convert_megatron_mimo.py import \
        --hf-model "Qwen/${MODEL_NAME}" \
        --megatron-path "${MEGATRON_PATH}" \
        --component "language=tp=${LANGUAGE_TP}" \
        --component "images=tp=${VISION_TP}" \
        --torch-dtype "${TORCH_DTYPE}"

# Export MIMO → HF.
HF_PATH=${HF_PATH:-"${WORKSPACE}/${MODEL_NAME}-mimo-export-hf"}

uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
    examples/conversion/convert_megatron_mimo.py export \
        --hf-model "Qwen/${MODEL_NAME}" \
        --megatron-path "${MEGATRON_PATH}" \
        --hf-path "${HF_PATH}" \
        --component "language=tp=${LANGUAGE_TP}" \
        --component "images=tp=${VISION_TP}" \
        --torch-dtype "${TORCH_DTYPE}"
