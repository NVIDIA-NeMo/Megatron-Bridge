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
# Nemotron Omni / Super 3.5 HF <-> MegatronMIMO conversion.
# Run from the repository root with a local HF checkpoint or HF repository ID:
#   HF_MODEL=/path/to/hf-checkpoint WORKSPACE=/workspace \
#     bash examples/megatron_mimo/nemotron_omni/conversion.sh
# Pass "import" or "export" to run only that phase; default is both.
# Defaults match the four-GPU Super 3.5 layout: image DP2, language TP2/EP2.
# The images component contains RADIO and its projector. Audio is not supported.

set -euo pipefail

: "${HF_MODEL:?Set HF_MODEL to the source HF checkpoint path or repository ID}"
PHASE=${1:-all}
case "${PHASE}" in
    import|export|all) ;;
    *) echo "Usage: $0 [import|export|all]" >&2; exit 1 ;;
esac

WORKSPACE=${WORKSPACE:-/workspace}
MEGATRON_PATH=${MEGATRON_PATH:-"${WORKSPACE}/nemotron-omni-mimo"}
HF_PATH=${HF_PATH:-"${WORKSPACE}/nemotron-omni-mimo-export-hf"}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}

LANGUAGE_TP=${LANGUAGE_TP:-2}
LANGUAGE_DP=${LANGUAGE_DP:-1}
LANGUAGE_EP=${LANGUAGE_EP:-2}
LANGUAGE_ETP=${LANGUAGE_ETP:-1}
VISION_TP=${VISION_TP:-1}
VISION_DP=${VISION_DP:-2}
LANGUAGE_RANKS=$((LANGUAGE_TP * LANGUAGE_DP))
VISION_RANKS=$((VISION_TP * VISION_DP))
NPROC_PER_NODE=${NPROC_PER_NODE:-$((LANGUAGE_RANKS + VISION_RANKS))}

# Import uses the verified layout: images first, then language.
if [[ "${PHASE}" == import || "${PHASE}" == all ]]; then
    uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
        examples/conversion/convert_megatron_mimo.py import \
        --hf-model "${HF_MODEL}" \
        --megatron-path "${MEGATRON_PATH}" \
        --component "images=tp=${VISION_TP},dp=${VISION_DP},rank_offset=0" \
        --component "language=tp=${LANGUAGE_TP},dp=${LANGUAGE_DP},ep=${LANGUAGE_EP},etp=${LANGUAGE_ETP},rank_offset=${VISION_RANKS}" \
        --torch-dtype "${TORCH_DTYPE}" \
        --trust-remote-code
fi

# Export places language on rank zero so its weights stream directly to the
# HF writer, avoiding a whole-language-component gather from another rank.
if [[ "${PHASE}" == export || "${PHASE}" == all ]]; then
    uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
        examples/conversion/convert_megatron_mimo.py export \
        --hf-model "${HF_MODEL}" \
        --megatron-path "${MEGATRON_PATH}" \
        --hf-path "${HF_PATH}" \
        --component "language=tp=${LANGUAGE_TP},dp=${LANGUAGE_DP},ep=${LANGUAGE_EP},etp=${LANGUAGE_ETP},rank_offset=0" \
        --component "images=tp=${VISION_TP},dp=${VISION_DP},rank_offset=${LANGUAGE_RANKS}" \
        --torch-dtype "${TORCH_DTYPE}" \
        --trust-remote-code
fi
