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

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_SIZE=${MODEL_SIZE:-3B}

case "$MODEL_SIZE" in
    3B)
        DEFAULT_MODEL_NAME=Ministral-3-3B-Base-2512
        DEFAULT_TP=2
        ;;
    8B)
        DEFAULT_MODEL_NAME=Ministral-3-8B-Base-2512
        DEFAULT_TP=2
        ;;
    14B)
        DEFAULT_MODEL_NAME=Ministral-3-14B-Base-2512
        DEFAULT_TP=4
        ;;
    *)
        echo "Unsupported MODEL_SIZE=${MODEL_SIZE}; expected one of: 3B, 8B, 14B" >&2
        exit 2
        ;;
esac

HF_MODEL_ID=${HF_MODEL_ID:-mistralai/${DEFAULT_MODEL_NAME}}
HF_MODEL_BASENAME=${HF_MODEL_ID%/}
MODEL_NAME=${MODEL_NAME:-${HF_MODEL_BASENAME##*/}}
MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE}/models/${MODEL_NAME}}
MEGATRON_LOAD_PATH=${MEGATRON_LOAD_PATH:-${MEGATRON_PATH}/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/${MODEL_NAME}-hf-export}
ROUNDTRIP_OUTPUT_DIR=${ROUNDTRIP_OUTPUT_DIR:-${WORKSPACE}/models/${MODEL_NAME}-roundtrip}

TP=${TP:-$DEFAULT_TP}
PP=${PP:-1}
EP=${EP:-1}
ETP=${ETP:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$((TP * PP * EP))}

# Import HF -> Megatron.
./scripts/conversion/convert.sh import \
    --hf-model "$HF_MODEL_ID" \
    --megatron-path "$MEGATRON_PATH"

# Export Megatron -> HF.
./scripts/conversion/convert.sh export \
    --hf-model "$HF_MODEL_ID" \
    --megatron-path "$MEGATRON_LOAD_PATH" \
    --hf-path "$HF_EXPORT_PATH"

# Multi-GPU verification of the imported checkpoint and HF export.
uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id "$HF_MODEL_ID" \
    --megatron-load-path "$MEGATRON_LOAD_PATH" \
    --output-dir "$ROUNDTRIP_OUTPUT_DIR" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP"
