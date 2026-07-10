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

set -euo pipefail

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}
MODEL_NAME=${MODEL_NAME:-gpt-oss-20b}
HF_MODEL_ID_IMPORT=${HF_MODEL_ID_IMPORT:-openai/gpt-oss-20b}
HF_MODEL_ID_EXPORT=${HF_MODEL_ID_EXPORT:-unsloth/gpt-oss-20b-BF16}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/models/${MODEL_NAME}}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/${MODEL_NAME}-hf-export}

NUM_GPUS=${NUM_GPUS:-8}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

if (( TP * PP * EP != NUM_GPUS )); then
    echo "TP * PP * EP must equal NUM_GPUS for this DP=1 example." >&2
    exit 1
fi

run_distributed() {
    uv run python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" "$@"
}

# Import HF → Megatron
run_distributed examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model "$HF_MODEL_ID_IMPORT" \
    --megatron-path "$MEGATRON_MODEL_PATH" \
    --torch-dtype bfloat16 \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --trust-remote-code

# Export Megatron → HF
run_distributed examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model "$HF_MODEL_ID_EXPORT" \
    --megatron-path "${MEGATRON_MODEL_PATH}/iter_0000000" \
    --hf-path "$HF_EXPORT_PATH" \
    --torch-dtype bfloat16 \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP"

# Pure BF16 mapping validation is separate from the lossy MXFP4 source import.
run_distributed \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id "$HF_MODEL_ID_EXPORT" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --skip-save \
    --trust-remote-code
