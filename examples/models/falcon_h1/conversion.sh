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
MODEL_NAME=${MODEL_NAME:-Falcon-H1-0.5B-Instruct}
HF_MODEL_ID=${HF_MODEL_ID:-tiiuae/${MODEL_NAME}}

MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/models/${MODEL_NAME}}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/${MODEL_NAME}-hf-export}

./scripts/conversion/convert.sh import \
    --hf-model "$HF_MODEL_ID" \
    --megatron-path "$MEGATRON_MODEL_PATH" \
    --torch-dtype bfloat16 \
    --trust-remote-code

./scripts/conversion/convert.sh export \
    --hf-model "$HF_MODEL_ID" \
    --megatron-path "${MEGATRON_MODEL_PATH}/iter_0000000" \
    --hf-path "$HF_EXPORT_PATH" \
    --trust-remote-code

# The config-only export path writes model config/weights, but not tokenizer
# artifacts. Copy them so $HF_EXPORT_PATH is loadable by AutoTokenizer for
# standalone generation checks.
uv run python - "$HF_MODEL_ID" "$HF_EXPORT_PATH" <<'PY'
import sys

from transformers import AutoTokenizer


hf_model_id, hf_export_path = sys.argv[1:]
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
tokenizer.save_pretrained(hf_export_path)
PY

uv run python -m torch.distributed.run --nproc_per_node=1 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id "$HF_MODEL_ID" \
    --megatron-load-path "${MEGATRON_MODEL_PATH}/iter_0000000" \
    --tp 1 --pp 1 --ep 1 --etp 1 \
    --trust-remote-code
