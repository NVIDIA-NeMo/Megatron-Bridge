#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "${EXAMPLE_DIR}/_common.sh"

HF_MODEL=${HF_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}
WORKSPACE=${WORKSPACE:-/workspace}
LORA_CHECKPOINT=${LORA_CHECKPOINT:-}
HF_ADAPTER_PATH=${HF_ADAPTER_PATH:-${WORKSPACE}/models/nemotron-3.5-lightning-lora-adapter}
HF_MERGED_PATH=${HF_MERGED_PATH:-${WORKSPACE}/models/nemotron-3.5-lightning-lora-merged}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

usage() {
    echo "Usage: $0 {export|verify|merge}"
    echo ""
    echo "export/verify require LORA_CHECKPOINT. merge requires HF_ADAPTER_PATH."
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    export)
        require_path LORA_CHECKPOINT
        run_bridge_distributed examples/conversion/adapter/export_adapter.py \
            --hf-model-path "${HF_MODEL}" \
            --lora-checkpoint "${LORA_CHECKPOINT}" \
            --output "${HF_ADAPTER_PATH}" \
            --dtype bf16 \
            --tp "${TP}" --pp "${PP}" --ep "${EP}" --etp "${ETP}"
        ;;
    verify)
        require_path LORA_CHECKPOINT
        require_path HF_ADAPTER_PATH
        run_bridge_distributed examples/conversion/adapter/verify_adapter.py \
            --hf-model-path "${HF_MODEL}" \
            --hf-adapter-path "${HF_ADAPTER_PATH}" \
            --lora-checkpoint "${LORA_CHECKPOINT}" \
            --tp "${TP}" --pp "${PP}" --ep "${EP}"
        ;;
    merge)
        require_path HF_ADAPTER_PATH
        run_bridge_python "${EXAMPLE_DIR}/merge_adapter.py" \
            --hf-model "${HF_MODEL}" \
            --adapter-path "${HF_ADAPTER_PATH}" \
            --output "${HF_MERGED_PATH}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

