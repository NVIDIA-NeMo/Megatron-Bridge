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
MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE}/models/nemotron-3.5-lightning-megatron}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/models/nemotron-3.5-lightning-hf-export}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

usage() {
    echo "Usage: $0 {import|export}"
    echo ""
    echo "Environment: HF_MODEL, MEGATRON_PATH, HF_EXPORT_PATH, TP, PP, EP, ETP"
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    import)
        run_bridge_distributed examples/conversion/convert_checkpoints_multi_gpu.py import \
            --hf-model "${HF_MODEL}" \
            --megatron-path "${MEGATRON_PATH}" \
            --tp "${TP}" --pp "${PP}" --ep "${EP}" --etp "${ETP}" \
            --torch-dtype bfloat16 \
            --distributed-timeout-minutes 120
        ;;
    export)
        require_path MEGATRON_PATH
        run_bridge_distributed examples/conversion/convert_checkpoints_multi_gpu.py export \
            --hf-model "${HF_MODEL}" \
            --megatron-path "${MEGATRON_PATH}" \
            --hf-path "${HF_EXPORT_PATH}" \
            --tp "${TP}" --pp "${PP}" --ep "${EP}" --etp "${ETP}" \
            --torch-dtype bfloat16 \
            --distributed-timeout-minutes 120
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

