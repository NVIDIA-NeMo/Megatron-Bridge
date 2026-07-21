#!/bin/bash
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

usage() {
    cat <<'USAGE'
Run Bridge/AutoBridge-backed concurrent async text generation.

Example:
  examples/inference/run_async_text_generation.sh --nproc 1 \
    --hf_model_path Qwen/Qwen2.5-1.5B \
    --dtype bf16 \
    --prompt "Megatron async inference is" \
    --prompt "Concurrent generation is"

Pass scripts/inference/async_text_generation.py arguments after --nproc.
USAGE
}

NPROC=1
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
    scripts/inference/async_text_generation.py "${ARGS[@]}"
