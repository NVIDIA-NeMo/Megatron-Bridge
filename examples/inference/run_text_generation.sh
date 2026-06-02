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
Run Bridge/AutoBridge-backed synchronous offline text generation.

Example:
  examples/inference/run_text_generation.sh --nproc 1 \
    --hf_model_path meta-llama/Llama-3.2-1B \
    --prompt "Megatron Bridge inference is" \
    --max_new_tokens 32

Pass any scripts/inference/text_generation.py argument after --nproc.
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
    scripts/inference/text_generation.py "${ARGS[@]}"
