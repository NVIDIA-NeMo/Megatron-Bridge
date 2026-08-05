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
REPO_ROOT=$(cd -- "${EXAMPLE_DIR}/../../../.." && pwd)

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

run_bridge_python() {
    uv run --active --no-sync python "$@"
}

run_bridge_distributed() {
    if [[ -n "${SLURM_PROCID:-}" ]]; then
        run_bridge_python "$@"
        return
    fi

    local nproc_per_node=${NPROC_PER_NODE:-8}
    uv run --active --no-sync python -m torch.distributed.run \
        --nproc_per_node="${nproc_per_node}" "$@"
}

require_path() {
    local variable_name=$1
    local path_value=${!variable_name:-}
    if [[ -z "${path_value}" ]]; then
        echo "ERROR: ${variable_name} must be set." >&2
        exit 2
    fi
    if [[ ! -e "${path_value}" ]]; then
        echo "ERROR: ${variable_name} does not exist: ${path_value}" >&2
        exit 2
    fi
}
