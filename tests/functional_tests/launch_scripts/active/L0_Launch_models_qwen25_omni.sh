#!/bin/bash
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

set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
export MEGATRON_BRIDGE_COVERAGE_ROOT="${MEGATRON_BRIDGE_COVERAGE_ROOT:-$REPO_ROOT}"

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [ "${GPU_COUNT}" -ge 2 ]; then
  export CUDA_VISIBLE_DEVICES="0,1"
else
  export CUDA_VISIBLE_DEVICES="0"
fi

uv run coverage run --data-file="${MEGATRON_BRIDGE_COVERAGE_ROOT}/.coverage" --source="${MEGATRON_BRIDGE_COVERAGE_ROOT}/" --parallel-mode -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/models/qwen_omni/test_qwen25_omni_conversion.py
coverage combine -q
