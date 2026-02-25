# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

echo "=================================================="
echo "🧪 UNIT TESTS"
echo "=================================================="

# Display MCore commit SHA if triggered from MCore CI
if [ -f "/opt/Megatron-Bridge/.mcore_commit_sha" ]; then
    echo "📦 MCore commit: $(cat /opt/Megatron-Bridge/.mcore_commit_sha)"
fi
echo ""

# Skip timeout on Azure runners because the machines are slower
TIMEOUT_ARG="--timeout=2"
if [[ "${GHA_RUNNER:-}" == *"azure"* ]]; then
    TIMEOUT_ARG=""
fi

CUDA_VISIBLE_DEVICES="0,1" uv run coverage run -a --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ -m pytest \
    $TIMEOUT_ARG \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/unit_tests -m "not pleasefixme"
