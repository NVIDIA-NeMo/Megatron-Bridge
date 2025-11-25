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
echo "🧪 UNIT TESTS - MCore Commit Information"
echo "=================================================="

# Check if the commit SHA file exists
MCORE_SHA_FILE="/opt/Megatron-Bridge/.mcore_commit_sha"
echo "🔍 Checking for file: ${MCORE_SHA_FILE}"

if [ -f "${MCORE_SHA_FILE}" ]; then
    MCORE_COMMIT=$(cat "${MCORE_SHA_FILE}")
    echo "✅ File exists!"
    echo "📦 MCore commit SHA from file: ${MCORE_COMMIT}"
    echo "📏 SHA length: ${#MCORE_COMMIT} characters"
    
    # Verify it's a valid SHA format (40 hex characters)
    if [[ "${MCORE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "✅ Valid SHA-1 hash format"
    else
        echo "⚠️  WARNING: Does not match SHA-1 format (expected 40 hex chars)"
    fi
else
    echo "❌ File NOT found: ${MCORE_SHA_FILE}"
    echo "⚠️  MCore commit: Unknown (image built before commit tracking was added)"
    echo "📂 Listing /opt/Megatron-Bridge directory:"
    ls -la /opt/Megatron-Bridge/ | grep -E "^\.|mcore" || echo "   (no .mcore files found)"
fi

# Additional debugging: Check if MCore is installed
echo ""
echo "📍 MCore installation path:"
ls -ld /opt/Megatron-Bridge/3rdparty/Megatron-LM 2>/dev/null || echo "   Directory not found"

echo "=================================================="
echo ""

CUDA_VISIBLE_DEVICES="0,1" uv run coverage run -a --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ -m pytest \
    --timeout=0.75 \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/unit_tests -m "not pleasefixme"
