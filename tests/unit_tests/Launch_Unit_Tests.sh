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

# Independent verification: Hash key files (doesn't rely on .mcore_commit_sha)
echo ""
echo "🔍 Independent MCore fingerprint (source directory):"
cd /opt/Megatron-Bridge/3rdparty/Megatron-LM
if [ -f "megatron/core/__init__.py" ]; then
    INIT_HASH=$(sha256sum megatron/core/__init__.py | cut -d' ' -f1 | cut -c1-16)
    echo "   __init__.py hash:     ${INIT_HASH}"
fi
if [ -f "megatron/core/package_info.py" ]; then
    PKG_HASH=$(sha256sum megatron/core/package_info.py | cut -d' ' -f1 | cut -c1-16)
    echo "   package_info.py hash: ${PKG_HASH}"
fi
if [ -f "pyproject.toml" ]; then
    PROJ_HASH=$(sha256sum pyproject.toml | cut -d' ' -f1 | cut -c1-16)
    echo "   pyproject.toml hash:  ${PROJ_HASH}"
    
    VERSION=$(grep -E '^\s*version\s*=' pyproject.toml | head -1 | sed 's/.*version\s*=\s*"\([^"]*\)".*/\1/' || echo "unknown")
    echo "   Version string:       ${VERSION}"
fi

# Check for git metadata (rare but possible)
if [ -d ".git" ]; then
    GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    echo "   🎯 Git commit:        ${GIT_COMMIT}"
fi

cd /opt/Megatron-Bridge

echo "=================================================="
echo ""

CUDA_VISIBLE_DEVICES="0,1" uv run coverage run -a --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ -m pytest \
    --timeout=0.75 \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/unit_tests -m "not pleasefixme"
