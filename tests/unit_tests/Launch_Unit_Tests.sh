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

echo "=========================================="
echo "MCore Version Verification"
echo "=========================================="

# 1. What we INTENDED to install (from Docker build)
if [ -f "/opt/Megatron-Bridge/.mcore_commit_sha" ]; then
    EXPECTED_SHA=$(cat /opt/Megatron-Bridge/.mcore_commit_sha)
    echo "Expected MCore commit: ${EXPECTED_SHA}"
else
    EXPECTED_SHA="unknown"
    echo "Expected MCore commit: Unknown (no .mcore_commit_sha file)"
fi

# 2. What's ACTUALLY installed and verify it matches source
echo ""
echo "Installed MCore package verification:"
python3 << 'PYEOF'
import megatron.core
import hashlib
import os

# Show installation location
installed_path = megatron.core.__file__
print(f"  Installed: {installed_path}")

# Get version if available
version = getattr(megatron.core, '__version__', 'unknown')
print(f"  Version: {version}")

# Hash the installed __init__.py
def hash_file(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    return None

installed_hash = hash_file(installed_path)
print(f"  Installed __init__.py hash: {installed_hash}")

# Hash the source __init__.py (from 3rdparty/Megatron-LM)
source_init = "/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/__init__.py"
source_hash = hash_file(source_init)
if source_hash:
    print(f"  Source __init__.py hash:    {source_hash}")
    
    # Compare them
    if installed_hash == source_hash:
        print(f"  ✅ MATCH: Installed package matches source directory")
    else:
        print(f"  ❌ MISMATCH: Installed package differs from source!")
else:
    print(f"  ⚠️ Could not find source at {source_init}")
PYEOF

echo ""
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES="0,1" uv run coverage run -a --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ -m pytest \
    --timeout=0.75 \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/unit_tests -m "not pleasefixme"
