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

# Backport a Python-only TransformerEngine fix onto the installed package. The TE pin
# comes from Megatron-LM, so this carries a fix the pinned release does not include
# without moving that pin or rebuilding TE.
set -euo pipefail

patch_file="${1:?Usage: patch_transformer_engine.sh TE_PATCH}"

# A preserved base runtime ships its own TransformerEngine build; leave it untouched.
if [[ "${MBRIDGE_PRESERVE_BASE_RUNTIME:-False}" == "True" ]]; then
    echo "Base runtime preserved; not patching its TransformerEngine"
    exit 0
fi

# Locate the package without importing it: the import loads CUDA libraries.
site_packages="$(python - <<'PY'
import importlib.util
from pathlib import Path

print(Path(importlib.util.find_spec("transformer_engine").origin).parent.parent)
PY
)"

# A TE pin that already carries the fix reverses cleanly. Otherwise the patch must apply
# exactly, so a refactored TE fails the build instead of shipping silently unpatched.
if patch --reverse --force --dry-run --fuzz=0 --silent -d "$site_packages" -p1 < "$patch_file" > /dev/null 2>&1; then
    echo "TransformerEngine already includes $(basename "$patch_file"); skipping"
    exit 0
fi
patch --batch --forward --fuzz=0 -d "$site_packages" -p1 < "$patch_file"

# The installed version string does not change, so record the backport next to the code.
basename "$patch_file" >> "$site_packages/transformer_engine/MBRIDGE_DOCKER_PATCHES"
