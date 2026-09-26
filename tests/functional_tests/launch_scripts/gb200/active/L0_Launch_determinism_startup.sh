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

# CI_TIMEOUT=15
#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0

# Each case launches its own fresh single-rank worker; do not wrap pytest in torchrun.
uv run python -m pytest -v -s -x --tb=short \
  tests/functional_tests/test_groups/training/test_determinism_startup.py
