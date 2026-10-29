# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

REPO_ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "${REPO_ROOT}"
GTP_CHECKPOINT_TEST="${REPO_ROOT}/tests/functional_tests/test_groups/converter/test_gtp_checkpoint_conversion.py"
GTP_NATIVE_FP8_TEST="${REPO_ROOT}/tests/functional_tests/test_groups/converter/test_gtp_native_fp8_conversion.py"

export CUDA_VISIBLE_DEVICES="0,1"

# Existing converter tests launch their own distributed subprocesses when needed.
uv run python -m coverage run --data-file="${REPO_ROOT}/.coverage" --source="${REPO_ROOT}" --parallel-mode -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/converter \
  --ignore="${GTP_CHECKPOINT_TEST}" --ignore="${GTP_NATIVE_FP8_TEST}"

# GTP checkpoint and native FP8 conversion require two ranks in the pytest process.
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m coverage run --data-file="${REPO_ROOT}/.coverage" --source="${REPO_ROOT}" --parallel-mode \
  -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  "${GTP_CHECKPOINT_TEST}" "${GTP_NATIVE_FP8_TEST}"
uv run python -m coverage combine --append -q
