# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
GTP_CHECKPOINT_TEST="${REPO_ROOT}/tests/functional_tests/test_groups/data/energon/test_gtp_checkpoint_state.py"

# The ordinary data tests create their own single-rank process groups.
CUDA_VISIBLE_DEVICES="0,1" uv run python -m coverage run -a --data-file="${REPO_ROOT}/.coverage" --source="${REPO_ROOT}" -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/functional_tests/test_groups/data -m "not pleasefixme" --ignore="${GTP_CHECKPOINT_TEST}"

# Exercise GTP/CP stream ownership and checkpoint resume with two real ranks.
CUDA_VISIBLE_DEVICES="0,1" uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m coverage run --parallel-mode --data-file="${REPO_ROOT}/.coverage" --source="${REPO_ROOT}" \
    -m pytest -o log_cli=true -o log_cli_level=INFO --disable-warnings -vs \
    "${GTP_CHECKPOINT_TEST}" -m "not pleasefixme"
uv run python -m coverage combine --append -q
