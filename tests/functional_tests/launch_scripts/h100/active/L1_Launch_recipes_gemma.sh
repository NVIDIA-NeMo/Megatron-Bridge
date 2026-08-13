#!/bin/bash
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

set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

REPO_ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)

export CUDA_VISIBLE_DEVICES="0,1"

# Run Gemma3 recipe functional tests on 2 GPUs
# This script tests recipe configurations with their default settings to ensure
# they can run basic training without crashes
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file="${REPO_ROOT}/.coverage" --source="${REPO_ROOT}" --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/test_groups/recipes/test_gemma3_recipes_pretrain.py

# The recipe constructors use a repository-local default output directory before
# the functional test redirects checkpoints and logs to tmp_path. Remove that
# test-owned residue before starting a separate pytest invocation whose unit-test
# fixture deliberately refuses to delete pre-existing experiment directories.
rm -rf -- "${REPO_ROOT}/NeMo_experiments" "${REPO_ROOT}/nemo_experiments"
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file="${REPO_ROOT}/.coverage" --source="${REPO_ROOT}" --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/unit_tests/models/gemma/test_gemma4_ple_sequence_parallel_distributed.py
coverage combine -q
