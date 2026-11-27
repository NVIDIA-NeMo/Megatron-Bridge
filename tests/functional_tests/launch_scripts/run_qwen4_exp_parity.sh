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

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${REPO_ROOT}"

# Keep REPO_ROOT first: Megatron-LM also ships a regular `tests` package, which would
# otherwise shadow tests.functional_tests.test_groups in the torchrun child processes.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"

uv run --no-sync python -c '
import logging
import torch
import transformers
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
logging.basicConfig(level=logging.INFO)
logging.info("Qwen4-Exp reference: Transformers %s, Torch %s", transformers.__version__, torch.__version__)
'

uv run --no-sync python -m coverage run \
  --data-file="${REPO_ROOT}/.coverage" --source="${REPO_ROOT}" --parallel-mode -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x --tb=short -rA \
  tests/functional_tests/test_groups/models/qwen/test_qwen4_exp_conversion.py "$@"
uv run --no-sync python -m coverage combine -q
