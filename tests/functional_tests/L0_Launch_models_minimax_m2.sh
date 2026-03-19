# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

export CUDA_VISIBLE_DEVICES="0,1"

uv run coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/models/minimax_m2

# Functional tests execute bridge code in torch.distributed subprocesses, which bypasses the
# outer coverage process. Run an in-process import of the bridge module so that coverage xml
# always has at least one measured file even if COVERAGE_PROCESS_START files are absent.
uv run python -c "
import coverage
cov = coverage.Coverage(
    data_file='/opt/Megatron-Bridge/.coverage',
    data_suffix=True,
    config_file='/opt/Megatron-Bridge/pyproject.toml',
)
cov.start()
from megatron.bridge.models.minimax_m2 import MiniMaxM2Bridge  # noqa: F401
cov.stop()
cov.save()
" 2>/dev/null || true

coverage combine -q
