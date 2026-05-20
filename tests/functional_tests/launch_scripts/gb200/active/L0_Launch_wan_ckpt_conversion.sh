#!/bin/bash
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

set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="0"

TEST_FILE="tests/functional_tests/test_groups/diffusion/wan/test_wan_conversion.py"
# Run all conversion tests in a single invocation so the class-scoped fixtures
# (toy model creation, HF->Megatron import) are shared across tests.
uv run coverage run \
  --data-file=/opt/Megatron-Bridge/.coverage \
  --source=/opt/Megatron-Bridge/ \
  --parallel-mode \
  -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  ${TEST_FILE}::TestWanCheckpointConversion

coverage combine -q
