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

set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="0"

# Path to the local WAN HF checkpoint (Wan-AI/Wan2.1-T2V-1.3B-Diffusers).
# Download with:
#   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
#     --local-dir ${WAN_HF_CKPT_DIR} --local-dir-use-symlinks False
export WAN_HF_CKPT_DIR="${WAN_HF_CKPT_DIR:-/workspace/checkpoints/wan_hf/wan2.1}"

TEST_FILE="tests/functional_tests/diffusion/wan/test_wan_ckpt_conversion.py"
PYTEST_OPTS="-o log_cli=true -o log_cli_level=INFO -v -s -x -m 'not pleasefixme' --tb=short -rA"

# 1) HF -> Megatron
uv run coverage run \
  --data-file=/opt/Megatron-Bridge/.coverage \
  --source=/opt/Megatron-Bridge/ \
  --parallel-mode \
  -m pytest ${PYTEST_OPTS} \
  ${TEST_FILE}::TestWanCkptConversion::test_hf_to_megatron

coverage combine -q

# 2) Megatron -> HF
uv run coverage run \
  --data-file=/opt/Megatron-Bridge/.coverage \
  --source=/opt/Megatron-Bridge/ \
  --parallel-mode \
  -m pytest ${PYTEST_OPTS} \
  ${TEST_FILE}::TestWanCkptConversion::test_megatron_to_hf

coverage combine -q

# 3) Clean up artifacts
pytest ${PYTEST_OPTS} \
  ${TEST_FILE}::TestWanCkptConversion::test_remove_artifacts
