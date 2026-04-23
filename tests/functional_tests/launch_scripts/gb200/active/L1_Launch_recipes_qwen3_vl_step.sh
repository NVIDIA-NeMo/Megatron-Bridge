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

export CUDA_VISIBLE_DEVICES="0,1"

# L1 smoke coverage for the Qwen3-VL-specific forward step (`qwen3_vl_step`).
# The rest of the Qwen3-VL / Qwen3.5-VL recipe matrix lives in the flaky L2
# script; these two cases are promoted so regressions in `qwen3_vl_step` block
# CI on push to main and under `needs-more-tests`.

# Qwen3-VL finetune with qwen3_vl_step
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
  -m coverage run --data-file=/opt/Megatron-Bridge/.coverage \
  --source=/opt/Megatron-Bridge/ --parallel-mode \
  -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x \
  -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/recipes/test_qwen3_vl_recipes_finetune.py \
  -k "test_qwen3_vl_finetune_with_qwen3_vl_step"

# Qwen3.5-VL finetune with qwen3_vl_step
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
  -m coverage run --data-file=/opt/Megatron-Bridge/.coverage \
  --source=/opt/Megatron-Bridge/ --parallel-mode \
  -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x \
  -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/recipes/test_qwen35_vl_recipes_finetune.py \
  -k "test_sft_with_qwen3_vl_step"

coverage combine -q
