# GPU_COUNT=x4
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

export CUDA_VISIBLE_DEVICES="0,1,2,3"
MEGATRON_BRIDGE_ROOT="/opt/Megatron-Bridge/"

# Run DeepSeek V3 perf recipe functional tests on 4 GPUs (expert_model_parallel_size=4)
# This script tests the GB300 proxy configuration (FSDP + EP=4 + MoE optimizations)
# to ensure perf-related features can run basic training without crashes
output_log_file="/tmp/test_deepseek_recipes_pretrain_perf_gb200.log"
uv run python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -s -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/test_groups/recipes/test_deepseek_recipes_pretrain_fsdp.py 2>&1 | tee -a $output_log_file
coverage combine -q

golden_values_path="$MEGATRON_BRIDGE_ROOT/tests/functional_tests/test_groups/recipes/golden_values/test_deepseek_recipes_pretrain_fsdp_gb200.json"
# Setting the threshold to 0.25 as the GPU utilization is flaky.
uv run python -m scripts.performance.utils.evaluate --log_paths $output_log_file --golden_values_path $golden_values_path --assets_dir /tmp \
  --model_family_name deepseek --model_recipe_name deepseek_fsdp_1node_gb200 --timing_threshold 0.25
