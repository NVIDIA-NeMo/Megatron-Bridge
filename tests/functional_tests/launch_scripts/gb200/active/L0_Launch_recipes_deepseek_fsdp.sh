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
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NVLINK_DOMAIN_SIZE=4
export USE_MNNVL=1
# Run the canonical GB300 DeepSeek V3 FSDP recipe as a compact compatibility
# proxy on the four-GPU GB200 runner. The test only reduces topology and model
# size; FSDP, MXFP8, HybridEP, offload, and overlap settings come from the
# production performance recipe.
output_log_file="/tmp/test_deepseek_recipes_pretrain_perf_gb200.log"
uv run python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -s -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/test_groups/recipes/test_deepseek_recipes_pretrain_fsdp.py 2>&1 | tee -a $output_log_file
coverage combine -q

# Print compact bootstrap metrics on the first migrated run. Once the canonical
# recipe has a stable baseline, this command is given a dedicated golden JSON.
uv run python -m tests.functional_tests.test_groups.recipes.proxy_metrics \
  --log-path "$output_log_file"
