# CI_TIMEOUT=35
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

#!/bin/bash
set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="0,1"

# Run each scenario in fresh workers so their test-owned WORLD and NCCL groups
# are released at process exit instead of leaking into the next scenario.
for test_case in \
  "TestLocalCheckpointing::test_local_checkpoint_save_and_resume" \
  "TestLocalCheckpointing::test_local_checkpoint_save_resume_with_most_recent_k"; do
  uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode \
    -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    "tests/functional_tests/test_groups/training/test_local_checkpointing.py::${test_case}"
done

coverage combine -q
