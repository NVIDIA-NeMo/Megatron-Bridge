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

# Run standard tests first (excluding inprocess restart tests)
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/training -k "not test_inprocess_restart"

# Run inprocess restart tests with ft_launcher if available
if command -v ft_launcher >/dev/null 2>&1; then
    echo "ft_launcher found, running inprocess restart tests..."
    
    # Set torch log level to reduce noise for inprocess restart tests
    export TORCH_CPP_LOG_LEVEL="error"
    
    ft_launcher \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
      --nnodes=1 --nproc-per-node=2 \
      --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
      --ft-param-rank_out_of_section_timeout=300 \
      --monitor-interval=5 --max-restarts=3 \
      --ft-restart-policy=min-healthy \
      -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
      tests/functional_tests/training/test_inprocess_restart.py
fi

coverage combine -q
