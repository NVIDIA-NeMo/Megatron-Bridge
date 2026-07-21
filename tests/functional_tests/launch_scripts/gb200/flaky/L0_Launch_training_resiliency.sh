# CI_TIMEOUT=50
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

uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
  --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode \
  -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/training/test_nvrx_straggler.py

# HAS_MCORE_DEV_BRANCH: skip the ft_launcher inprocess-restart path on the Megatron-Core dev
# lane only. mcore dev pins nvidia-resiliency-ext to a git build whose ft_launcher imports
# mcp.types, which the resolved mcp prerelease drops (ModuleNotFoundError). The gitlink tracks
# main again after the dev branch's phase-B revert, so this runs normally on main.
BRIDGE_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo /opt/Megatron-Bridge)"
GITLINK="$(git -C "$BRIDGE_ROOT" rev-parse HEAD:3rdparty/Megatron-LM 2>/dev/null || true)"
DEV_COMMIT="$(tr -d '[:space:]' < "$BRIDGE_ROOT/.dev.commit" 2>/dev/null || true)"
MAIN_COMMIT="$(tr -d '[:space:]' < "$BRIDGE_ROOT/.main.commit" 2>/dev/null || true)"
if [ -n "$GITLINK" ] && [ "$GITLINK" = "$DEV_COMMIT" ] && [ "$DEV_COMMIT" != "$MAIN_COMMIT" ]; then
  HAS_MCORE_DEV_BRANCH=1
else
  HAS_MCORE_DEV_BRANCH=0
fi

if [ "$HAS_MCORE_DEV_BRANCH" != "1" ] && command -v ft_launcher >/dev/null 2>&1; then
  echo "ft_launcher found, running inprocess restart tests..."

  export TORCH_CPP_LOG_LEVEL="error"
  export GROUP_RANK=0

  uv run ft_launcher \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
    --nnodes=1 --nproc-per-node=2 \
    --ft-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
    --ft-rank_out_of_section_timeout=300 \
    --monitor-interval=5 --max-restarts=3 \
    --ft-restart-policy=any-failed \
    -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/test_groups/training/test_inprocess_restart.py
else
  echo "Skipping ft_launcher inprocess restart tests (HAS_MCORE_DEV_BRANCH=$HAS_MCORE_DEV_BRANCH)"
fi

coverage combine -q
