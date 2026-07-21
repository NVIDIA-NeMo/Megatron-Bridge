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

# Detect the unreleased Megatron-Core dev ref (submodule gitlink == .dev.commit and
# != .main.commit), mirroring tests/mcore_dev.py::HAS_MCORE_DEV_BRANCH. On the dev ref
# the pinned nvidia-resiliency-ext git build pulls the broken mcp 2.0.0b2 beta
# (mcp.types split into mcp-types), so ft_launcher crashes on import before pytest can
# run. Skip the inprocess-restart block on the dev lane; it stays active on main.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo /opt/Megatron-Bridge)"
HAS_MCORE_DEV_BRANCH=0
if [ -f "${REPO_ROOT}/.dev.commit" ] && [ -f "${REPO_ROOT}/.main.commit" ]; then
  _gitlink="$(git -C "${REPO_ROOT}" ls-tree HEAD -- 3rdparty/Megatron-LM 2>/dev/null | awk '{print $3}')"
  _dev="$(tr -d '[:space:]' < "${REPO_ROOT}/.dev.commit")"
  _main="$(tr -d '[:space:]' < "${REPO_ROOT}/.main.commit")"
  if [ -n "${_gitlink}" ] && [ "${_gitlink}" = "${_dev}" ] && [ "${_gitlink}" != "${_main}" ]; then
    HAS_MCORE_DEV_BRANCH=1
  fi
fi

uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
  --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode \
  -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/training/test_nvrx_straggler.py

if [ "${HAS_MCORE_DEV_BRANCH}" -eq 0 ] && command -v ft_launcher >/dev/null 2>&1; then
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
fi

coverage combine -q
