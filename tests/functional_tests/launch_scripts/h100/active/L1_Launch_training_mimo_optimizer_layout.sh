# CI_TIMEOUT=15
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MIMO_LAYOUT_OUTPUT=$(mktemp -d)
uv run python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
    --master-addr=127.0.0.1 --master-port=29517 \
    tests/functional_tests/training/mimo_optimizer_layout.py --output "$MIMO_LAYOUT_OUTPUT"
