# CI_TIMEOUT=30
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
MUON_TEST_OUTPUT=$(mktemp -d)
DRIVER=tests/functional_tests/training/muon_distributed_checkpoint.py
for test_case in muon_on muon_off muon_inherited muon_alias muon_resume muon_hash \
    adam_on adam_off adam_resume muon_builder_on muon_builder_off muon_builder_resume \
    muon_reshard_on muon_reshard_resume muon_legacy_on muon_legacy_resume; do
    uv run python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
        --master-addr=127.0.0.1 --master-port=29512 \
        "$DRIVER" --root "$MUON_TEST_OUTPUT" --case "$test_case"
done
uv run python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
    --master-addr=127.0.0.1 --master-port=29512 \
    "$DRIVER" --root "$MUON_TEST_OUTPUT" --case muon_dp2_save
uv run python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
    --master-addr=127.0.0.1 --master-port=29512 \
    "$DRIVER" --root "$MUON_TEST_OUTPUT" --case muon_dp1_resume
uv run python "$DRIVER" --root "$MUON_TEST_OUTPUT" --compare
