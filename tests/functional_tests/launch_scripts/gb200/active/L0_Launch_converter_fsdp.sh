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
# L0 launcher for HuggingFace <-> Megatron-FSDP weight conversion roundtrip.
#
# Scope: Only exercises TP=1 parametrizations on a tiny Qwen3 MoE toy model.
# TP>1 cases depend on Megatron-LM commit 8cbc45b6e
# ("[M-FSDP] Fix Tensor Parallel mode detection", PR #3191, merged 2026-04-07).
# That fix is already included in the pinned 3rdparty/Megatron-LM submodule,
# but the TP code path is sensitive to MLM version drift, so TP>1 coverage
# lives in the offline validation matrix under docs/mfsdp-rl-v2/ rather than
# this CI test.
#
# Minimum required Megatron-LM: must contain commit 8cbc45b6e (PR #3191).

set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="0,1"

uv run coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/converter/test_hf_fsdp_conversion.py
coverage combine -q
