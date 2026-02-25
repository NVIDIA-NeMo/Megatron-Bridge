#!/usr/bin/env bash
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

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-3-4b-it \
    --megatron-path ${WORKSPACE}/models/gemma-3-4b-it

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model google/gemma-3-4b-it \
    --megatron-path ${WORKSPACE}/models/gemma-3-4b-it/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gemma-3-4b-it-hf-export

# Round-trip validation
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id google/gemma-3-4b-it --tp 2 --pp 2
