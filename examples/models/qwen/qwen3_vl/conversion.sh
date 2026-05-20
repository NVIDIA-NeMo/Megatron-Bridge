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

# Import HF → Megatron for dense model
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct

# Export Megatron → HF for dense model
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct-hf-export

# Round-trip validation for dense model
uv run python -m torch.distributed.run --nproc_per_node=4 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id Qwen/Qwen3-VL-8B-Instruct --tp 2 --pp 2

# Import HF → Megatron for MoE model
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --megatron-path ${WORKSPACE}/models/Qwen3-VL-30B-A3B-Instruct

# Export Megatron → HF for MoE model
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --megatron-path ${WORKSPACE}/models/Qwen3-VL-30B-A3B-Instruct/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Qwen3-VL-30B-A3B-Instruct-hf-export

# Round-trip validation for MoE model
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id Qwen/Qwen3-VL-30B-A3B-Instruct --ep 8
