#!/usr/bin/env bash
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

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}
MODEL_NAME=Qwen3.5-397B-A17B # Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Qwen3.5-27B 
MODEL_PATH=Qwen/${MODEL_NAME}
# Make sure to upgrade to transformers >= 5.2.0
# uv add transformers>=5.2.0

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model ${MODEL_PATH} \
    --megatron-path ${WORKSPACE}/models/${MODEL_NAME} \
    --torch-dtype bfloat16

uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path ${MODEL_PATH} \
    --megatron_model_path ${WORKSPACE}/models/${MODEL_NAME} \
    --model_class "Qwen3_5MoeForConditionalGeneration" \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --tp 1 --pp 1 --ep 8

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model ${MODEL_PATH} \
    --megatron-path ${WORKSPACE}/models/${MODEL_NAME}/iter_0000000 \
    --hf-path ${WORKSPACE}/models/${MODEL_NAME}-hf-export

# Round-trip validation
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id Qwen/${MODEL_NAME} --tp 1 --pp 2 --ep 4 --trust-remote-code
