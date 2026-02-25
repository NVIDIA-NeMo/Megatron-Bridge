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
MODEL_NAME=Qwen3.5-397B-A17B

# Make sure to upgrade to transformers >= 5.2.0
# uv add transformers>=5.2.0

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model Qwen/${MODEL_NAME} \
    --megatron-path ${WORKSPACE}/models/${MODEL_NAME} \
    --torch-dtype bfloat16

# DEBUG: compare with the 4 layer model
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "/lustre/fs1/portfolios/coreai/users/chcui/mbridge_home/models/Qwen/Qwen3.5-4layer-HF" \
    --megatron_model_path /lustre/fs1/portfolios/coreai/users/chcui/mbridge_home/models/Qwen/Qwen3.5-397B-A17B_4layer \
    --model_class "Qwen3_5MoeForConditionalGeneration" \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --tp 2 --pp 1 --ep 4

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model Qwen/${MODEL_NAME} \
    --megatron-path ${WORKSPACE}/models/${MODEL_NAME}/iter_0000000 \
    --hf-path ${WORKSPACE}/models/${MODEL_NAME}-hf-export

# Round-trip validation
# Note: Qwen3.5 is a large MoE model, adjust parallelism as needed
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id Qwen/${MODEL_NAME} --tp 1 --pp 2 --ep 4 --trust-remote-code
