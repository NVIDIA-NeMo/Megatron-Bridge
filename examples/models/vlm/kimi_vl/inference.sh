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

set -e

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_NAME=Kimi-K2.5
HF_MODEL=moonshotai/${MODEL_NAME}
TP=2
EP=4

# Inference with HF checkpoints (on-the-fly conversion)
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/models/vlm/kimi_vl/hf_to_megatron_generate_vlm.py \
    --hf_model_path ${HF_MODEL} \
    --trust_remote_code \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp ${TP} --ep ${EP}

# Inference with pre-converted Megatron checkpoint
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/models/vlm/kimi_vl/hf_to_megatron_generate_vlm.py \
    --hf_model_path ${HF_MODEL} \
    --megatron_model_path ${WORKSPACE}/${MODEL_NAME}/iter_0000000 \
    --trust_remote_code \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp ${TP} --ep ${EP}
