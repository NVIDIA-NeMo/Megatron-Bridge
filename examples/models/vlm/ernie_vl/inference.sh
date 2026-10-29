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

# Supported model variants:
#   ERNIE-4.5-VL-28B-A3B-Instruct, ERNIE-4.5-VL-28B-A3B-Thinking
MODEL_NAME=ERNIE-4.5-VL-28B-A3B-Instruct

EP=4
TP=2
PP=1

# ERNIE 4.5 VL uses a custom processor API that differs from Qwen-style models,
# so we use a dedicated generate script instead of the generic hf_to_megatron_generate_vlm.py.

# Inference with Hugging Face checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/models/vlm/ernie_vl/hf_to_megatron_generate_ernie_vl.py \
    --hf_model_path baidu/${MODEL_NAME} \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --trust_remote_code
