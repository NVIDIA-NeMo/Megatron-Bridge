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

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}
# Supported model variants:
#   ERNIE-4.5-VL-28B-A3B-Instruct, ERNIE-4.5-VL-28B-A3B-Thinking
MODEL_NAME=ERNIE-4.5-VL-28B-A3B-Instruct

EP=4
TP=2
PP=1

# Import HF -> Megatron
./scripts/conversion/convert.sh import \
    --hf-model baidu/${MODEL_NAME} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME} \
    --torch-dtype bfloat16 \
    --trust-remote-code

# HF and Megatron models logits comparison validation
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path baidu/${MODEL_NAME} \
    --megatron_model_path ${WORKSPACE}/${MODEL_NAME} \
    --model_class "Ernie4_5_VLMoeForConditionalGeneration" \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --trust-remote-code

# Export Megatron -> HF
./scripts/conversion/convert.sh export \
    --hf-model baidu/${MODEL_NAME} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME}/iter_0000000 \
    --hf-path ${WORKSPACE}/${MODEL_NAME}-hf-export \
    --trust-remote-code

# Round-trip validation
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id baidu/${MODEL_NAME} --tp ${TP} --pp ${PP} --ep ${EP} --trust-remote-code
