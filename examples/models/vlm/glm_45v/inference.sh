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

# GLM-4.5V is a large MoE model (106B parameters)
# Using TP=1, PP=2, EP=4 for inference (8 GPUs minimum)

# Inference with Hugging Face checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path zai-org/GLM-4.5V \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 1 \
    --pp 2 \
    --ep 4 \
    --trust_remote_code

# Inference with imported Megatron checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path zai-org/GLM-4.5V \
    --megatron_model_path ${WORKSPACE}/models/GLM-4.5V/iter_0000000 \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 1 \
    --pp 2 \
    --ep 4 \
    --trust_remote_code

# Inference with exported HF checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path ${WORKSPACE}/models/GLM-4.5V-hf-export \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 1 \
    --pp 2 \
    --ep 4 \
    --trust_remote_code
