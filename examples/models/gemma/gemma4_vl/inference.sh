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
IMAGE_PATH=${IMAGE_PATH:-"https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/cb5a65ff10232128389d882d805fa609427544f1/images/table.png"}

# Inference with HuggingFace checkpoints (text only)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-4-26B-A4B \
    --prompt "The capital of France is" \
    --max_new_tokens 20 \
    --tp 4 \
    --pp 2

# Inference with HuggingFace checkpoints (vision + text)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-4-26B-A4B-it \
    --image_path "${IMAGE_PATH}" \
    --prompt "According to the table, how much GPU memory does H100 NVL have?" \
    --max_new_tokens 50 \
    --tp 4 \
    --pp 2

# Inference with imported Megatron checkpoints (IT model, VLM)
# conversion.sh imports both the base and IT models; step 3 uses the IT checkpoint.
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-4-26B-A4B-it \
    --megatron_model_path ${WORKSPACE}/models/gemma-4-26B-A4B-it/iter_0000000 \
    --image_path "${IMAGE_PATH}" \
    --prompt "According to the table, how much GPU memory does H100 NVL have?" \
    --max_new_tokens 50 \
    --tp 4 \
    --pp 2
