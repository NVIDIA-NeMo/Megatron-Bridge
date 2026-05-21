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
# Set the model name to any of the supported dense or MoE Qwen3.5-VL models:
#   Dense: Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B
#   MoE:   Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B
# For Qwen3.5-397B-A17B, please use the slurm_inference.sh script for multinode inference.
MODEL_NAME=Qwen3.5-35B-A3B

# Set EP (Expert Parallelism) to 1 for dense models, 4 for MoE models
case "$MODEL_NAME" in
    Qwen3.5-0.8B|Qwen3.5-2B|Qwen3.5-4B|Qwen3.5-9B|Qwen3.5-27B)
        EP=1
        ;;
    Qwen3.5-35B-A3B|Qwen3.5-122B-A10B|Qwen3.5-397B-A17B)
        EP=4
        ;;
    *)
        echo "ERROR: Unknown model type for \$MODEL_NAME: $MODEL_NAME"
        exit 1
        ;;
esac

# Inference with Hugging Face checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path Qwen/${MODEL_NAME} \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 2 --pp 2 --ep ${EP}

# Inference with imported Megatron checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path Qwen/${MODEL_NAME} \
    --megatron_model_path ${WORKSPACE}/${MODEL_NAME}/iter_0000000 \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 2 --pp 2 --ep ${EP}

# Inference with exported HF checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path ${WORKSPACE}/${MODEL_NAME}-hf-export \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 2 --pp 2 --ep ${EP}
