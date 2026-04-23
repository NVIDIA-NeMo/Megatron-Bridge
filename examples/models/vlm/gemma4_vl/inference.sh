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

# gemma4 requires transformers>=5.5.0; the lockfile pins 5.3.0.
# Upgrade first, then use --no-sync so uv run does not revert the upgrade.
uv pip install -q --upgrade 'transformers>=5.5.0' mistral_common

# Inference with HuggingFace checkpoints (text only)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_gemma4.py \
    --hf_model_path google/gemma-4-26B-A4B \
    --prompt "The capital of France is" \
    --max_new_tokens 20 \
    --tp 4 \
    --pp 2

# Inference with HuggingFace checkpoints (vision + text)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_gemma4.py \
    --hf_model_path google/gemma-4-26B-A4B-it \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg" \
    --prompt "What is shown in this image?" \
    --max_new_tokens 50 \
    --tp 4 \
    --pp 2

# Inference with imported Megatron checkpoints (IT model, VLM)
# conversion.sh imports both the base and IT models; step 3 uses the IT checkpoint.
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_gemma4.py \
    --hf_model_path google/gemma-4-26B-A4B-it \
    --megatron_model_path ${WORKSPACE}/models/gemma-4-26B-A4B-it/iter_0000000 \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG" \
    --prompt "What animal is on the candy?" \
    --max_new_tokens 50 \
    --tp 4 \
    --pp 2
