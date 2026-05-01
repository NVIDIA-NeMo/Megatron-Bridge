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
#
# Usage:
#   bash examples/models/audio_lm/qwen2_audio/inference.sh

set -e

export HF_MODEL="Qwen/Qwen2-Audio-7B-Instruct"

AUDIO_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"

echo "============================================"
echo "Qwen2-Audio Megatron Bridge Inference Test"
echo "============================================"

echo ""
echo "Direct inference from HuggingFace..."
echo "Audio: ${AUDIO_URL}"
echo ""

uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path ${HF_MODEL} \
  --audio_url "${AUDIO_URL}" \
  --prompt "Describe what you hear in this audio." \
  --tp 2 \
  --max_new_tokens 50

echo ""
echo "============================================"
echo "Inference complete!"
echo "============================================"
