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
#
# Usage:
#   bash examples/models/audio_lm/inference_qwen3_asr.sh

set -e

export HF_MODEL="Qwen/Qwen3-ASR-1.7B"
export MEGATRON_PATH="examples/models/audio_lm/qwen3_asr"


AUDIO_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"

echo "============================================"
echo "Qwen3-ASR Megatron Bridge Inference Test"
echo "============================================"

# Option 1: Direct inference from HuggingFace (no conversion)
echo ""
echo "Option 1: Direct inference from HuggingFace..."
echo "Audio: ${AUDIO_URL}"
echo ""

uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path ${HF_MODEL} \
  --audio_url "${AUDIO_URL}" \
  --prompt "" \
  --tp 2 \
  --max_new_tokens 50

# Option 2: Convert to Megatron format and run inference
# Uncomment the following to test checkpoint conversion workflow

echo ""
echo "Option 2: Converting HF checkpoint to Megatron format..."
uv run --no-sync python examples/conversion/convert_checkpoints.py import \
  --hf-model ${HF_MODEL} \
  --megatron-path ${MEGATRON_PATH}

echo ""
echo "Running inference on converted checkpoint..."
uv run --no-sync python -m torch.distributed.run examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path ${HF_MODEL} \
  --megatron_model_path ${MEGATRON_PATH}/iter_0000000 \
  --audio_url "${AUDIO_URL}" \
  --prompt "" \
  --max_new_tokens 50

echo ""
echo "============================================"
echo "Inference complete!"
echo "============================================"
