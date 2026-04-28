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

# ==============================================================================
# Nemotron-3 Nano Omni - Multimodal generation (image / video / audio / video+audio)
#
# Drives examples/conversion/hf_to_megatron_generate_nemotron_omni.py against
# the imported Megatron checkpoint produced by conversion.sh. Omit
# --megatron_model_path to convert HF -> Megatron on the fly instead.
# ==============================================================================

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
HF_MODEL_ID=${HF_MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}
MODEL_NAME=$(basename "$HF_MODEL_ID")
MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE}/models/${MODEL_NAME}}

# Default asset paths — override with your own image / mp4 / wav if desired.
IMAGE_PATH=${IMAGE_PATH:-${WORKSPACE}/assets/table.png}
VIDEO_PATH=${VIDEO_PATH:-${WORKSPACE}/assets/demo.mp4}
AUDIO_PATH=${AUDIO_PATH:-${WORKSPACE}/assets/2414-165385-0000.wav}

# ---------------------------------------------------------------------------
# Assets: download from the public HF model card if the paths don't exist.
# ---------------------------------------------------------------------------
_HF_ASSETS="https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/resolve/main/media"

mkdir -p "$(dirname "$IMAGE_PATH")" "$(dirname "$VIDEO_PATH")" "$(dirname "$AUDIO_PATH")"

[[ -f "$IMAGE_PATH" ]] || curl -fL "${_HF_ASSETS}/table.png"             -o "$IMAGE_PATH"
[[ -f "$VIDEO_PATH" ]] || curl -fL "${_HF_ASSETS}/demo.mp4"              -o "$VIDEO_PATH"
[[ -f "$AUDIO_PATH" ]] || curl -fL "${_HF_ASSETS}/2414-165385-0000.wav"  -o "$AUDIO_PATH"

GEN_SCRIPT=examples/conversion/hf_to_megatron_generate_nemotron_omni.py

# 1) Image + Text (1 GPU)
uv run python -m torch.distributed.run --nproc_per_node=1 ${GEN_SCRIPT} \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path "$MEGATRON_PATH" \
    --image_path "$IMAGE_PATH" \
    --prompt "Describe this image." \
    --max_new_tokens 100

# 2) Video + Text (TP=4, EP=4, 8 GPUs)
uv run python -m torch.distributed.run --nproc_per_node=8 ${GEN_SCRIPT} \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path "$MEGATRON_PATH" \
    --video_path "$VIDEO_PATH" \
    --prompt "Describe what you see." \
    --tp 4 --ep 4 \
    --max_new_tokens 100

# 3) Audio + Text (1 GPU)
uv run python -m torch.distributed.run --nproc_per_node=1 ${GEN_SCRIPT} \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path "$MEGATRON_PATH" \
    --audio_path "$AUDIO_PATH" \
    --prompt "Transcribe the audio." \
    --max_new_tokens 100

# 4) Video + Audio + Text (TP=4, EP=2, 8 GPUs)
uv run python -m torch.distributed.run --nproc_per_node=8 ${GEN_SCRIPT} \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path "$MEGATRON_PATH" \
    --video_path "$VIDEO_PATH" \
    --audio_path "$AUDIO_PATH" \
    --prompt "Describe the video and audio." \
    --tp 4 --ep 2 \
    --max_new_tokens 150
