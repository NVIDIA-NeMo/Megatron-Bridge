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

MODEL_NAME=Qwen2.5-Omni-7B
# For --use_audio_in_video, audio extraction requires a local file (URL streaming not supported).
# Download the video first: wget -O /path/to/video.mp4 <url>
VIDEO_PATH=${VIDEO_PATH:-/path/to/video.mp4}
PROMPT="What was the first sentence the boy said when he met the girl?"

# Requires: uv pip install qwen-omni-utils[decord]
# Requires ffmpeg for audio: uv pip install imageio-ffmpeg && ln -sf $(uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") /usr/local/bin/ffmpeg

# Inference with Hugging Face checkpoints (video + audio)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_omni_lm.py \
    --hf_model_path Qwen/${MODEL_NAME} \
    --video_path "${VIDEO_PATH}" \
    --prompt "${PROMPT}" \
    --use_audio_in_video \
    --max_new_tokens 64 \
    --tp 2

# Inference with imported Megatron checkpoint (video + audio)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_omni_lm.py \
    --hf_model_path Qwen/${MODEL_NAME} \
    --megatron_model_path ${WORKSPACE}/models/${MODEL_NAME}/iter_0000000 \
    --video_path "${VIDEO_PATH}" \
    --prompt "${PROMPT}" \
    --use_audio_in_video \
    --max_new_tokens 64 \
    --tp 2

# Inference with exported HF checkpoint
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_omni_lm.py \
    --hf_model_path ${WORKSPACE}/models/${MODEL_NAME}-hf-export \
    --video_path "${VIDEO_PATH}" \
    --prompt "${PROMPT}" \
    --use_audio_in_video \
    --max_new_tokens 64 \
    --tp 2
