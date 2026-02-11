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

"""
Qwen2-Audio Model Bridge and Provider implementations.

This module provides support for Qwen2-Audio audio-language models.

Reference: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

Supported models:
- Qwen2-Audio-7B
- Qwen2-Audio-7B-Instruct

Example usage:
    >>> from megatron.bridge import AutoBridge
    >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    >>> provider = bridge.to_megatron_provider()
"""

from megatron.bridge.models.qwen_audio.modeling_qwen2_audio import Qwen2AudioModel
from megatron.bridge.models.qwen_audio.qwen2_audio_bridge import Qwen2AudioBridge
from megatron.bridge.models.qwen_audio.qwen2_audio_provider import (
    Qwen2AudioModelProvider,
    Qwen2AudioModelProvider7B,
)


__all__ = [
    # Bridge
    "Qwen2AudioBridge",
    # Model
    "Qwen2AudioModel",
    # Model Providers
    "Qwen2AudioModelProvider",
    "Qwen2AudioModelProvider7B",
]
