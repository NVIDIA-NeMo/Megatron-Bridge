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

"""Compatibility exports for VLM model-specific collators."""

from megatron.bridge.data.vlm_processing import build_assistant_loss_mask
from megatron.bridge.models.gemma_vl.data.collate_fn import gemma3_vl_collate_fn, gemma4_vl_collate_fn
from megatron.bridge.models.glm_vl.data.collate_fn import glm4v_collate_fn
from megatron.bridge.models.kimi_vl.data.collate_fn import kimi_k25_vl_collate_fn
from megatron.bridge.models.ministral3.data.collate_fn import ministral3_collate_fn
from megatron.bridge.models.nemotron_omni.data.collate_fn import nemotron_omni_collate_fn
from megatron.bridge.models.nemotron_vl.data.collate_fn import nemotron_nano_v2_vl_collate_fn
from megatron.bridge.models.qwen_audio.data.collate_fn import qwen2_audio_collate_fn
from megatron.bridge.models.qwen_vl.data.collate_fn import qwen2_5_collate_fn


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3VLProcessor": qwen2_5_collate_fn,
    "NemotronNanoVLV2Processor": nemotron_nano_v2_vl_collate_fn,
    "NemotronH_Nano_Omni_Reasoning_V3Processor": nemotron_omni_collate_fn,
    "PixtralProcessor": ministral3_collate_fn,  # Ministral3 uses PixtralProcessor
    "Gemma3Processor": gemma3_vl_collate_fn,  # Gemma3 VL
    "Gemma4Processor": gemma4_vl_collate_fn,  # Gemma4 VL
    "Qwen2AudioProcessor": qwen2_audio_collate_fn,
    "Glm4vProcessor": glm4v_collate_fn,
    "KimiK25Processor": kimi_k25_vl_collate_fn,
}


__all__ = [
    "COLLATE_FNS",
    "build_assistant_loss_mask",
    "gemma3_vl_collate_fn",
    "gemma4_vl_collate_fn",
    "glm4v_collate_fn",
    "kimi_k25_vl_collate_fn",
    "ministral3_collate_fn",
    "nemotron_nano_v2_vl_collate_fn",
    "nemotron_omni_collate_fn",
    "qwen2_5_collate_fn",
    "qwen2_audio_collate_fn",
]
