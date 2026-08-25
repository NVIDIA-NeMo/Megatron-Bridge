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

"""Nemotron 3.5 Super VL training recipes."""

from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    _make_nemotron_omni_energon_dataset,
    _nemotron_omni_base,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_super import (
    _apply_nemotron_3_super_64gpu_h100_training_stack,
)
from megatron.bridge.training.config import ConfigContainer


NEMOTRON_35_SUPER_VL_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth"


def nemotron_35_super_vl_sft_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return the 64-H100 BF16 SFT configuration for Nemotron 3.5 Super VL.

    The model and audio-free image/video Energon data path reuse the Nemotron
    Omni stack. The language decoder reuses the measured Nemotron 3 Super
    TP1/PP2/EP32 HybridEP training stack without replacing this checkpoint's
    native one-layer MTP or separate temporal video embedder configuration.

    The Energon shard path must be set with ``dataset.path=<path>``. Samples
    may contain text, images, and videos, but must not contain audio.

    Returns:
        The Super-VL SFT configuration.
    """
    cfg = _nemotron_omni_base(hf_path=NEMOTRON_35_SUPER_VL_HF_MODEL_ID)
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = True
    cfg.model.calculate_per_token_loss = True

    cfg.dataset = _make_nemotron_omni_energon_dataset(
        cfg.train.micro_batch_size,
        hf_processor_path=NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
    )
    return _apply_nemotron_3_super_64gpu_h100_training_stack(cfg)


__all__ = [
    "nemotron_35_super_vl_sft_64gpu_h100_bf16_config",
]
