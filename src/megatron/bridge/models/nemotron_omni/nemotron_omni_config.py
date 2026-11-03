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

"""Builder-backed model config for Nemotron Omni and Nemotron 3.5 Super VL.

The language half is the Nemotron-H hybrid decoder, configured with the same fields as
``BridgeHybridModelConfig``. The media settings live on the outer config;
``NemotronOmniModelBuilder`` combines the decoder with the RADIO vision encoder and the
optional Parakeet sound encoder.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig


def _validate_nemotron_omni_settings(settings: Any) -> None:
    """Validate the media settings that the checkpoint configuration must supply.

    Args:
        settings: Nemotron Omni model config or provider.

    Raises:
        ValueError: If dynamic resolution is disabled, the image token ID is not positive, or a
            sound-enabled configuration lacks a positive sound token ID or a sound encoder config.
    """
    if settings.dynamic_resolution is not True:
        raise ValueError("Nemotron Omni only supports dynamic_resolution=True.")
    if settings.image_token_index is None or settings.image_token_index <= 0:
        raise ValueError(
            "Nemotron Omni requires a positive image_token_index from the checkpoint configuration; "
            f"got {settings.image_token_index}. Construct the config through AutoBridge or set it explicitly."
        )
    if settings.has_sound and settings.sound_context_token_id <= 0:
        raise ValueError(
            "Sound-enabled Nemotron Omni requires a positive sound_context_token_id from the checkpoint "
            f"configuration; got {settings.sound_context_token_id}."
        )
    if settings.has_sound and settings.sound_config is None:
        raise ValueError("Sound-enabled Nemotron Omni requires sound_config from the checkpoint configuration.")


@dataclass(kw_only=True)
class NemotronOmniModelConfig(BridgeHybridModelConfig):
    """Builder config for the canonical expanded-sequence Nemotron Omni model.

    It serves Nemotron-3 Nano and Super Omni and Nemotron 3.5 Super VL. The media fields are the
    ``NemotronOmniModelProvider`` fields of the same names and defaults.
    """

    builder: ClassVar[str] = "megatron.bridge.models.nemotron_omni.nemotron_omni_builder.NemotronOmniModelBuilder"

    image_token_index: int = 0
    """Token ID that each projected image feature replaces."""

    vision_proj_ffn_hidden_size: int = 20480
    vision_class_token_len: int | None = None

    # The public processor emits variable-resolution images, which Bridge pre-patchifies for RADIO.
    dynamic_resolution: Literal[True] = True
    use_vision_backbone_fp8_arch: bool = True
    radio_force_eval_mode: bool = True
    radio_force_cpe_eval_mode: bool = True
    radio_interpolate_only_cpe: bool = False
    radio_cpe_aspect_ratio_select: bool = False
    radio_disable_cpe: bool = False

    recompute_vision: bool = False
    vision_recompute_granularity: Literal["full", "selective"] | None = None
    vision_recompute_modules: list[str] | None = None
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None

    temporal_patch_dim: int = 1
    separate_video_embedder: bool = False
    temporal_ckpt_compat: bool = False
    vision_final_layernorm: bool = False

    vision_dp_over_cp: bool = True
    """Shard images across the context-parallel group instead of encoding them on every CP rank."""

    has_sound: bool = False
    """Whether the checkpoint has a sound encoder; disabling it omits the encoder and its projector."""

    sound_model_type: str = "parakeet"
    sound_hidden_size: int = 1024
    sound_projection_hidden_size: int = 4096
    sound_context_token_id: int = 0
    sound_config: dict[str, Any] | None = None

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    freeze_sound_encoder: bool = False
    freeze_sound_projection: bool = False

    def finalize(self) -> None:
        """Validate the media settings, then finalize the hybrid language config."""
        _validate_nemotron_omni_settings(self)
        super().finalize()


__all__ = ["NemotronOmniModelConfig"]
