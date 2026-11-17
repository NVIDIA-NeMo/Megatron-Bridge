# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builder configuration for MiniCPM-o's understanding stage."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class MiniCPMOModelConfig(BridgeGPTModelConfig):
    """Qwen3 language config plus the original MiniCPM-o perception config.

    Remote checkpoint code must be explicitly trusted before model construction.
    Encoder and projector freezing are independent of language-model training.
    """

    builder: ClassVar[str] = "megatron.bridge.models.minicpmo.minicpmo_builder.MiniCPMOModelBuilder"
    minicpmo_config: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = False
    freeze_language_model: bool = False
    freeze_vision_model: bool = True
    freeze_audio_model: bool = True
    freeze_vision_projection: bool = True
    freeze_audio_projection: bool = True
