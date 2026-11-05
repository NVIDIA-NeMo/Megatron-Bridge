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

"""Builder-backed model config for K-EXAONE MoE models.

The per-layer attention, SwiGLU limit and MTP fields live on the nested
transformer config because ``ExaoneMoeDecoderLayer`` reads them from the
``TransformerConfig`` it is built with. The standard GPT builder constructs the
model.
"""

from dataclasses import dataclass
from typing import ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class ExaoneMoeTransformerConfig(TransformerConfig):
    """K-EXAONE decoder fields not represented by MCore's base config."""

    layer_types: list[str] | None = None
    sliding_windows: list[int] | None = None
    mtp_layer_types: list[str] | None = None
    mtp_sliding_windows: list[int] | None = None
    swiglu_limits: list[float] | None = None


@dataclass(kw_only=True)
class ExaoneMoeModelConfig(BridgeGPTModelConfig):
    """Builder config for K-EXAONE MoE models (236B-A23B, 2.0 750B-A37B)."""

    transformer_config_class: ClassVar[type[TransformerConfig]] = ExaoneMoeTransformerConfig


__all__ = ["ExaoneMoeModelConfig", "ExaoneMoeTransformerConfig"]
