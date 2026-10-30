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

"""Builder-backed model config for Step-3.5-Flash.

The per-layer attention, RoPE and SwiGLU-limit fields live on the nested
transformer config because ``Step35DecoderLayer`` reads them from the
``TransformerConfig`` it is built with. The standard GPT builder constructs the
model.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class Step35TransformerConfig(TransformerConfig):
    """Step-3.5 decoder fields not represented by MCore's base config.

    The per-layer lists are indexed by the global layer id, with MTP layers
    following the main decoder layers.
    """

    layer_types: list[str] | None = None
    rotary_percents: list[float] | None = None
    rotary_base_per_layer: list[float] | None = None
    sliding_attention_setting: dict[str, Any] | None = None
    swiglu_limits: list[float | None] | None = None
    swiglu_limits_shared: list[float | None] | None = None
    head_wise_attn_gate: bool | None = False


@dataclass(kw_only=True)
class Step35ModelConfig(BridgeGPTModelConfig):
    """Builder config for Step-3.5-Flash."""

    transformer_config_class: ClassVar[type[TransformerConfig]] = Step35TransformerConfig


__all__ = ["Step35ModelConfig", "Step35TransformerConfig"]
