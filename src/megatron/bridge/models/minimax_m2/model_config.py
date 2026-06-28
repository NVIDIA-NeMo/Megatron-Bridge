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

"""Provider-neutral MiniMax-M2 model configuration."""

from dataclasses import dataclass, field
from typing import Callable

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.minimax_m2.layer_specs import minimax_m2_layer_spec


@dataclass(kw_only=True)
class MiniMaxM2ModelConfig(BridgeGPTModelConfig):
    """Builder-backed MiniMax-M2 config with full-dimension QK normalization."""

    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: minimax_m2_layer_spec)


__all__ = ["MiniMaxM2ModelConfig"]
