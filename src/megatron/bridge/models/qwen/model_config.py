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

"""Builder-backed model configuration for Qwen hybrid text models."""

from dataclasses import dataclass, field
from typing import Callable

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class QwenHybridModelConfig(BridgeGPTModelConfig):
    """GPT build config using Qwen's mixed GDN and attention block spec."""

    transformer_layer_spec: Callable[..., TransformerBlockSubmodules] = field(
        default_factory=lambda: get_transformer_block_with_experimental_attention_variant_spec
    )


__all__ = ["QwenHybridModelConfig"]
