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

"""Builder-backed model config for the Kimi K3 language backbone."""

from dataclasses import dataclass
from typing import ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig


@dataclass
class KimiK3TransformerConfig(MLATransformerConfig):
    """MLA transformer config with the settings that Kimi K3's KDA, MLA and AttnRes layers read."""

    variable_seq_lengths: bool = True
    """A pipeline stage passes the AttnRes snapshot bank packed with its hidden states, so the tensor width
    depends on the stage boundary and pipeline communication exchanges shapes."""

    kimi_kda_layers: tuple[int, ...] = ()
    """1-based numbers of the layers that use Kimi Delta Attention (KDA); the other layers use MLA."""

    kimi_linear_num_heads: int = 96
    kimi_linear_head_dim: int = 128
    kimi_linear_conv_kernel_size: int = 4
    kimi_kda_gate_lower_bound: float = -5.0
    kimi_attn_res_block_size: int = 12
    """Number of layers whose outputs each AttnRes snapshot covers."""


@dataclass(kw_only=True)
class KimiK3ModelConfig(BridgeGPTMLAModelConfig):
    """Kimi K3 language backbone, the builder counterpart of ``KimiK3ModelProvider``.

    The standard GPT builder constructs it from the K3 decoder block spec; like the
    provider, it adds nothing to the GPT construction and finalization. It nests the
    K3 transformer config.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]] = KimiK3TransformerConfig


__all__ = ["KimiK3ModelConfig", "KimiK3TransformerConfig"]
