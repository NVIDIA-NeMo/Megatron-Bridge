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

"""Builder-backed model config for DeepSeek-V4 expressed as a Megatron-Core ``HybridModel``."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from megatron.bridge.models.hybrid.model_config import BridgeHybridMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig


@dataclass
class DeepSeekV4TransformerConfig(MLATransformerConfig):
    """MLA transformer config with the DeepSeek-V4 settings that every supported Megatron Core pin needs."""

    cp_partition_mode: Literal["zigzag", "contiguous"] = "zigzag"
    """Token assignment across context-parallel ranks.

    DeepSeek-V4's compressed sparse attention exchanges boundary states between adjacent ranks, so
    packed training with CP > 1 uses ``contiguous`` through ``dsv4_step``. Megatron Core's dev pin
    declares this field; its main pin does not.
    """


@dataclass(kw_only=True)
class DeepSeekV4ModelConfig(BridgeHybridMLAModelConfig):
    """DeepSeek-V4 config, the builder counterpart of ``DeepSeekV4HybridModelProvider``.

    Each logical DeepSeek-V4 block is an attention-only Hybrid layer (``W``, ``C`` or ``H``)
    followed by a MoE-only layer (``E``), and each MTP depth appends ``WE``. Like the provider,
    it adds nothing to the Hybrid construction and finalization; it nests the DeepSeek-V4 MLA
    transformer config.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]] = DeepSeekV4TransformerConfig


__all__ = ["DeepSeekV4ModelConfig", "DeepSeekV4TransformerConfig"]
