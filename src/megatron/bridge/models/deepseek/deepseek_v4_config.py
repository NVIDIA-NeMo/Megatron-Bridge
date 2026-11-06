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

from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import (
    rebuild_deepseek_v4_pipeline_layout,
    validate_deepseek_v4_pipeline_segments,
)
from megatron.bridge.models.hybrid.model_config import BridgeHybridMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig


@dataclass
class DeepSeekV4TransformerConfig(MLATransformerConfig):
    """MLA transformer config with the DeepSeek-V4 settings that every supported Megatron Core pin needs."""

    cp_partition_mode: Literal["zigzag", "contiguous"] = "zigzag"
    """Token assignment across context-parallel ranks on Megatron Core's dev pin.

    Megatron Core's dev pin declares this field; its main pin does not. On the main pin,
    ``dsv4_step`` selects contiguous packed context parallelism through ``attention_cp_layout`` and
    ``linear_cp_layout`` instead and does not read this field.
    """


@dataclass(kw_only=True)
class DeepSeekV4ModelConfig(BridgeHybridMLAModelConfig):
    """DeepSeek-V4 config, the builder counterpart of ``DeepSeekV4HybridModelProvider``.

    Each logical DeepSeek-V4 block is an attention-only Hybrid layer (``W``, ``C`` or ``H``)
    followed by a MoE-only layer (``E``), and each MTP depth appends ``WE``. It nests the
    DeepSeek-V4 MLA transformer config. Like the provider, it rebuilds its native ``|`` pipeline
    segments when the recipe runner overrides PP or VPP, and ``validate()`` rejects segments that
    do not match the topology.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]] = DeepSeekV4TransformerConfig

    def _pipeline_model_parallel_layout_builder(
        self, pipeline_model_parallel_size: int, virtual_pipeline_model_parallel_size: int | None
    ) -> list[list[str]] | None:
        """Rebuild this config's pattern and layout after runner PP/VPP overrides."""
        return rebuild_deepseek_v4_pipeline_layout(
            self, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size
        )

    def validate(self) -> None:
        """Check the Hybrid settings and that the native pipeline segments match PP x VPP."""
        super().validate()
        validate_deepseek_v4_pipeline_segments(self)


__all__ = ["DeepSeekV4ModelConfig", "DeepSeekV4TransformerConfig"]
