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
from typing import Any, ClassVar, Literal

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.common.dsa_metrics import register_dsa_metric_hook
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import (
    rebuild_deepseek_v4_pipeline_layout,
    validate_deepseek_v4_pipeline_segments,
)
from megatron.bridge.models.hybrid import BridgeHybridModelBuilder
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
    followed by a MoE-only layer (``E``), and each MTP depth appends ``WE``. Like the provider,
    it uses Hybrid construction and finalization with the DeepSeek-V4 MLA config
    and registers DSA metric setup after wrapping. It also rebuilds native pipeline segments
    after PP/VPP overrides and rejects segments that do not match the topology.
    """

    builder: ClassVar[str] = "megatron.bridge.models.deepseek.deepseek_v4_config.DeepSeekV4ModelBuilder"

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

    def get_builder_cls(self) -> type:
        """Use DSA setup when restoring an older config that saved the generic Hybrid builder."""
        builder_cls = super().get_builder_cls()
        if builder_cls is BridgeHybridModelBuilder:
            return DeepSeekV4ModelBuilder
        return builder_cls


class DeepSeekV4ModelBuilder(BridgeHybridModelBuilder):
    """Build DeepSeek-V4 with its DSA metric hook."""

    def build_distributed_models(
        self, pg_collection: ProcessGroupCollection, *args: Any, **kwargs: Any
    ) -> list[HybridModel]:
        """Register DSA metric setup for the complete list of wrapped model chunks."""
        register_dsa_metric_hook(self._model_config, pg_collection)
        return super().build_distributed_models(pg_collection, *args, **kwargs)


__all__ = ["DeepSeekV4ModelBuilder", "DeepSeekV4ModelConfig", "DeepSeekV4TransformerConfig"]
