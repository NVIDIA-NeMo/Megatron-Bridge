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

"""Pure model configurations and builders for Qwen Omni models."""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.training.models.gpt import GPTModelBuilder
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniModel
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.model import Qwen25OmniModel


@dataclass(kw_only=True)
class QwenOmniModelConfig(BridgeGPTModelConfig):
    """Serializable thinker-side Qwen Omni configuration."""

    thinker_config: dict[str, Any] = field(default_factory=dict)
    language_max_sequence_length: int = 2048
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151646
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_audio_model: bool = False
    vit_gradient_checkpointing: bool = False
    multimodal_attn_impl: str = "auto"


def _language_transformer(config: QwenOmniModelConfig):
    """Return an exact MCore config with its declared M-RoPE field populated."""
    return replace(config.transformer, mrope_section=config.mrope_section)


def _pipeline_flags(config, pg_collection, pre_process, post_process, vp_stage):
    vp_size = config.transformer.virtual_pipeline_model_parallel_size
    if pre_process is None:
        pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
    if post_process is None:
        post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)
    return pre_process, post_process


class _QwenOmniBuilder(GPTModelBuilder):
    model_cls = None
    thinker_cls = None

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MegatronModule:
        """Build one Qwen Omni pipeline stage.

        Args:
            pg_collection: Process groups for distributed construction.
            pre_process: Whether this stage owns input processing.
            post_process: Whether this stage owns output processing.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Qwen Omni stage.
        """
        config = self._model_config
        transformer = _language_transformer(config)
        pre_process, post_process = _pipeline_flags(config, pg_collection, pre_process, post_process, vp_stage)
        thinker_config = self.thinker_cls(**config.thinker_config)
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=transformer.num_moe_experts,
            moe_grouped_gemm=transformer.moe_grouped_gemm,
            qk_layernorm=transformer.qk_layernorm,
            fp8=False,
        )
        model = self.model_cls(
            language_transformer_config=transformer,
            language_transformer_layer_spec=layer_spec,
            thinker_transformer_config=thinker_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            model_config=config,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_audio_model:
            model.freeze(config.freeze_language_model, config.freeze_vision_model, config.freeze_audio_model)
        return model


@dataclass(kw_only=True)
class Qwen25OmniModelConfig(QwenOmniModelConfig):
    """Builder input for Qwen2.5-Omni."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen_omni.model_config.Qwen25OmniModelBuilder"


class Qwen25OmniModelBuilder(_QwenOmniBuilder):
    """Build Qwen2.5-Omni without a provider."""

    model_cls = Qwen25OmniModel
    thinker_cls = Qwen2_5OmniThinkerConfig


@dataclass(kw_only=True)
class Qwen3OmniModelConfig(QwenOmniModelConfig):
    """Builder input for Qwen3-Omni."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen_omni.model_config.Qwen3OmniModelBuilder"


class Qwen3OmniModelBuilder(_QwenOmniBuilder):
    """Build Qwen3-Omni without a provider."""

    model_cls = Qwen3OmniModel
    thinker_cls = Qwen3OmniMoeThinkerConfig


__all__ = ["Qwen25OmniModelBuilder", "Qwen25OmniModelConfig", "Qwen3OmniModelBuilder", "Qwen3OmniModelConfig"]
