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

"""Serializable EXAONE 4.5 VLM config and standalone builder."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config, Exaone4_5_VisionConfig

from megatron.bridge.models.exaone.exaone45.layer_specs import exaone_45_transformer_layer_spec
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.model import Exaone45Model
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.transformer_config import Exaone45TransformerConfig
from megatron.bridge.models.gpt.model_builder import LayerSpecGPTModelBuilder
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class Exaone45ModelConfig(BridgeGPTModelConfig):
    """Pure-data EXAONE 4.5 multimodal build configuration."""

    builder: ClassVar[str] = "megatron.bridge.models.exaone.exaone45.model_config.Exaone45ModelBuilder"
    transformer: Exaone45TransformerConfig
    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: exaone_45_transformer_layer_spec)
    vision_config: dict[str, Any] = field(default_factory=dict)
    hf_text_config: dict[str, Any] = field(default_factory=dict)
    image_token_id: int = 67
    video_token_id: int = 68
    vision_token_id: int = 67
    vision_start_token_id: int = 73
    vision_end_token_id: int = 74
    bos_token_id: int = 1
    eos_token_id: int = 53
    spatial_merge_size: int = 2
    scatter_embedding_sequence_parallel: bool = False
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    freeze_mtp_model: bool = False


class Exaone45ModelBuilder(LayerSpecGPTModelBuilder):
    """Build EXAONE 4.5 directly from serializable text and vision configs."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Exaone45Model:
        """Build one EXAONE 4.5 pipeline stage."""
        config = self._model_config
        assert isinstance(config, Exaone45ModelConfig)
        transformer = replace(config.transformer)
        transformer.vocab_size = config.vocab_size
        transformer.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        transformer.rotary_percent = config.rotary_percent
        transformer.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy
        transformer.image_token_id = config.image_token_id
        transformer.video_token_id = config.video_token_id
        transformer.vision_start_token_id = config.vision_start_token_id
        transformer.hf_text_config = Exaone4_5_Config.from_dict(config.hf_text_config)
        layer_spec = config.transformer_layer_spec(config)
        model = Exaone45Model(
            language_transformer_config=transformer,
            language_transformer_layer_spec=layer_spec,
            vision_transformer_config=Exaone4_5_VisionConfig.from_dict(config.vision_config),
            parallel_output=config.parallel_output,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        if (
            config.freeze_language_model
            or config.freeze_vision_model
            or config.freeze_vision_projection
            or config.freeze_mtp_model
        ):
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
                freeze_mtp_model=config.freeze_mtp_model,
            )
        return model


__all__ = ["Exaone45ModelBuilder", "Exaone45ModelConfig"]
