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

"""Serializable Gemma multimodal configs and builders."""

import copy
from dataclasses import dataclass, field
from typing import ClassVar

from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.gemma.model_config import (
    Gemma3ModelBuilder,
    Gemma3ModelConfig,
    Gemma4DenseModelBuilder,
    Gemma4DenseModelConfig,
    Gemma4ModelBuilder,
    Gemma4ModelConfig,
)
from megatron.bridge.models.gemma_vl.modeling_gemma3_vl import Gemma3VLModel
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel


@dataclass(kw_only=True)
class Gemma3VLModelConfig(Gemma3ModelConfig):
    """Pure-data Gemma3 VL build configuration."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma_vl.model_config.Gemma3VLModelBuilder"
    vision_config: dict[str, object] = field(default_factory=dict)
    vision_projector_input_size: int = 0
    vision_projector_hidden_size: int = 0
    mm_tokens_per_image: int = 256
    bos_token_id: int = 2
    eos_token_id: int | list[int] = 1
    vision_start_token_id: int = 255999
    vision_end_token_id: int = 256000
    image_token_id: int = 262144
    return_dict: bool = True
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False


class Gemma3VLModelBuilder(Gemma3ModelBuilder):
    """Build Gemma3 language stages and inject them into the VL wrapper."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Gemma3VLModel:
        """Build one Gemma3 VL pipeline stage."""
        language_model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        config = self._model_config
        model = Gemma3VLModel(
            config=config,
            language_model=language_model,
            pg_collection=pg_collection,
            pre_process=pre_process if pre_process is not None else True,
            post_process=post_process if post_process is not None else True,
            vp_stage=vp_stage,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
            )
        return model


@dataclass(kw_only=True)
class _Gemma4VLFields:
    vision_config: dict[str, object] = field(default_factory=dict)
    text_config: dict[str, object] = field(default_factory=dict)
    audio_config: dict[str, object] | None = None
    vision_soft_tokens_per_image: int = 280
    bos_token_id: int = 2
    eos_token_id: int | list[int] = 1
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False


@dataclass(kw_only=True)
class Gemma4VLModelConfig(_Gemma4VLFields, Gemma4ModelConfig):
    """Pure Gemma4 MoE VL construction state."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma_vl.model_config.Gemma4VLModelBuilder"


@dataclass(kw_only=True)
class Gemma4DenseVLModelConfig(_Gemma4VLFields, Gemma4DenseModelConfig):
    """Pure Gemma4 Dense VL construction state."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma_vl.model_config.Gemma4DenseVLModelBuilder"


def _hf_config(config_dict: dict[str, object] | None):
    if config_dict is None:
        return None
    from transformers import AutoConfig

    values = dict(config_dict)
    model_type = values.pop("model_type")
    return AutoConfig.for_model(model_type, **values)


class _Gemma4VLBuilderMixin:
    def _wrap_vl(self, config, language_model, pg_collection, pre_process, post_process, vp_stage):
        runtime_config = copy.copy(config)
        runtime_config.vision_config = _hf_config(config.vision_config)
        runtime_config.text_config = _hf_config(config.text_config)
        runtime_config.audio_config = _hf_config(config.audio_config)
        model = Gemma4VLModel(
            runtime_config,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            vp_stage=vp_stage,
            language_model=language_model,
            pg_collection=pg_collection,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
            )
        return model


class Gemma4VLModelBuilder(_Gemma4VLBuilderMixin, Gemma4ModelBuilder):
    """Build and wrap the Gemma4 MoE language model."""

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        """Build one Gemma4 MoE VL pipeline stage."""
        language_model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        return self._wrap_vl(self._model_config, language_model, pg_collection, pre_process, post_process, vp_stage)


class Gemma4DenseVLModelBuilder(_Gemma4VLBuilderMixin, Gemma4DenseModelBuilder):
    """Build and wrap the Gemma4 Dense language model."""

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        """Build one Gemma4 Dense VL pipeline stage."""
        language_model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        return self._wrap_vl(self._model_config, language_model, pg_collection, pre_process, post_process, vp_stage)


__all__ = [
    "Gemma3VLModelBuilder",
    "Gemma3VLModelConfig",
    "Gemma4DenseVLModelBuilder",
    "Gemma4DenseVLModelConfig",
    "Gemma4VLModelBuilder",
    "Gemma4VLModelConfig",
]
