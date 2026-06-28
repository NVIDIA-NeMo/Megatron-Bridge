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

"""Serializable GLM 4.5V config and builder."""

from dataclasses import dataclass, field
from typing import ClassVar

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.glm.model_config import GLM45ModelConfig
from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel


@dataclass(kw_only=True)
class GLM45VModelConfig(GLM45ModelConfig):
    """Pure-data GLM 4.5V build configuration."""

    builder: ClassVar[str] = "megatron.bridge.models.glm_vl.model_config.GLM45VModelBuilder"
    vision_config: dict[str, object] = field(default_factory=dict)
    eos_token_id: int = 151329
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342
    image_token_id: int = 151363
    video_token_id: int = 151364
    spatial_merge_size: int = 2
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False


class GLM45VModelBuilder(GPTModelBuilder):
    """Build the GLM language stage and inject it into the vision wrapper."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GLM45VModel:
        """Build one GLM 4.5V pipeline stage."""
        language_model = super().build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        model = GLM45VModel(
            config=self._model_config,
            language_model=language_model,
            pg_collection=pg_collection,
            pre_process=pre_process if pre_process is not None else True,
            post_process=post_process if post_process is not None else True,
            vp_stage=vp_stage,
        )
        config = self._model_config
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
            )
        return model


__all__ = ["GLM45VModelBuilder", "GLM45VModelConfig"]
