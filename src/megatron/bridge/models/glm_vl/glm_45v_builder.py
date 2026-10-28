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

"""Model builder for GLM-4.5V and GLM-4.6V vision-language models."""

from functools import partial

from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.glm_vl.glm_45v_config import GLM45VModelConfig
from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel
from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder


class GLM45VModelBuilder(GPTModelBuilder):
    """Combine the GLM-4.5 MoE language model with the Hugging Face GLM-4V vision encoder."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GLM45VModel:
        """Build one GLM-4.5V pipeline stage, as ``GLM45VModelProvider.provide()`` does.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings and the vision encoder.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Virtual pipeline stage.

        Returns:
            A GLM-4.5V model stage.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` is unset.
        """
        model_config = self._model_config
        if not isinstance(model_config, GLM45VModelConfig):
            raise TypeError(f"Expected GLM45VModelConfig, got {type(model_config).__name__}.")
        if model_config.vision_config is None:
            raise ValueError("GLM-4.5V vision_config must be configured before model construction.")

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        model = GLM45VModel(
            model_config.transformer,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            build_language_model=partial(super().build_model, pg_collection),
            pg_collection=pg_collection,
        )
        if pre_process:
            # The Hugging Face vision encoder is created in fp32. The provider path casts it with
            # Float16Module; conversion builds without that wrapper, so cast it to params_dtype here.
            model.visual.to(dtype=model_config.params_dtype)

        if (
            model_config.freeze_language_model
            or model_config.freeze_vision_model
            or model_config.freeze_vision_projection
        ):
            model.freeze(
                freeze_language_model=model_config.freeze_language_model,
                freeze_vision_model=model_config.freeze_vision_model,
                freeze_vision_projection=model_config.freeze_vision_projection,
            )
        return model


__all__ = ["GLM45VModelBuilder"]
