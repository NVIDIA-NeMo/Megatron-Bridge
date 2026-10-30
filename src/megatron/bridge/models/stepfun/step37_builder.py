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

"""Model builder for Step-3.7-Flash."""

from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.modelling_step37.transformer_block import get_step37_text_layer_spec
from megatron.bridge.models.stepfun.step37_config import Step37ModelConfig


class Step37ModelBuilder(GPTModelBuilder):
    """Combine the Step-3.5 language decoder with the Step-3.7 Perception-Encoder vision tower."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Step37Model:
        """Build one Step-3.7 pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings and the vision tower.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            A Step-3.7 model stage.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` is unset.
        """
        model_config = self._model_config
        if not isinstance(model_config, Step37ModelConfig):
            raise TypeError(f"Expected Step37ModelConfig, got {type(model_config).__name__}.")
        if model_config.vision_config is None:
            raise ValueError("Step-3.7 vision_config must be configured before model construction.")

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        # Resolve the decoder spec as Step37ModelProvider.provide() does.
        language_spec = model_config.transformer_layer_spec
        if language_spec is None:
            language_spec = get_step37_text_layer_spec
        if callable(language_spec):
            language_spec = language_spec(model_config, vp_stage=vp_stage)

        model = Step37Model(
            language_transformer_config=model_config.transformer,
            language_transformer_layer_spec=language_spec,
            vision_transformer_config=model_config.vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            model_config=model_config,
        )
        if vp_stage is not None:
            model.vp_stage = vp_stage

        # The vision tower and the projector are torch modules created in FP32. The provider path relies
        # on Float16Module to cast them with the rest of the model; conversion builds without that wrapper.
        if model.vision_model is not None:
            convert_module_to_dtype_except_fp32_marked(model.vision_model, model_config.params_dtype)
        if model.image_insert_embedding is not None:
            convert_module_to_dtype_except_fp32_marked(
                model.image_insert_embedding.align_projector, model_config.params_dtype
            )

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


__all__ = ["Step37ModelBuilder"]
