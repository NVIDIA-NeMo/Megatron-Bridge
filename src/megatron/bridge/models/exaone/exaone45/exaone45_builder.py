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

"""Model builder for EXAONE 4.5 vision-language models."""

import inspect

from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.exaone.exaone45.exaone45_config import Exaone45ModelConfig
from megatron.bridge.models.exaone.exaone45.exaone45_provider import exaone_45_transformer_layer_spec
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.model import Exaone45Model
from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder


class Exaone45ModelBuilder(GPTModelBuilder):
    """Combine the EXAONE Post-LN language decoder with the EXAONE 4.5 vision encoder."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Exaone45Model:
        """Build one EXAONE 4.5 pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings and the vision encoder.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            An EXAONE 4.5 model stage.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` is unset.
        """
        model_config = self._model_config
        if not isinstance(model_config, Exaone45ModelConfig):
            raise TypeError(f"Expected Exaone45ModelConfig, got {type(model_config).__name__}.")
        if model_config.vision_config is None:
            raise ValueError("EXAONE 4.5 vision_config must be configured before model construction.")

        language_spec = model_config.transformer_layer_spec
        if language_spec is None:
            language_spec = exaone_45_transformer_layer_spec(model_config)
        elif not isinstance(language_spec, ModuleSpec):
            if "vp_stage" in inspect.signature(language_spec).parameters:
                language_spec = language_spec(model_config, vp_stage=vp_stage)
            else:
                language_spec = language_spec(model_config)

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        model = Exaone45Model(
            language_transformer_config=model_config.transformer,
            language_transformer_layer_spec=language_spec,
            vision_transformer_config=model_config.vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            model_config=model_config,
        )
        if model.vision_model is not None:
            # get_vision_model_config() leaves the vision encoder in fp32. The provider path relies on
            # Float16Module to cast it with the rest of the model; conversion builds without that wrapper.
            model.vision_model.to(dtype=model_config.params_dtype)

        if (
            model_config.freeze_language_model
            or model_config.freeze_vision_model
            or model_config.freeze_vision_projection
            or model_config.freeze_mtp_model
        ):
            model.freeze(
                freeze_language_model=model_config.freeze_language_model,
                freeze_vision_model=model_config.freeze_vision_model,
                freeze_vision_projection=model_config.freeze_vision_projection,
                freeze_mtp_model=model_config.freeze_mtp_model,
            )
        return model


__all__ = ["Exaone45ModelBuilder"]
