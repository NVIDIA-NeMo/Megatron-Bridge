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

"""Model builder for Kimi K2.5 VL.

``KimiK25VLModelBuilder`` constructs the existing ``KimiK25VLModel`` as the legacy
``KimiK25VLModelProvider`` does: the vision tower and the projector first, from the checkpoint's own
code, then the language model from the GPT builder.
"""

from functools import partial

from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.kimi_vl.kimi_k25_vl_config import KimiK25VLModelConfig
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel


class KimiK25VLModelBuilder(GPTModelBuilder):
    """Combine the Kimi K2 MoE language model with the MoonViT vision tower and the projector."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> KimiK25VLModel:
        """Build one Kimi K2.5 VL pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings, the vision tower and the projector.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            A Kimi K2.5 VL model stage.

        Raises:
            TypeError: If the builder was given another model config.
        """
        model_config = self._model_config
        if not isinstance(model_config, KimiK25VLModelConfig):
            raise TypeError(f"Expected KimiK25VLModelConfig, got {type(model_config).__name__}.")

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        model = KimiK25VLModel(
            model_config.transformer,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            model_config=model_config,
            build_language_model=partial(super().build_model, pg_collection),
            pg_collection=pg_collection,
        )
        if pre_process:
            # The Hugging Face modules are built in FP32. The provider path relies on Float16Module to
            # cast them with the rest of the model; conversion builds without that wrapper.
            convert_module_to_dtype_except_fp32_marked(model.vision_tower, model_config.params_dtype)
            convert_module_to_dtype_except_fp32_marked(model.mm_projector, model_config.params_dtype)

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


__all__ = ["KimiK25VLModelBuilder"]
