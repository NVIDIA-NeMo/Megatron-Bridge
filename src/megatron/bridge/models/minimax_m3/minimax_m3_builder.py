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

"""Model builders for MiniMax-M3.

``MiniMaxM3ModelBuilder`` builds the language model of ``MiniMaxM3ModelProvider``, and
``MiniMaxM3VLModelBuilder`` the ``MiniMaxM3VLModel`` of ``MiniMaxM3VLModelProvider``: the vision
tower and projectors first, then the language model.

The router weights and expert biases and the vision patch embedding stay in FP32, as the checkpoint
stores them. They are marked so that ``Float16Module`` leaves them in FP32.
"""

from functools import partial

from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.minimax_m3.minimax_m3_bridge import _promote_router_weights_to_float32
from megatron.bridge.models.minimax_m3.minimax_m3_config import MiniMaxM3ModelConfig, MiniMaxM3VLModelConfig
from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VLModel


class MiniMaxM3ModelBuilder(GPTModelBuilder):
    """Build the MiniMax-M3 language model with its routers in FP32."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one MiniMax-M3 language-model pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings.
            post_process: Whether this stage owns the output layer.
            vp_stage: Virtual pipeline stage.

        Returns:
            A language-model stage whose router weights and expert biases are FP32.

        Raises:
            TypeError: If the builder was given another model config.
        """
        if not isinstance(self._model_config, MiniMaxM3ModelConfig):
            raise TypeError(f"Expected MiniMaxM3ModelConfig, got {type(self._model_config).__name__}.")
        model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        _promote_router_weights_to_float32([model])
        return model


class MiniMaxM3VLModelBuilder(MiniMaxM3ModelBuilder):
    """Combine the MiniMax-M3 language model with its vision tower and projectors."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MiniMaxM3VLModel:
        """Build one MiniMax-M3 vision-language pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings, the vision tower and the projectors.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Virtual pipeline stage.

        Returns:
            A MiniMax-M3 vision-language model stage.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` is unset.
        """
        model_config = self._model_config
        if not isinstance(model_config, MiniMaxM3VLModelConfig):
            raise TypeError(f"Expected MiniMaxM3VLModelConfig, got {type(model_config).__name__}.")
        if model_config.vision_config is None:
            raise ValueError("MiniMax-M3 vision_config must be configured before model construction.")

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        model = MiniMaxM3VLModel(
            model_config.transformer,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            build_language_model=partial(super().build_model, pg_collection),
            pg_collection=pg_collection,
        )
        if pre_process:
            # Checkpoint conversion builds the model without the Float16Module wrapper, which casts
            # a provider-built model to the model dtype. Cast the vision tower and projectors the
            # same way; the language model is already built in params_dtype.
            for module in (model.vision_tower, model.multi_modal_projector, model.patch_merge_mlp):
                convert_module_to_dtype_except_fp32_marked(module, model_config.params_dtype)
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


__all__ = ["MiniMaxM3ModelBuilder", "MiniMaxM3VLModelBuilder"]
