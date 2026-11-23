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

"""Model builders for Gemma 4 vision-language models.

Both builders construct the existing ``Gemma4VLModel`` as the legacy ``Gemma4VLModelProvider``
and ``Gemma4DenseVLProvider`` do: the Hugging Face vision (and audio) encoders first, then the
language model from the Gemma 4 text builder.
"""

from collections.abc import Callable
from functools import partial

from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.gemma.gemma4_builder import Gemma4DenseModelBuilder, Gemma4ModelBuilder
from megatron.bridge.models.gemma_vl.gemma4_vl_config import Gemma4DenseVLModelConfig, Gemma4VLModelConfig
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel


def _build_gemma4_vl_model(
    model_config: Gemma4VLModelConfig | Gemma4DenseVLModelConfig,
    build_language_model: Callable[..., GPTModel],
    pg_collection: ProcessGroupCollection,
    pre_process: bool | None,
    post_process: bool | None,
    vp_stage: int | None,
) -> Gemma4VLModel:
    if model_config.vision_config is None or model_config.text_config is None:
        raise ValueError("Gemma 4 vision_config and text_config must be configured before model construction.")

    vp_size = model_config.virtual_pipeline_model_parallel_size
    if pre_process is None:
        pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
    if post_process is None:
        post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

    model = Gemma4VLModel(
        model_config.transformer,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        build_language_model=build_language_model,
        pg_collection=pg_collection,
    )
    if model_config.freeze_language_model or model_config.freeze_vision_model or model_config.freeze_vision_projection:
        model.freeze(
            freeze_language_model=model_config.freeze_language_model,
            freeze_vision_model=model_config.freeze_vision_model,
            freeze_vision_projection=model_config.freeze_vision_projection,
        )
    return model


class Gemma4VLModelBuilder(Gemma4ModelBuilder):
    """Combine the Gemma 4 MoE language model with the Hugging Face vision encoder."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Gemma4VLModel:
        """Build one Gemma 4 MoE vision-language pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings and the vision encoder.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Virtual pipeline stage.

        Returns:
            A Gemma 4 vision-language model stage.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` or ``text_config`` is unset.
        """
        if not isinstance(self._model_config, Gemma4VLModelConfig):
            raise TypeError(f"Expected Gemma4VLModelConfig, got {type(self._model_config).__name__}.")
        return _build_gemma4_vl_model(
            self._model_config,
            partial(super().build_model, pg_collection),
            pg_collection,
            pre_process,
            post_process,
            vp_stage,
        )


class Gemma4DenseVLModelBuilder(Gemma4DenseModelBuilder):
    """Combine the Gemma 4 dense language model with the Hugging Face vision and audio encoders."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Gemma4VLModel:
        """Build a Gemma 4 dense vision-language model.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether to build the embeddings and the vision encoder.
            post_process: Whether to build the language-model output layer.
            vp_stage: Virtual pipeline stage; Gemma 4 dense supports none.

        Returns:
            A Gemma 4 dense vision-language model.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` or ``text_config`` is unset.
        """
        if not isinstance(self._model_config, Gemma4DenseVLModelConfig):
            raise TypeError(f"Expected Gemma4DenseVLModelConfig, got {type(self._model_config).__name__}.")
        return _build_gemma4_vl_model(
            self._model_config,
            partial(super().build_model, pg_collection),
            pg_collection,
            pre_process,
            post_process,
            vp_stage,
        )


__all__ = ["Gemma4DenseVLModelBuilder", "Gemma4VLModelBuilder"]
