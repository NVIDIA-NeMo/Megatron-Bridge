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

"""Model builder for Muse Glimmer."""

from __future__ import annotations

from dataclasses import replace

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.muse_glimmer.model_config import MuseGlimmerModelConfig
from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerModel,
    customize_muse_glimmer_language_model,
    get_muse_glimmer_layer_spec,
)


class MuseGlimmerModelBuilder(GPTModelBuilder):
    """Construct the replicated vision stack around a builder-backed MCore decoder."""

    def __init__(self, model_config: MuseGlimmerModelConfig) -> None:
        super().__init__(model_config)

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MuseGlimmerModel:
        """Build one Muse Glimmer pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns embeddings and the vision stack.
            post_process: Whether this stage owns the language-model output head.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            A combined Muse Glimmer model stage.
        """
        model_config = self._model_config
        if not isinstance(model_config, MuseGlimmerModelConfig):
            raise TypeError(f"Expected MuseGlimmerModelConfig, got {type(model_config).__name__}.")

        language_config = replace(
            model_config,
            transformer_layer_spec=get_muse_glimmer_layer_spec(model_config.transformer),
        )
        language_model = GPTModelBuilder(language_config).build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        customize_muse_glimmer_language_model(language_model)
        return MuseGlimmerModel(
            model_config,
            language_model,
            pre_process=language_model.pre_process,
            post_process=language_model.post_process,
            vp_stage=vp_stage,
        )


__all__ = ["MuseGlimmerModelBuilder"]
