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

from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.hybrid import HybridModelBuilder
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerModel,
    get_muse_glimmer_hybrid_stack_spec,
)
from megatron.bridge.models.muse_glimmer.muse_glimmer_config import MuseGlimmerModelConfig


class MuseGlimmerModelBuilder(HybridModelBuilder):
    """Construct the complete Muse Glimmer model on MCore HybridModel."""

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

        if model_config.vocab_size is None:
            raise ValueError("Muse Glimmer vocab_size must be configured before model construction.")
        if model_config.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                model_config.vocab_size,
                model_config.make_vocab_size_divisible_by,
                model_config.transformer.tensor_model_parallel_size,
            )
        else:
            padded_vocab_size = model_config.vocab_size

        pre_process = pre_process if pre_process is not None else is_pp_first_stage(pg_collection.pp)
        post_process = post_process if post_process is not None else is_pp_last_stage(pg_collection.pp)
        return MuseGlimmerModel(
            model_config,
            get_muse_glimmer_hybrid_stack_spec(model_config.transformer),
            padded_vocab_size,
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )


__all__ = ["MuseGlimmerModelBuilder"]
