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

"""Model builder for GLM-5.3-Flash.

It constructs ``GLM53FlashModel`` as the legacy ``GLM53FlashModelProvider`` does: the Hugging Face
vision tower first on the pre-process stage, then the Hybrid language model.
"""

from functools import partial

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashModel
from megatron.bridge.models.glm5next.glm53flash_config import GLM53FlashModelConfig
from megatron.bridge.models.hybrid.hybrid_builder import BridgeHybridModelBuilder
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


class GLM53FlashModelBuilder(BridgeHybridModelBuilder):
    """Combine the GLM-5.3-Flash Hybrid language model with the Hugging Face vision tower."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GLM53FlashModel:
        """Build one GLM-5.3-Flash pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings and the vision tower.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Virtual pipeline stage.

        Returns:
            A GLM-5.3-Flash model stage.

        Raises:
            TypeError: If the builder was given another model config.
        """
        model_config = self._model_config
        if not isinstance(model_config, GLM53FlashModelConfig):
            raise TypeError(f"Expected GLM53FlashModelConfig, got {type(model_config).__name__}.")

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        model = GLM53FlashModel(
            model_config.transformer,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            build_language_model=partial(self._build_language_model, pg_collection),
            pg_collection=pg_collection,
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

    def _build_language_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool,
        post_process: bool,
        vp_stage: int | None,
    ) -> HybridModel:
        """Build the Hybrid language model with the arguments ``HybridModelProvider.provide`` passes.

        ``GLM53FlashModel`` inserts vision features into full-sequence embeddings and scatters them
        for sequence parallelism itself, so the embedding does not scatter. Megatron Core's Hybrid
        builder has no argument for that, so this builds the model directly.
        """
        model_config = self._model_config
        if model_config.vocab_size is None:
            raise ValueError("vocab_size must be configured before building GLM-5.3-Flash.")
        vocab_size = model_config.vocab_size
        if model_config.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size,
                model_config.make_vocab_size_divisible_by,
                model_config.transformer.tensor_model_parallel_size,
            )
        return HybridModel(
            config=model_config.transformer,
            hybrid_stack_spec=self._resolve_hybrid_stack_spec(),
            vocab_size=vocab_size,
            max_sequence_length=model_config.seq_length,
            hybrid_layer_pattern=model_config.hybrid_layer_pattern,
            fp16_lm_cross_entropy=model_config.fp16_lm_cross_entropy,
            **logit_dtype_kwarg(HybridModel, model_config.logit_dtype),
            parallel_output=model_config.parallel_output,
            share_embeddings_and_output_weights=model_config.share_embeddings_and_output_weights,
            position_embedding_type=model_config.position_embedding_type,
            rotary_percent=model_config.rotary_percent,
            rotary_base=model_config.rotary_base,
            seq_len_interpolation_factor=model_config.seq_len_interpolation_factor,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            scatter_embedding_sequence_parallel=False,
        )


__all__ = ["GLM53FlashModelBuilder"]
