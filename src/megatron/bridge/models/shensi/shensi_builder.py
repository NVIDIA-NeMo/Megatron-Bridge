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

"""Builder-backed configuration and builder for Shensi on Megatron-Core's ``HybridModel``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron.bridge.models.common import ModelConfigOverrideMixin
from megatron.bridge.models.hybrid.hybrid_builder import HybridModelBuilder
from megatron.bridge.models.hybrid.hybrid_builder import HybridModelConfig as BridgeHybridModelConfig
from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig
from megatron.bridge.models.shensi.modeling_shensi import ShensiModel
from megatron.bridge.models.shensi.shensi_hybrid import build_shensi_model


@dataclass(kw_only=True)
class ShensiModelConfig(ModelConfigOverrideMixin, BridgeHybridModelConfig):
    """Complete builder configuration for Shensi.

    The nested ``transformer`` carries the shensi decoder fields (`ShensiTransformerConfig`); the outer
    fields are ``HybridModel``'s surface - vocabulary, sequence length, rope, the layer pattern, ...
    """

    builder: ClassVar[str] = "megatron.bridge.models.shensi.shensi_builder.ShensiModelBuilder"
    transformer_config_class: ClassVar[type[ShensiTransformerConfig]] = ShensiTransformerConfig


class ShensiModelBuilder(HybridModelBuilder):
    """Build one Shensi pipeline stage on MCore's ``HybridModel``."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> ShensiModel:
        """Build a single Shensi stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embedding.
            post_process: Whether this stage owns the output layer and the MTP heads.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            The constructed Shensi stage.
        """
        model_config = self._model_config
        if not isinstance(model_config, ShensiModelConfig):
            raise TypeError(f"Expected ShensiModelConfig, got {type(model_config).__name__}.")
        transformer = model_config.transformer
        if not isinstance(transformer, ShensiTransformerConfig):
            raise TypeError(f"Expected ShensiTransformerConfig for the transformer, got {type(transformer).__name__}.")
        if model_config.vocab_size is None:
            raise ValueError("Shensi vocab_size must be configured before model construction.")

        if model_config.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                model_config.vocab_size,
                model_config.make_vocab_size_divisible_by,
                transformer.tensor_model_parallel_size,
            )
        else:
            padded_vocab_size = model_config.vocab_size
        return build_shensi_model(
            transformer,
            pg_collection,
            vocab_size=padded_vocab_size,
            max_sequence_length=model_config.seq_length,
            vp_stage=vp_stage,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=model_config.fp16_lm_cross_entropy,
            logit_dtype=model_config.logit_dtype,
            parallel_output=model_config.parallel_output,
            share_embeddings_and_output_weights=model_config.share_embeddings_and_output_weights,
            position_embedding_type=model_config.position_embedding_type,
            rotary_percent=model_config.rotary_percent,
            rotary_base=model_config.rotary_base,
            scatter_embedding_sequence_parallel=model_config.scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=model_config.seq_len_interpolation_factor,
        )


__all__ = ["ShensiModelBuilder", "ShensiModelConfig"]
