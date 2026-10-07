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

import logging
from dataclasses import dataclass
from typing import Callable, Literal

from megatron.core.models.bert.bert_layer_specs import get_bert_layer_with_transformer_engine_spec
from megatron.core.models.bert.bert_model import BertModel as MCoreBertModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


logger = logging.getLogger(__name__)


@dataclass
class BertModelProvider(TransformerConfig, ModelProviderMixin[MCoreBertModel]):
    """Model provider for Megatron-Core's stock encoder-only BERT model.

    Note this targets Megatron-Core's `BertModel`, which applies LayerNorm
    *before* self-attention/MLP (Pre-LayerNorm) rather than the original BERT
    paper's Post-LayerNorm placement used by HuggingFace's vanilla
    ``transformers.BertModel``. The HuggingFace architecture that matches this
    provider exactly is ``MegatronBertForMaskedLM`` (see `bert_bridge.py`).
    """

    # No NSP/pooler head by default: `MegatronBertForMaskedLM` does not include one
    # (`add_pooling_layer=False` on the HF side), and it is not needed for masked-LM-only use.
    add_binary_head: bool = False
    num_tokentypes: int = 2
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: float | None = None
    return_embeddings: bool = False

    make_vocab_size_divisible_by: int = 128
    max_position_embeddings: int = 512
    vocab_size: int | None = None
    should_pad_vocab: bool = False

    transformer_layer_spec: ModuleSpec | Callable[[], ModuleSpec] = get_bert_layer_with_transformer_engine_spec

    _pg_collection: ProcessGroupCollection | None = None

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MCoreBertModel:
        """Configure and instantiate a Megatron Core BertModel based on this configuration.

        Args:
            pre_process: Whether to include the embedding layer.
            post_process: Whether to include the masked-LM head.
            vp_stage: Virtual pipeline stage.

        Returns:
            Configured Megatron Core BERT model.
        """
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec()

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        model = MCoreBertModel(
            config=self,
            num_tokentypes=self.num_tokentypes,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.max_position_embeddings,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            add_binary_head=self.add_binary_head,
            return_embeddings=self.return_embeddings,
            pre_process=pre_process if pre_process is not None else is_pp_first_stage(self._pg_collection.pp),
            post_process=post_process if post_process is not None else is_pp_last_stage(self._pg_collection.pp),
            vp_stage=vp_stage,
            pg_collection=self._pg_collection,
        )

        return model
