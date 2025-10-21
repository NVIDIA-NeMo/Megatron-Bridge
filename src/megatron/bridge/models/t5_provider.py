# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from megatron.core.models.T5.t5_model import T5Model as MCoreT5Model
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


logger = logging.getLogger(__name__)


def transformer_engine_layer_spec(encoder_config: "T5ModelProvider", decoder_config: "T5ModelProvider") -> ModuleSpec:
    """Spec for T5 when using transformer_engine mcore implementation"""
    from megatron.core.models.T5.t5_spec import (
        get_t5_decoder_with_transformer_engine_block_spec,
        get_t5_encoder_with_transformer_engine_block_spec,
    )

    en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]


def local_layer_spec(encoder_config: "T5ModelProvider", decoder_config: "T5ModelProvider") -> ModuleSpec:
    """Spec for T5 when using local mcore implementation"""
    from megatron.core.models.T5.t5_spec import (
        get_t5_decoder_with_local_block_spec,
        get_t5_encoder_with_local_block_spec,
    )

    en_block_spec = get_t5_encoder_with_local_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_local_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]


@dataclass
class T5ModelProvider(TransformerConfig, ModelProviderMixin[MCoreT5Model]):
    """Model config for T5 model. Adpated from megatron.core.models.t5.t5_model.T5Model"""

    encoder_num_layers: int | None = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    apply_rope_fusion: bool = True
    max_position_embeddings: int = 512
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: float | None = None
    seq_length: int = 512
    seq_length_dec: int = 128
    encoder_pipeline_model_parallel_size: int = 0
    attention_softmax_in_fp32: float = False
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    deallocate_pipeline_outputs: bool = True
    num_moe_experts: int | None = None
    recompute_num_layers: int = 1
    distribute_saved_activations: bool = False
    enable_autocast: bool = False

    transformer_layer_spec: ModuleSpec | Callable[["T5ModelProvider"], ModuleSpec] = (
        transformer_engine_layer_spec
    )

    vocab_size: int | None = None
    should_pad_vocab: bool = False
    tp_comm_overlap_cfg: str | dict[str, Any] | None = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreT5Model:
        """Setup the T5 Model based on config definition."""

        assert self.virtual_pipeline_model_parallel_size is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in T5 "
            "due to upstream MCore T5Model API dependency"
        )

        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        from megatron.core import parallel_state

        encoder_config = copy.deepcopy(self)
        encoder_config.num_layers = self.encoder_num_layers
        if self.pipeline_model_parallel_size > 1:
            assert self.encoder_pipeline_model_parallel_size > 0, "Need to know how to shard the encoder & decoder."
            encoder_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(encoder_config=encoder_config, decoder_config=self)

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        model = MCoreT5Model(
            config=self,
            encoder_config=encoder_config,
            transformer_encoder_layer_spec=transformer_layer_spec[0],
            transformer_decoder_layer_spec=transformer_layer_spec[1],
            vocab_size=padded_vocab_size,
            max_sequence_length=self.max_position_embeddings,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )

        return model
