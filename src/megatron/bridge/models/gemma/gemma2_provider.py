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

"""Legacy provider compatibility for Gemma2 models."""

from dataclasses import dataclass
from typing import Callable

from megatron.core.activations import fast_gelu
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.gemma.modeling_gemma2 import (
    _HAVE_FLEX_ATTN,
    Gemma2DotProductAttention,
    Gemma2FlexDotProductAttention,
    Gemma2OutputLayer,
    _create_flex_block_mask,
    _flex_attn_func,
    _get_softcap_score_mod,
    gemma2_layer_spec,
    get_swa,
    logit_softcapping,
)
from megatron.bridge.models.gemma.modules import EmbeddingScalingMixin, extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class Gemma2ModelProvider(GPTModelProvider):
    """Configuration and legacy provider for Megatron-Core Gemma2 models."""

    normalization: str = "RMSNorm"
    activation_func: Callable = fast_gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 8192
    kv_channels: int = 256
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = True
    layernorm_zero_centered_gamma: bool = True
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 10000
    window_size: tuple[int, int] = (4095, 0)
    vocab_size: int = 256000
    transformer_layer_spec: ModuleSpec | Callable[[GPTModelProvider], ModuleSpec] = gemma2_layer_spec
    query_pre_attn_scalar: int = 224
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MCoreGPTModel:
        """Instantiate one legacy provider-backed Gemma2 model stage."""
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if is_vp_first_stage(
            vp_stage=vp_stage, vp_size=self.virtual_pipeline_model_parallel_size
        ) and is_pp_first_stage(self._pg_collection.pp):
            extend_instance(model.embedding, EmbeddingScalingMixin)
        if is_vp_last_stage(vp_stage=vp_stage, vp_size=self.virtual_pipeline_model_parallel_size) and is_pp_last_stage(
            self._pg_collection.pp
        ):
            extend_instance(model.output_layer, Gemma2OutputLayer)
        return model


__all__ = [
    "Gemma2DotProductAttention",
    "Gemma2FlexDotProductAttention",
    "Gemma2ModelProvider",
    "Gemma2OutputLayer",
    "_HAVE_FLEX_ATTN",
    "_create_flex_block_mask",
    "_flex_attn_func",
    "_get_softcap_score_mod",
    "gemma2_layer_spec",
    "get_swa",
    "logit_softcapping",
]
