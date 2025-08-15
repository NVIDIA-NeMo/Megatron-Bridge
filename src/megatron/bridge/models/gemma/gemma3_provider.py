# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Union

import torch
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import ModuleSpec, TransformerConfig, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron.bridge.models.activations import openai_gelu
from megatron.bridge.models.gemma.gemma2_provider import TERowParallelLinearLayerNorm
from megatron.bridge.models.gpt_provider import GPTModelProvider


"""
The special design of gemma3:

- Special RMSNorm
  x * (1 + w) instead of x * w
  (x * w).to(dtype) instead of x.to(dtype) * w

- Post attention norm

- Post MLP norm

- Post word embedding scaling

- The 27B model sets custom q_scaling as 168^(-0.5), others use default head_dim^(-0.5)

- Interleaved attention layers
  Pattern: 5 local layers + 1 global layers

- Global layer and local layer calculate rope embedding differently
  rope_base:
    local: 10_000
    global: 1_000_000
  rope_scaling (linear):
    local: 1.0
    global: 8.0

vision:
- Post vision encoding norm
- Single layer linear vision projection
"""


def gemma3_layer_spec(config: GPTModelProvider) -> ModuleSpec:
    """Gemma3 custom layer spec."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Gemma3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Gemma3TEDotProductAttention,  # mixed gloabl/local attn
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    linear_proj=TERowParallelLinearLayerNorm,  # post attn RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,  # residual link
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,  # post mlp RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,  # residual link
        ),
    )


@dataclass
class Gemma3ModelProvider(GPTModelProvider):
    """Gemma3 Model Provider for Megatron-Bridge."""

    seq_length: int = 131_072

    # embedding
    vocab_size: int = 262_208
    position_embedding_type: str = "rope"
    rotary_base: tuple[int, int] = (10_000, 1_000_000)  # (local, global)
    share_embeddings_and_output_weights: bool = True

    rope_scaling_factor: float = 8.0  # Provide method requires this to be set

    # norm
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = True  # x * (1 + w)
    layernorm_epsilon: float = 1e-6

    # attention
    window_size: int = 512  # local attention window
    interleaved_attn_pattern: tuple[int, int] = (5, 1)  # (local, global)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    # Disable cuDNN attention since TE 1.8 does not support head dim > 128
    attention_backend: AttnBackend = AttnBackend.flash

    # mlp
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = openai_gelu

    # Do not change
    is_vision_language: bool = False
    flash_decode: bool = False
    gradient_accumulation_fusion: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = gemma3_layer_spec
    scatter_embedding_sequence_parallel: bool = True

    def provide(self, pre_process=None, post_process=None, vp_stage=None, tokenizer=None) -> "MCoreGPTModel":
        """Provide a configured MCore Gemma3 model."""
        if self.context_parallel_size > 1:
            raise ValueError("Context Parallel is not supported for Gemma3 model.")

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallel size is not yet supported for Gemma3 model."
        )

        rotary_base_local, rotary_base_global = self.rotary_base
        # Trick megatron's RotaryEmbedding to initialize the model successfully
        self.rotary_base = rotary_base_global
        model = super().provide(
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            tokenizer=tokenizer,
        )
        self.rotary_base = (rotary_base_local, rotary_base_global)

        # Replace model's embedding and rope with customized ones
        if hasattr(model, "embedding"):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=self,
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )

        model.rotary_pos_emb = Gemma3RotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            rope_scaling_factor=self.rope_scaling_factor,
            use_cpu_initialization=self.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
        )
        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()
        return model


@dataclass
class Gemma3ModelProvider1B(Gemma3ModelProvider):
    """Gemma3 1B Model Provider."""

    is_vision_language: bool = False
    num_layers: int = 26
    hidden_size: int = 1152
    num_attention_heads: int = 4
    num_query_groups: int = 1
    kv_channels: int = 256
    ffn_hidden_size: int = 6912
    window_size: int = 512
    rope_scaling_factor: float = 1.0  # no rope scaling
    seq_length: int = 32768
    bf16: bool = True
    vocab_size: int = 262_144


@dataclass
class Gemma3ModelProvider4B(Gemma3ModelProvider):
    """Gemma3 4B Model Provider."""

    is_vision_language: bool = True
    num_layers: int = 34
    hidden_size: int = 2560
    num_attention_heads: int = 8
    num_query_groups: int = 4
    kv_channels: int = 256
    ffn_hidden_size: int = 10240
    window_size: int = 1024
    rope_scaling_factor: float = 8.0


@dataclass
class Gemma3ModelProvider12B(Gemma3ModelProvider):
    """Gemma3 12B Model Provider."""

    is_vision_language: bool = True
    num_layers: int = 48
    hidden_size: int = 3840
    num_attention_heads: int = 16
    num_query_groups: int = 8
    kv_channels: int = 256
    ffn_hidden_size: int = 15360
    window_size: int = 1024
    rope_scaling_factor: float = 8.0


@dataclass
class Gemma3ModelProvider27B(Gemma3ModelProvider):
    """Gemma3 27B Model Provider."""

    is_vision_language: bool = True
    num_layers: int = 62
    hidden_size: int = 5376
    num_attention_heads: int = 32
    num_query_groups: int = 16
    kv_channels: int = 128
    softmax_scale: float = 1.0 / math.sqrt(168)  # only for 27B, (5376 // 32)^(-0.5)
    ffn_hidden_size: int = 21504
    window_size: int = 1024
    rope_scaling_factor: float = 8.0


class Gemma3LanguageModelEmbedding(LanguageModelEmbedding):
    """Gemma3 language token embedding.

    Adds a normalization to the embedding.
    """

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, tokentype_ids: Optional[int] = None
    ) -> torch.Tensor:
        """Calculate embedding and normalize"""
        embeddings = super().forward(input_ids, position_ids, tokentype_ids)
        embeddings = embeddings * (self.config.hidden_size**0.5)
        return embeddings


class Gemma3RotaryEmbedding(RotaryEmbedding):
    """Gemma3 position rope embedding.

    Calculates rope embeddings for both local and global attention layers.
    """

    def __init__(
        self,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        rotary_base: int = 1_000_000,
        rotary_base_local: int = 10_000,
        **kwargs,
    ):
        # The rope scaling in RotaryEmbedding is not linear scaling,
        # so this flag must be off. Will calculate linear scaling below.
        assert rope_scaling is False

        # Get inv_freq for global attention layers
        super().__init__(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base,
            **kwargs,
        )
        self.inv_freq /= rope_scaling_factor

        # Setup Rotary Embedding for local attentions
        self.rope_local = RotaryEmbedding(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base_local,
            **kwargs,
        )

    @lru_cache(maxsize=32)
    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> torch.Tensor:
        """Get global and local rope embedding"""
        rope_global = super().forward(max_seq_len, offset, packed_seq)
        rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq)
        return rope_local, rope_global


def _is_local_attn_layer(layer_number: int, layer_pattern: tuple[int, int]) -> bool:
    pattern_size = sum(layer_pattern)
    return layer_number % pattern_size != 0


class Gemma3SelfAttention(SelfAttention):
    """Gemma3 self attention.

    Uses local rope embedding for local layers,
    global rope embedding for global layers.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Switch to either local or global rope embedding before forward"""
        assert isinstance(rotary_pos_emb, tuple)
        assert rotary_pos_cos is None and rotary_pos_sin is None

        if _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern):
            final_rotary_pos_emb = rotary_pos_emb[0]
        else:
            final_rotary_pos_emb = rotary_pos_emb[1]
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=final_rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )


class Gemma3TEDotProductAttention(TEDotProductAttention):
    """Gemma3 core attention.

    Switches between global and local sliding window attention
    based on the layer_number and pre-defined layer pattern.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        # Overwrite config.window_size based on layer_number
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            # local attention, (q, k)
            config.window_size = (config.window_size, 0)
        else:
            # global attention
            config.window_size = None

        # The VL model calculates mask manually
        if config.is_vision_language:
            attn_mask_type = AttnMaskType.arbitrary

        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )
