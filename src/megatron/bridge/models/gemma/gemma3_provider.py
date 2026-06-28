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

"""Legacy provider compatibility for Gemma3 models."""

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from megatron.core.activations import fast_gelu
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gemma.modeling_gemma3 import (
    Gemma3LanguageModelEmbedding,
    Gemma3RotaryEmbedding,
    Gemma3SelfAttention,
    Gemma3TEDotProductAttention,
    TERowParallelLinearLayerNorm,
    _is_local_attn_layer,
    gemma3_layer_spec,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class Gemma3ModelProvider(GPTModelProvider):
    """Configuration and legacy provider for Megatron-Core Gemma3 models."""

    seq_length: int = 131_072
    position_embedding_type: str = "rope"
    rotary_base: tuple = (10_000, 1_000_000)
    share_embeddings_and_output_weights: bool = True
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = True
    layernorm_epsilon: float = 1e-6
    qk_layernorm: bool = True
    window_size: tuple = 512
    interleaved_attn_pattern: tuple = (5, 1)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    rope_scaling_factor: float = 1.0
    attention_backend: AttnBackend = AttnBackend.flash
    softmax_scale: float = 1.0 / math.sqrt(256)
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = fast_gelu
    is_vision_language: bool = False
    flash_decode: bool = False
    transformer_layer_spec: ModuleSpec | Callable[["Gemma3ModelProvider"], ModuleSpec] = field(
        default_factory=lambda: gemma3_layer_spec
    )
    scatter_embedding_sequence_parallel: bool = True
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MCoreGPTModel:
        """Instantiate one legacy provider-backed Gemma3 model stage."""
        rotary_base_local, rotary_base_global = self.rotary_base
        self.rotary_base = rotary_base_local
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        self.rotary_base = (rotary_base_local, rotary_base_global)
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


__all__ = [
    "Gemma3LanguageModelEmbedding",
    "Gemma3ModelProvider",
    "Gemma3RotaryEmbedding",
    "Gemma3SelfAttention",
    "Gemma3TEDotProductAttention",
    "TERowParallelLinearLayerNorm",
    "_is_local_attn_layer",
    "gemma3_layer_spec",
]
