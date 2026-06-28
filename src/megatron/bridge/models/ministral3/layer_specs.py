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

"""Provider-neutral Ministral 3 attention and layer specification."""

from functools import partial

import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


class MinistralTEDotProductAttention(TEDotProductAttention):
    """TE attention with Ministral's Llama 4 position-dependent query scale."""

    def __init__(
        self,
        config: object,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        beta: float = 0.0,
        max_position_embeddings: int = 16384,
        **kwargs: object,
    ) -> None:
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.beta = beta
        self.max_position_embeddings = max_position_embeddings

    @staticmethod
    def _get_llama_4_attn_scale(
        positions_ids: torch.Tensor,
        beta: float,
        max_position_embeddings: int,
        query_shape: tuple[int, ...],
    ) -> torch.Tensor:
        scaling = 1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings))
        for _ in range(len(query_shape) - 1):
            scaling = scaling.unsqueeze(-1)
        return scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType,
        **kwargs: object,
    ) -> torch.Tensor:
        positions_ids = torch.arange(query.shape[0], device=query.device)
        query *= self._get_llama_4_attn_scale(
            positions_ids,
            self.beta,
            self.max_position_embeddings,
            tuple(query.shape),
        ).to(query.dtype)
        return super().forward(query, key, value, attention_mask, attn_mask_type, **kwargs)


def ministral_layer_spec(config: BridgeGPTModelConfig) -> ModuleSpec:
    """Build the Ministral 3 TE layer spec."""
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
    )
    layer_spec.submodules.self_attention.submodules.core_attention = partial(
        MinistralTEDotProductAttention,
        beta=config.llama_4_scaling_beta,
        max_position_embeddings=config.llama_4_original_max_position_embeddings,
    )
    return layer_spec


__all__ = ["MinistralTEDotProductAttention", "ministral_layer_spec"]
