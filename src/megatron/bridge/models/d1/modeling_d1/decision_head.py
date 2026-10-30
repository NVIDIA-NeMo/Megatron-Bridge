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

"""Omni's marker readout composed from MCore attention, MLP and TE norms."""

from dataclasses import replace

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.nn import functional as F


class D1DecisionHeadLayer(MegatronModule):
    """Released pre-LayerNorm, ReLU encoder composition without RoPE.

    Args:
        config: Native head configuration, with standard LayerNorm and biases.
        layer_number: One-based native attention index.
    """

    def __init__(self, config: TransformerConfig, *, layer_number: int) -> None:
        super().__init__(config)
        self.norm1 = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.norm2 = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attn = SelfAttention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.arbitrary,
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        )
        self.mlp = MLP(
            config=config,
            ffn_hidden_size=config.ffn_hidden_size,
            submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
        )

    def forward(self, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return sequence-first states after two residual sublayers.

        Args:
            hidden_states: States [length, batch, width].
            attention_mask: Forbidden positions [batch, 1, length, length].

        Returns:
            Residual states with the same shape.
        """
        mixed, bias = self.self_attn(self.norm1(hidden_states), attention_mask)
        if bias is not None:
            mixed = mixed + bias
        residual = hidden_states + mixed
        mixed, bias = self.mlp(self.norm2(residual))
        if bias is not None:
            mixed = mixed + bias
        return residual + mixed


class D1DecisionHead(MegatronModule):
    """Native Omni head preserving marker and question-type semantics.

    Logits use the internal option order supplied by the prompt. Noul prompts
    use [false, true]; public [yes, no] order and inference temperature are
    applied by the caller, outside the trainable head.

    Args:
        config: Language configuration supplying width, dtype and initialization.
        layers: Released number of head transformer layers.
    """

    def __init__(self, config: TransformerConfig, *, layers: int) -> None:
        if config.hidden_size % 64 or layers <= 0:
            raise ValueError("Omni requires a width divisible by 64 and a positive head layer count")
        if (
            config.tensor_model_parallel_size != 1
            or config.pipeline_model_parallel_size != 1
            or config.context_parallel_size != 1
            or config.sequence_parallel
        ):
            raise ValueError("D1DecisionHead currently requires TP=PP=CP=1 without sequence parallelism")
        head_config = replace(
            config,
            num_layers=layers,
            num_attention_heads=config.hidden_size // 64,
            num_query_groups=config.hidden_size // 64,
            kv_channels=64,
            ffn_hidden_size=4 * config.hidden_size,
            normalization="LayerNorm",
            layernorm_epsilon=1e-5,
            layernorm_zero_centered_gamma=False,
            fused_residual_rmsnorm=False,
            qk_layernorm=False,
            qk_l2_norm=False,
            gated_linear_unit=False,
            activation_func=F.relu,
            add_bias_linear=True,
            add_qkv_bias=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=False,
            bias_activation_fusion=False,
            use_te_activation_func=False,
            softmax_scale=None,
            no_rope_freq=None,
        )
        # Bridge providers defer post-init. Recompute the MCore head fields
        # after replacing the language attention/MLP dimensions and flags.
        TransformerConfig.__post_init__(head_config)
        super().__init__(head_config)
        width = head_config.hidden_size
        device = (
            torch.device("cpu")
            if head_config.use_cpu_initialization
            else torch.device("cuda", torch.cuda.current_device())
        )
        self.type_emb = nn.Embedding(3, width, device=device, dtype=head_config.params_dtype)
        if head_config.perform_initialization:
            head_config.init_method(self.type_emb.weight)
        self.type_emb.weight.allreduce = True
        self.type_emb.weight.tensor_model_parallel = False
        self.layers = nn.ModuleList(
            D1DecisionHeadLayer(head_config, layer_number=index + 1) for index in range(layers)
        )
        self.scorer_norm = TENorm(head_config, width, eps=1e-5)
        self.scorer_hidden = TELinear(
            width,
            width,
            parallel_mode="duplicated",
            config=head_config,
            init_method=head_config.init_method,
            bias=True,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        self.scorer_output = TELinear(
            width,
            1,
            parallel_mode="duplicated",
            config=head_config,
            init_method=head_config.output_layer_init_method,
            bias=True,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        valid_tokens: torch.Tensor,
        marker_positions: torch.Tensor,
        marker_mask: torch.Tensor,
        question_types: torch.Tensor,
    ) -> torch.Tensor:
        """Return float32 option logits from valid decision marker positions.

        Args:
            hidden_states: Batch-first encoder states [batch, length, width].
            valid_tokens: Boolean real-token mask [batch, length].
            marker_positions: Long option-marker indices [batch, options].
            marker_mask: Boolean real-option mask [batch, options].
            question_types: Long choice=0, score=1 or noul=2 IDs [batch].

        Returns:
            Float32 logits [batch, options], with absent options set to -1e4.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states must have shape [batch, length, hidden_size]")
        batch, length, _ = hidden_states.shape
        if valid_tokens.shape != (batch, length) or valid_tokens.dtype != torch.bool:
            raise ValueError("valid_tokens must be boolean with shape [batch, length]")
        if marker_positions.ndim != 2 or marker_positions.shape[0] != batch or marker_positions.dtype != torch.long:
            raise ValueError("marker_positions must be long with shape [batch, options]")
        if marker_mask.shape != marker_positions.shape or marker_mask.dtype != torch.bool:
            raise ValueError("marker_mask must be boolean with the marker position shape")
        if question_types.shape != (batch,) or question_types.dtype != torch.long:
            raise ValueError("question_types must be long with shape [batch]")
        if any(
            tensor.device != hidden_states.device
            for tensor in (valid_tokens, marker_positions, marker_mask, question_types)
        ):
            raise ValueError("states and head metadata must share a device")
        if torch.any(~valid_tokens.any(dim=1)) or torch.any(~marker_mask.any(dim=1)):
            raise ValueError("every row needs at least one real token and option")
        if torch.any((question_types < 0) | (question_types > 2)):
            raise ValueError("question types must be choice=0, score=1 or noul=2")
        if torch.any((marker_positions < 0) | (marker_positions >= length)):
            raise ValueError("marker positions must lie inside the sequence")
        if torch.any(marker_mask & ~valid_tokens.gather(1, marker_positions)):
            raise ValueError("real option markers must point to real tokens")
        values = (hidden_states + self.type_emb(question_types)[:, None, :]).transpose(0, 1)
        mask = (~valid_tokens[:, None, None, :]).expand(batch, 1, length, length)
        for layer in self.layers:
            values = layer(values, attention_mask=mask)
        values = values.transpose(0, 1).gather(1, marker_positions[..., None].expand(-1, -1, self.config.hidden_size))
        values, _ = self.scorer_hidden(self.scorer_norm(values))
        logits, _ = self.scorer_output(F.gelu(values))
        return logits.squeeze(-1).float().masked_fill(~marker_mask, -1e4)
