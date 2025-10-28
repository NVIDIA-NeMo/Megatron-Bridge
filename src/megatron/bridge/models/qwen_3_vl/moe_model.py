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

import torch
from torch import nn

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer

from .model import Qwen3VLModel, Qwen3VLTextDecoderLayer, Qwen3VLTextModel


class Qwen3VLMoETextExperts(MegatronModule):
    """
    MoE experts, adapted from HuggingFace's Qwen3VLMoETextExperts.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.num_experts = self.config.num_moe_experts
        self.intermediate_size = self.config.moe_ffn_hidden_size
        self.hidden_size = self.config.hidden_size
        self.expert_dim = self.intermediate_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty((self.num_experts, self.expert_dim, self.hidden_size))
        )
        self.act_fn = self.config.activation_func

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)

        if self.training:
            next_states = torch.zeros_like(
                hidden_states, dtype=hidden_states.dtype, device=hidden_states.device
            )
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    router_indices, num_classes=self.num_experts
                )
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                gated_output = up * self.act_fn(gate)
                out = gated_output @ self.down_proj[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states
                * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class Qwen3VLMoETextSparseMoeBlock(MegatronModule):
    """
    Sparse MoE block, adapted from HuggingFace's Qwen3VLMoETextSparseMoeBlock.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.num_experts = self.config.num_moe_experts
        self.top_k = self.config.moe_router_topk
        self.gate = nn.Linear(self.config.hidden_size, self.config.num_moe_experts, bias=False)
        self.experts = Qwen3VLMoETextExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(
            1, router_indices, routing_weights
        )
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out

class Qwen3VLMoETextDecoderLayer(Qwen3VLTextDecoderLayer):
    """
    Qwen3VLMoETextDecoderLayer
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.post_attention_layernorm = Qwen3VLRMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )


class Qwen3VLMoETextModel(Qwen3VLTextModel):
    """
    The bare Qwen3VLMoETextModel outputting raw hidden-states without any specific head on top.
    """

    def __init__(self, config: Qwen3VLConfig):
        super(Qwen3VLTextModel, self).__init__(config)
        self.layers = torch.nn.ModuleList(
            [
                Qwen3VLMoETextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.text_config.num_hidden_layers)
            ]
        )
