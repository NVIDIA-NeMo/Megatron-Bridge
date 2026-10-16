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

"""Missing Muse Spark Open operations composed with existing MCore building blocks."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.utils import sharded_state_dict_default
from torch import Tensor, nn

from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import MuseSelfAttention


class InputScaledRMSNorm(nn.Module):
    """Apply learned input scaling before native, parameter-free RMSNorm.

    TE's affine RMSNorm scales after normalization and cannot express this
    operation. Native RMSNorm supplies the normalization without an extra weight.
    """

    def __init__(self, hidden_size: int, *, eps: float, config: Any, sequence_parallel: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=config.params_dtype))
        self.weight.sequence_parallel = sequence_parallel
        self.norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize scaled activations using the existing framework kernel."""
        return self.norm(hidden_states * (1 + self.weight))


class MuseSparkResidualGate(nn.Module):
    """Post-normalize and mix a branch with its residual via learned gates.

    Implements the callable interface expected by TransformerLayer's BDA slot,
    keeping the standard layer's recompute, offload and checkpoint machinery.
    """

    def __init__(self, config: Any, *, input_scaled_norm: bool) -> None:
        super().__init__()
        self.temperature = config.residual_gate_temperature
        self.gate = nn.Parameter(torch.zeros(config.hidden_size, dtype=torch.float32))
        mark_keep_in_fp32(self.gate)
        self.gate.sequence_parallel = config.sequence_parallel
        if config.perform_initialization:
            nn.init.normal_(self.gate, std=config.init_method_std)
        if input_scaled_norm:
            self.post_norm = InputScaledRMSNorm(
                config.hidden_size,
                eps=config.post_norm_epsilon,
                config=config,
                sequence_parallel=config.sequence_parallel,
            )
        else:
            self.post_norm = nn.RMSNorm(config.hidden_size, eps=config.post_norm_epsilon, elementwise_affine=False)

    def forward(self, training: bool, fused: bool) -> Callable:
        """Return the residual operation for MCore's BDA call convention."""
        return self._apply_branch

    def _apply_branch(
        self, output_with_bias: tuple[Tensor, Tensor | None], residual: Tensor, dropout: float
    ) -> Tensor:
        if dropout:
            raise ValueError("Muse Spark Open residual mixing requires hidden_dropout=0.")
        output, bias = output_with_bias
        if bias is not None:
            output = output + bias
        output = self.post_norm(output)
        gate = self.gate.float() / self.temperature
        beta = torch.sigmoid(gate)
        alpha = (torch.sigmoid(-gate) * (1 + beta)).clamp_min(1e-3).sqrt()
        return (alpha * residual + beta * output).to(residual.dtype)


class MuseSparkSelfAttention(MuseSelfAttention):
    """Reuse GQA and gating, inserting the missing per-head output norm."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        # The shared Muse base supplies parameter-free Q/K norms.
        self.output_norm = nn.RMSNorm(config.kv_channels, eps=config.layernorm_epsilon, elementwise_affine=False)

    def _apply_output_gate(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        shape = hidden_states.shape
        hidden_states = self.output_norm(hidden_states.reshape(*shape[:-1], -1, self.config.kv_channels))
        return super()._apply_output_gate(hidden_states.reshape(shape), gate)


class MuseSparkRouter(TopKRouter):
    """Reuse sigmoid top-k routing with the model's pre-sigmoid logit scale."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.weight.data = self.weight.data.float()
        mark_keep_in_fp32(self.weight)

    def _apply(self, fn: Callable, recurse: bool = True) -> MuseSparkRouter:
        """Preserve the FP32 correction buffer across device and model-dtype moves."""
        bias = self.expert_bias
        result = super()._apply(fn, recurse=recurse)
        if bias is not None:
            # Module._apply replaces buffers, so a tensor attribute alone does
            # not survive the earlier device move. Retain the original values.
            self.expert_bias = mark_keep_in_fp32(bias.to(device=self.expert_bias.device, dtype=torch.float32))
        return result

    def gating(self, input: Tensor) -> Tensor:
        """Scale router logits before MCore performs expert selection."""
        # HF retains FP32 router weights. The native TE gating GEMM requires
        # matching operand dtypes, even when its output dtype is explicit.
        return super().gating(input.to(self.weight.dtype)) * self.config.output_multiplier


class MuseSparkExperts(TEGroupedMLP):
    """Normalize each expert's output before applying its routing weight."""

    def __init__(self, num_local_experts: int, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(num_local_experts, config, *args, **kwargs)
        self.attention_tp_group = kwargs["pg_collection"].tp
        if config.expert_tensor_parallel_size != 1:
            raise ValueError(
                "Muse Spark Open expert output normalization currently requires expert_tensor_parallel_size=1."
            )
        self.post_expert_norm = InputScaledRMSNorm(
            config.moe_latent_size,
            eps=config.post_norm_epsilon,
            config=config,
            sequence_parallel=config.sequence_parallel,
        )

    def forward(
        self, permuted_local_hidden_states: Tensor, tokens_per_expert: Tensor, permuted_probs: Tensor
    ) -> tuple[Tensor, None]:
        """Keep grouped GEMMs while delaying routing weights until after normalization."""
        output, bias = super().forward(
            permuted_local_hidden_states, tokens_per_expert, torch.ones_like(permuted_probs)
        )
        if bias is not None:
            raise ValueError("Muse Spark Open experts require bias-free projections.")
        output = self.post_expert_norm(output)
        return (output * permuted_probs.reshape(-1, 1)).to(output.dtype), None

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None) -> dict:
        """Save the shared norm with ordinary TP replication, independently of ETP."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        # TEGroupedMLP uses its expert TP group for children. This norm is shared
        # across all experts, so its checkpoint replicas follow the attention TP
        # and ordinary DP groups, just like its gradient synchronization.
        state.update(
            sharded_state_dict_default(
                self.post_expert_norm,
                f"{prefix}post_expert_norm.",
                sharded_offsets,
                metadata,
                tp_group=self.attention_tp_group,
            )
        )
        return state


class MuseSparkMoE(MoELayer):
    """Reuse latent projections and dispatch with a pre-expert RMSNorm."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.pre_expert_norm = TENorm(config, config.moe_latent_size, eps=config.layernorm_epsilon)

    def preprocess(
        self, hidden_states: Tensor, probs: Tensor, routing_map: Tensor, padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Normalize the shared latent representation before dispatching tokens.

        As in ``MoELayer.preprocess``, only the flex dispatcher takes the batch-first padding mask.
        """
        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        if isinstance(self.token_dispatcher, MoEFlexTokenDispatcher):
            return self.token_dispatcher.dispatch_preprocess(
                hidden_states, routing_map, probs, padding_mask=padding_mask
            )
        return self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)


def get_muse_spark_hybrid_stack_spec(config: Any) -> ModuleSpec:
    """Compose independent attention and MoE layers for a repeated ``*E`` stack."""
    experts = partial(
        MuseSparkExperts,
        submodules=GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        ),
    )
    moe = partial(MuseSparkMoE, submodules=MoESubmodules(experts=experts, router=MuseSparkRouter))
    return ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=TENorm,
                    self_attention=ModuleSpec(
                        module=MuseSparkSelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TEColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                        ),
                    ),
                    self_attn_bda=ModuleSpec(
                        module=MuseSparkResidualGate, params={"config": config, "input_scaled_norm": False}
                    ),
                ),
            ),
            moe_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=TENorm,
                    mlp=moe,
                    mlp_bda=ModuleSpec(
                        module=MuseSparkResidualGate, params={"config": config, "input_scaled_norm": True}
                    ),
                ),
            ),
        ),
    )
