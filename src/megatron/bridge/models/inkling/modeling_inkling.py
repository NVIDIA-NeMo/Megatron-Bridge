# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native Megatron text layers for Inkling's relative attention and joint MoE."""

from __future__ import annotations

from copy import copy, deepcopy
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    set_tensor_model_parallel_attributes,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


def joint_router_weights(
    logits: Tensor, expert_bias: Tensor, global_scale: Tensor, top_k: int, route_scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize selected routed and all shared experts together in FP32."""
    routed_count = expert_bias.numel()
    logits = logits.float()
    indices = (logits[..., :routed_count].sigmoid() + expert_bias.float()).topk(top_k, sorted=False).indices
    selected = torch.cat((logits[..., :routed_count].gather(-1, indices), logits[..., routed_count:]), dim=-1)
    log_weights = F.logsigmoid(selected)
    weights = (log_weights - log_weights.logsumexp(-1, keepdim=True)).exp() * route_scale * global_scale.float()
    return indices, weights[..., :top_k], weights[..., top_k:]


def relative_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    relative_bias: Tensor,
    *,
    window_size: int | None = None,
    attention_mask: Tensor | None = None,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Causal GQA with a compact [batch, heads, sequence, distance] bias bank.

    Masks use Megatron's convention: True excludes a position. The key-padding
    shape [batch, 1, 1, sequence] and full attention masks are both supported.
    """
    batch, _, length, head_dim = query.shape
    extent = relative_bias.shape[-1]

    def mask_mod(b, h, q, k):
        allowed = q >= k
        if window_size is not None:
            allowed = allowed & (q - k < window_size)
        if padding_mask is not None:
            allowed = allowed & ~padding_mask[b, k.clamp(max=length - 1)]
        if attention_mask is not None:
            mask_q = q.clamp(max=attention_mask.shape[-2] - 1)
            mask_h = h.clamp(max=attention_mask.shape[1] - 1)
            mask_b = b.clamp(max=attention_mask.shape[0] - 1)
            allowed = allowed & ~attention_mask[mask_b, mask_h, mask_q, k.clamp(max=length - 1)]
        return allowed

    def score_mod(score, b, h, q, k):
        distance = q - k
        bias = relative_bias[b, h, q, distance.clamp(0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias, 0.0)

    block_mask = create_block_mask(
        mask_mod, batch, None, length, length, device=str(query.device), _compile=query.is_cuda
    )
    attention = _compiled_flex_attention if query.is_cuda else flex_attention
    options = {"kernel_options": {"BACKEND": "TRITON"}} if query.is_cuda else {}
    return attention(
        query, key, value, score_mod=score_mod, block_mask=block_mask, scale=1.0 / head_dim, enable_gqa=True, **options
    )


class InklingShortConvolution(MegatronModule):
    """FP32 residual depthwise causal convolution, optionally over TP-sharded channels."""

    def __init__(
        self,
        config: TransformerConfig,
        channels: int,
        *,
        tp_group=None,
        channel_parallel: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(config)
        self.tp_group = tp_group
        self.channel_parallel = channel_parallel
        self.sequence_parallel = sequence_parallel
        self.conv1d = nn.Conv1d(
            channels,
            channels,
            config.inkling_conv_kernel_size,
            groups=channels,
            bias=False,
            dtype=torch.float32,
            device=(
                "cpu"
                if config.use_cpu_initialization
                else "meta"
                if config.init_model_with_meta_device
                else torch.cuda.current_device()
            ),
        )
        mark_keep_in_fp32(self.conv1d.weight)
        set_tensor_model_parallel_attributes(self.conv1d.weight, channel_parallel, 0, 1)
        # Replicated convolutions gather the whole sequence and therefore compute
        # identical full-sequence weight gradients on each TP rank.
        self.conv1d.weight.sequence_parallel = False
        if config.perform_initialization:
            config.init_method(self.conv1d.weight)

    def forward(self, hidden_states: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Convolve [sequence, batch, channels], excluding padded convolution inputs."""
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            inputs = hidden_states.float()
            masked = inputs if padding_mask is None else inputs.masked_fill(padding_mask.T.unsqueeze(-1), 0.0)
            conv = F.conv1d(
                masked.permute(1, 2, 0),
                self.conv1d.weight.float(),
                padding=self.conv1d.kernel_size[0] - 1,
                groups=self.conv1d.groups,
            )[..., : hidden_states.shape[0]].permute(2, 0, 1)
            output = (inputs + conv).to(hidden_states.dtype)
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Record channel sharding for K/V convolutions in native checkpoints."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(keep_vars=True),
            prefix,
            {"conv1d.weight": 0} if self.channel_parallel else {},
            sharded_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class InklingRelativeLogits(MegatronModule):
    """Learned distance bank shared by attention heads and replicated across TP."""

    def __init__(self, config: TransformerConfig, extent: int, tp_group) -> None:
        super().__init__(config)
        self.tp_group = tp_group
        self.proj = nn.Parameter(torch.empty(config.inkling_d_rel, extent, dtype=config.params_dtype))
        # Each TP rank sees different query heads, so bank gradients must be summed.
        self.proj.sequence_parallel = True
        if config.perform_initialization:
            config.init_method(self.proj)

    def forward(self, relative_states: Tensor) -> Tensor:
        """Project [sequence, batch, heads, rank] into [batch, heads, sequence, distance]."""
        return (relative_states @ self.proj).permute(1, 2, 0, 3)


class InklingSelfAttention(SelfAttention):
    """Native TP projections and TE RMSNorm with Inkling compact relative attention."""

    def __init__(self, config, submodules, layer_number, **kwargs) -> None:
        local = copy(config)
        self.is_sliding = config.inkling_layer_types[layer_number - 1] == "hybrid_sliding"
        if self.is_sliding:
            local.num_attention_heads = config.inkling_swa_num_attention_heads
            local.num_query_groups = config.inkling_swa_num_query_groups
            local.kv_channels = config.inkling_swa_kv_channels
        super().__init__(local, submodules, layer_number, **kwargs)
        if local.num_query_groups % self.pg_collection.tp.size():
            raise ValueError("Inkling currently requires KV heads divisible by tensor parallel size")
        self.window_size = config.inkling_sliding_window if self.is_sliding else None
        extent = self.window_size if self.is_sliding else config.inkling_rel_extent
        self.rel_logits_proj = InklingRelativeLogits(config, extent, self.pg_collection.tp)
        self.linear_r = TEColumnParallelLinear(
            config.hidden_size,
            local.num_attention_heads * config.inkling_d_rel,
            config=local,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
            pg_collection=self.pg_collection,
        )
        channels = self.num_query_groups_per_partition * self.hidden_size_per_attention_head
        self.k_sconv = InklingShortConvolution(config, channels, tp_group=self.pg_collection.tp, channel_parallel=True)
        self.v_sconv = InklingShortConvolution(config, channels, tp_group=self.pg_collection.tp, channel_parallel=True)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        padding_mask: Tensor | None = None,
        inference_context=None,
        packed_seq_params=None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        """Full-sequence attention; cached decoding and packed sequences are not enabled."""
        if inference_context is not None or packed_seq_params is not None:
            raise ValueError("Inkling attention supports full unpacked sequences without an inference cache")
        mixed, _ = self.linear_qkv(hidden_states)
        length, batch = mixed.shape[:2]
        groups = self.num_query_groups_per_partition
        head_dim = self.hidden_size_per_attention_head
        heads_per_group = self.num_attention_heads_per_partition // groups
        mixed = mixed.view(length, batch, groups, (heads_per_group + 2) * head_dim)
        query, key, value = mixed.split((heads_per_group * head_dim, head_dim, head_dim), dim=-1)
        query = query.reshape(length, batch, -1, head_dim)
        key = self.k_sconv(key.reshape(length, batch, -1), padding_mask).view(length, batch, groups, head_dim)
        value = self.v_sconv(value.reshape(length, batch, -1), padding_mask).view(length, batch, groups, head_dim)
        query = self.q_layernorm(query).permute(1, 2, 0, 3)
        key = self.k_layernorm(key).permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)
        relative, _ = self.linear_r(hidden_states)
        relative = relative.view(length, batch, self.num_attention_heads_per_partition, self.config.inkling_d_rel)
        bias = self.rel_logits_proj(relative)
        if not self.is_sliding and self.config.inkling_log_scaling_n_floor is not None:
            positions = torch.arange(1, length + 1, device=query.device, dtype=torch.float32)
            tau = (
                1.0
                + self.config.inkling_log_scaling_alpha
                * (positions / self.config.inkling_log_scaling_n_floor).clamp(min=1.0).log()
            )
            query = (query.float() * tau.view(1, 1, -1, 1)).to(query.dtype)
            bias = (bias.float() * tau.view(1, 1, -1, 1)).to(bias.dtype)
        output = relative_attention(
            query,
            key,
            value,
            bias,
            window_size=self.window_size,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
        )
        output = output.permute(2, 0, 1, 3).reshape(length, batch, -1).contiguous()
        return self.linear_proj(output)


class InklingRouter(Router):
    """Native MoE router with shared sink experts included in normalization."""

    def __init__(self, config, pg_collection=None, is_mtp_layer=False) -> None:
        router_config = copy(config)
        self.num_routed_experts = config.num_moe_experts
        router_config.num_moe_experts += config.inkling_n_shared_experts
        super().__init__(router_config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        self.register_buffer(
            "expert_bias", mark_keep_in_fp32(torch.zeros(self.num_routed_experts, dtype=torch.float32))
        )
        self.global_scale = mark_keep_in_fp32(nn.Parameter(torch.ones(1, dtype=torch.float32)))
        self.global_scale.sequence_parallel = config.sequence_parallel

    def reset_parameters(self) -> None:
        """Preserve router checkpoint values through native BF16 module conversion."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
        mark_keep_in_fp32(self.weight)
        self.weight.sequence_parallel = self.config.sequence_parallel

    def joint_weights(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return routed indices, routed weights, and shared-expert weights."""
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            logits = F.linear(hidden_states.float(), self.weight.float())
            return joint_router_weights(
                logits,
                self.expert_bias,
                self.global_scale,
                self.config.moe_router_topk,
                self.config.inkling_route_scale,
            )

    def routing(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Produce the native dense probabilities and Boolean expert map."""
        indices, weights, _ = joint_router_weights(
            logits, self.expert_bias, self.global_scale, self.config.moe_router_topk, self.config.inkling_route_scale
        )
        shape = (*logits.shape[:-1], self.num_routed_experts)
        probs = weights.new_zeros(shape).scatter(-1, indices, weights)
        routing_map = torch.zeros(shape, dtype=torch.bool, device=logits.device).scatter(-1, indices, True)
        return probs.reshape(-1, self.num_routed_experts), routing_map.reshape(-1, self.num_routed_experts)

    def forward(self, hidden_states: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Compute routing without auxiliary losses or online selection-bias updates."""
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            probs, routing_map = self.routing(F.linear(hidden_states.float(), self.weight.float()))
        if padding_mask is not None:
            padding = padding_mask.reshape(-1, 1)
            probs = probs.masked_fill(padding, 0.0)
            routing_map = routing_map & ~padding
        return probs, routing_map


class InklingRoutedExperts(TEGroupedMLP):
    """Keep routed expert weighting after the down projection, as in Inkling."""

    def forward(self, hidden_states, tokens_per_expert, permuted_probs, **kwargs):
        output, bias = super().forward(hidden_states, tokens_per_expert, torch.ones_like(permuted_probs), **kwargs)
        return (output.float() * permuted_probs.reshape(-1, 1)).to(output.dtype), bias


class InklingMoELayer(MoELayer):
    """Native EP dispatch with jointly gated, replicated shared experts."""

    def __init__(self, config, submodules, pg_collection=None, **kwargs) -> None:
        super().__init__(config, submodules=submodules, pg_collection=pg_collection, **kwargs)
        shared_config = copy(config)
        shared_config.ffn_hidden_size = config.moe_ffn_hidden_size
        mlp_submodules = MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear)
        self.shared_experts = nn.ModuleList(
            [
                MLP(
                    shared_config,
                    submodules=mlp_submodules,
                    tp_group=self.tp_group,
                    pg_collection=pg_collection,
                )
                for _ in range(config.inkling_n_shared_experts)
            ]
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Preserve each shared MLP's native TP metadata through its ModuleList."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        shared_prefix = f"{prefix}shared_experts."
        state = {key: value for key, value in state.items() if not key.startswith(shared_prefix)}
        for index, expert in enumerate(self.shared_experts):
            state.update(expert.sharded_state_dict(f"{shared_prefix}{index}.", sharded_offsets, metadata))
        return state

    def shared_experts_compute(self, hidden_states: Tensor) -> Tensor:
        """Use native weighted SwiGLU before each shared expert's down projection."""
        _, _, weights = self.router.joint_weights(hidden_states)
        if self.config.sequence_parallel:
            weights = gather_from_sequence_parallel_region(weights, group=self.tp_group)
        # Recomputing the small router projection here avoids storing forward
        # tensors on the module, which would break interleaved/recomputed forwards.
        outputs = [
            expert(hidden_states, per_token_scale=weights[..., i])[0].float()
            for i, expert in enumerate(self.shared_experts)
        ]
        return torch.stack(outputs).sum(0).to(hidden_states.dtype)


class InklingDenseMLP(MLP):
    """Native fused SwiGLU with the published dense-layer output scale."""

    def __init__(self, config, *args, **kwargs) -> None:
        # Keep dense activation intermediates in FP32 until the output cast.
        config = copy(config)
        config.bias_activation_fusion = True
        super().__init__(config, *args, **kwargs)
        self.global_scale = nn.Parameter(torch.ones(1, dtype=config.params_dtype))
        self.global_scale.sequence_parallel = config.sequence_parallel

    def forward(self, hidden_states: Tensor, **kwargs) -> tuple[Tensor, Tensor | None]:
        output, bias = super().forward(hidden_states, **kwargs)
        return output * self.global_scale, bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Include Inkling's replicated scale alongside the native MLP weights."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state.update(
            make_sharded_tensors_for_checkpoint(
                {"global_scale": self.global_scale},
                prefix,
                sharded_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return state


class InklingTransformerLayer(TransformerLayer):
    """Native layer modules with residual short convolutions before both residual adds."""

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.attn_sconv = InklingShortConvolution(
            config, config.hidden_size, tp_group=self.tp_group, sequence_parallel=config.sequence_parallel
        )
        self.mlp_sconv = InklingShortConvolution(
            config, config.hidden_size, tp_group=self.tp_group, sequence_parallel=config.sequence_parallel
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context=None,
        context_mask=None,
        inference_context=None,
        packed_seq_params=None,
        padding_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        """Run an unpacked layer; native TransformerBlock owns full-layer recomputation."""
        if context is not None or inference_context is not None or packed_seq_params is not None:
            raise ValueError("Inkling supports unpacked text training/scoring without cross-attention or a KV cache")
        if padding_mask is not None and self.config.sequence_parallel:
            full_padding = gather_from_sequence_parallel_region(
                padding_mask.T.contiguous(), tensor_parallel_output_grad=False, group=self.tp_group
            ).T
        else:
            full_padding = padding_mask
        if full_padding is None and attention_mask is not None:
            if attention_mask.dtype != torch.bool or attention_mask.ndim != 4:
                raise ValueError("Inkling expects a Boolean Megatron attention mask")
            if attention_mask.shape[-2] == 1:
                full_padding = attention_mask[:, 0, 0, :]
            else:
                # A key hidden from every query is padding; causal masking alone
                # leaves at least its diagonal visible.
                full_padding = attention_mask.all(dim=-2)[:, 0, :]
        attention, bias = self.self_attention(
            self.input_layernorm(hidden_states), attention_mask, padding_mask=full_padding
        )
        if bias is not None:
            attention = attention + bias
        hidden_states = hidden_states + self.attn_sconv(attention, full_padding)
        mlp_output, bias = self.mlp(self.pre_mlp_layernorm(hidden_states), padding_mask=padding_mask)
        if bias is not None:
            mlp_output = mlp_output + bias
        output = hidden_states + self.mlp_sconv(mlp_output, full_padding)
        return output, None


def inkling_layer_spec(config, vp_stage: int | None = None):
    """Select native dense/MoE layers and replace only Inkling's architectural blocks."""
    block = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    for i, original in enumerate(block.layer_specs):
        layer = deepcopy(original)
        layer.module = InklingTransformerLayer
        layer.submodules.input_layernorm = TENorm
        layer.submodules.pre_mlp_layernorm = TENorm
        layer.submodules.sharded_state_dict_keys_map = {}
        attention = layer.submodules.self_attention
        attention.module = InklingSelfAttention
        attention.submodules.linear_qkv = TEColumnParallelLinear
        attention.submodules.q_layernorm = TENorm
        attention.submodules.k_layernorm = TENorm
        mlp = layer.submodules.mlp
        if mlp.func is MoELayer:
            experts = partial(
                InklingRoutedExperts,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
                ),
            )
            layer.submodules.mlp = partial(
                InklingMoELayer, submodules=MoESubmodules(experts=experts, router=InklingRouter)
            )
        else:
            layer.submodules.mlp = partial(
                InklingDenseMLP.as_mlp_submodule,
                submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
            )
        block.layer_specs[i] = layer
    return block


class InklingGPTModel(GPTModel):
    """Megatron GPT with native TE embedding normalization and Inkling output scaling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.pre_process:
            self.embedding_norm = TENorm(
                config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
            )

    def _preprocess(self, input_ids, position_ids, decoder_input=None, **kwargs):
        output = super()._preprocess(input_ids, position_ids, decoder_input=decoder_input, **kwargs)
        if self.pre_process and decoder_input is None:
            output = (self.embedding_norm(output[0]), *output[1:])
        return output

    def _postprocess(self, hidden_states, *args, **kwargs):
        if self.post_process:
            hidden_states = hidden_states / self.config.inkling_logits_mup_width_multiplier
        return super()._postprocess(hidden_states, *args, **kwargs)

    def _scale_logits(self, logits: Tensor) -> Tensor:
        # With TP, retain equal shard shapes for native vocab-parallel loss and
        # exclude padding rows from its normalizer. Gathered scoring can crop.
        unpadded = self.config.inkling_unpadded_vocab_size
        if unpadded is None or unpadded >= self.vocab_size:
            return logits
        if logits.shape[-1] == self.vocab_size:
            return logits[..., :unpadded]
        offset = self.pg_collection.tp.rank() * logits.shape[-1]
        rows = torch.arange(logits.shape[-1], device=logits.device) + offset
        return logits.masked_fill(rows >= unpadded, -torch.inf)
