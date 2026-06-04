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

"""
Gemma 4 Dense layer specs, Dense provider, and Vision-Language model.

Dense (E4B) layer specification:
- 4-norm transformer structure (input, post-attn, pre-MLP, post-MLP)
- Dual RoPE (sliding θ=10000, global θ=1000000 with partial rotation)
- Per-Layer Embeddings (PLE)
- Shared KV cache (last N layers)

Vision-Language model (Gemma4VLModel):
- HuggingFace Gemma4 vision tower + multimodal embedder
- Megatron-Core GPT language model (Dense or MoE)
"""

import copy
import types
import weakref
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    LayerNormBuilder,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.utils import is_layer_window_attention
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params, get_pg_rank
from torch import Tensor
from transformers import AutoModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


# ---------------------------------------------------------------------------
# Gemma-4 Dense layer specs
# ---------------------------------------------------------------------------


class Gemma4RMSNorm(nn.Module):
    """HF Gemma4-compatible RMSNorm.

    Gemma4 uses ``torch.pow(mean_squared, -0.5)`` rather than ``rsqrt``. The
    forward values are very close, but using the same expression keeps parity
    tests stable for block/model gradients.

    Args:
        with_scale: If False, no learnable weight is created (matches HF's
                    ``with_scale=False`` used e.g. in the MoE router norm).
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
        with_scale: bool = True,
    ):
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        normed_output = hidden_states.float() * torch.pow(
            hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps,
            -0.5,
        )
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


RMSNorm = Gemma4RMSNorm


# ---------------------------------------------------------------------------
# Dense local MoE router/experts (local non-TE impl, Step 5 of Dense spec)
# ---------------------------------------------------------------------------


class Gemma4MoERouter(nn.Module):
    """Token router for Gemma-4 Dense MoE block.

    Mirrors HF ``Gemma4TextRouter``:
      - Scaleless RMSNorm → multiply by learnable per-dim scale × 1/√hidden_size
      - Linear projection → softmax → top-k selection
      - Normalize top-k weights; apply per-expert learned scale
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_experts = getattr(config, 'num_experts', 1)
        eps = getattr(config, 'layernorm_epsilon', 1e-6)
        top_k = getattr(config, 'top_k_experts', 1)

        self.hidden_size = hidden_size
        self.scalar_root_size = hidden_size ** -0.5
        self.top_k = top_k

        self.norm = Gemma4RMSNorm(config, hidden_size, eps=eps, with_scale=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        expert_scores = self.proj(h)
        router_probs = F.softmax(expert_scores.float(), dim=-1).to(h.dtype)
        top_k_weights, top_k_index = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probs, top_k_weights, top_k_index


class Gemma4MoEExperts(nn.Module):
    """Sparse expert collection for Gemma-4 Dense MoE block.

    Mirrors HF ``Gemma4TextExperts``.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        num_experts = getattr(config, 'num_experts', 1)
        hidden_size = config.hidden_size
        moe_intermediate_size = getattr(config, 'moe_intermediate_size', hidden_size)

        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [E, K, tokens]
            expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

        for idx in expert_hit:
            e = idx[0]
            if e >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            cur = hidden_states[token_idx]
            gate, up = F.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
            cur_out = F.gelu(gate, approximate='tanh') * up
            cur_out = F.linear(cur_out, self.down_proj[e])
            cur_out = cur_out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur_out.to(final.dtype))
        return final


# ---------------------------------------------------------------------------
# Dense TransformerLayer submodules dataclass
# ---------------------------------------------------------------------------


@dataclass
class Gemma4DenseTransformerLayerSubmodules(TransformerLayerSubmodules):
    """TransformerLayerSubmodules extended with Gemma-4 Dense post-sublayer norms."""

    post_self_attn_layernorm: LayerNormBuilder = IdentityOp
    post_mlp_layernorm: LayerNormBuilder = IdentityOp
    post_per_layer_input_norm: LayerNormBuilder = IdentityOp


def _is_gemma4_sliding_layer(config: TransformerConfig, layer_number: int) -> bool:
    """Return whether a Gemma4 layer uses sliding attention."""
    if not getattr(config, "window_size", None):
        return False

    skip_freq = getattr(config, "window_attn_skip_freq", None)
    if isinstance(skip_freq, list):
        layer_type = skip_freq[layer_number - 1]
        if isinstance(layer_type, str):
            return layer_type == "sliding_attention"
        return bool(layer_type)

    return is_layer_window_attention(config.window_size, skip_freq, layer_number)


# ---------------------------------------------------------------------------
# Gemma4DenseSelfAttention: v_norm + shared KV + k_eq_v
# ---------------------------------------------------------------------------


class Gemma4DenseSelfAttention(SelfAttention):
    """SelfAttention subclass for Gemma-4 Dense.

    Extends SelfAttention with:
    - v_norm: scaleless RMSNorm on value states
    - attention_k_eq_v: full-attention layers reuse K projection for V
    - Shared KV cache: last N layers reuse K/V from an earlier layer
    """

    def __init__(self, config: TransformerConfig, submodules, layer_number: int, *args, **kwargs):
        attention_config = copy.copy(config)
        attention_config.softmax_scale = 1.0 if config.softmax_scale is None else config.softmax_scale
        attention_config.qk_layernorm = True

        is_sliding = _is_gemma4_sliding_layer(config, layer_number)
        if not is_sliding:
            if getattr(config, 'global_kv_channels', None) is not None:
                attention_config.kv_channels = config.global_kv_channels
            if getattr(config, 'num_global_query_groups', None) is not None:
                attention_config.num_query_groups = config.num_global_query_groups

        super().__init__(attention_config, submodules, layer_number, *args, **kwargs)
        self.original_config = config
        self.is_gemma4_sliding_layer = is_sliding

        self.attention_k_eq_v = (
            getattr(config, 'attention_k_eq_v', False) and not is_sliding
        )

        layer_idx = layer_number - 1
        num_layers = getattr(config, 'num_layers', 0)
        num_kv_shared = getattr(config, 'num_kv_shared_layers', 0)
        first_kv_shared_idx = num_layers - num_kv_shared

        self.is_kv_shared_layer = (num_kv_shared > 0) and (layer_idx >= first_kv_shared_idx)
        self.store_full_length_kv = False
        self.kv_shared_layer_index: Optional[int] = None

        if num_kv_shared > 0:
            skip_freq = getattr(config, 'window_attn_skip_freq', None)
            if isinstance(skip_freq, list):
                layer_is_sliding = [
                    x == "sliding_attention" if isinstance(x, str) else bool(x)
                    for x in skip_freq[:num_layers]
                ]
            elif isinstance(skip_freq, int) and skip_freq > 0:
                layer_is_sliding = [(i + 1) % skip_freq != 0 for i in range(num_layers)]
            else:
                layer_is_sliding = [False] * num_layers

            if self.is_kv_shared_layer:
                prev_types = layer_is_sliding[:first_kv_shared_idx]
                for i in range(len(prev_types) - 1, -1, -1):
                    if prev_types[i] == is_sliding:
                        self.kv_shared_layer_index = i
                        break
            else:
                is_last_of_type = layer_idx < first_kv_shared_idx
                for i in range(layer_idx + 1, first_kv_shared_idx):
                    if layer_is_sliding[i] == is_sliding:
                        is_last_of_type = False
                        break
                self.store_full_length_kv = is_last_of_type

        self._stored_kv: Optional[Tuple[Tensor, Tensor]] = None
        self._kv_source_ref: Optional[weakref.ReferenceType["Gemma4DenseSelfAttention"]] = None

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        """Separate sliding and global layers in the checkpoint."""
        import dataclasses as _dataclasses

        from megatron.core.dist_checkpointing.mapping import ShardedObject as _ShardedObject
        from megatron.core.dist_checkpointing.mapping import ShardedTensor as _ShardedTensor

        is_sliding = self.is_gemma4_sliding_layer
        suffix = "_sliding" if is_sliding else "_global"
        modified_prefix = prefix[:-1] + suffix + "." if prefix.endswith(".") else prefix + suffix

        state_dict = super().sharded_state_dict(
            prefix=modified_prefix,
            sharded_offsets=sharded_offsets,
            metadata=metadata,
        )

        total_layers = self.config.num_layers
        type_total = sum(
            1 for layer_idx in range(1, total_layers + 1)
            if _is_gemma4_sliding_layer(self.original_config, layer_idx) == is_sliding
        )
        type_rank = sum(
            1 for layer_idx in range(1, self.layer_number)
            if _is_gemma4_sliding_layer(self.original_config, layer_idx) == is_sliding
        )

        def _remap(obj):
            if isinstance(obj, _ShardedTensor):
                if obj.prepend_axis_num <= 0 or obj.global_shape[0] != total_layers:
                    return obj
                new_axis_fragmentations = (
                    (type_total,) + obj.axis_fragmentations[1:]
                    if obj.axis_fragmentations is not None
                    else None
                )
                return _dataclasses.replace(
                    obj,
                    global_shape=(type_total,) + obj.global_shape[1:],
                    global_offset=(type_rank,) + obj.global_offset[1:],
                    axis_fragmentations=new_axis_fragmentations,
                )
            if isinstance(obj, _ShardedObject):
                if not obj.global_shape or obj.global_shape[0] != total_layers:
                    return obj
                return _dataclasses.replace(
                    obj,
                    global_shape=(type_total,) + obj.global_shape[1:],
                    global_offset=(type_rank,) + obj.global_offset[1:],
                )
            return obj

        def _walk(obj):
            if isinstance(obj, dict):
                return {key: _walk(value) for key, value in obj.items()}
            return _remap(obj)

        return _walk(state_dict)

    def _v_norm(self, value: Tensor) -> Tensor:
        vf = value.float()
        return (vf * torch.pow(vf.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)).to(value)

    def _get_k_eq_v_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mixed_qkv, split_arg_list = super().get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            output_gate=False,
            split_qkv=False,
        )
        query, key, _value = torch.split(mixed_qkv, split_arg_list, dim=3)
        raw_key = key

        query = query.reshape(
            query.size(0),
            query.size(1),
            -1,
            self.hidden_size_per_attention_head,
        )

        if self.config.num_query_groups < self.world_size:
            idx = get_pg_rank(self.pg_collection.tp) % (
                self.world_size // self.config.num_query_groups
            )
            size = self.num_attention_heads_per_partition // (
                self.world_size // self.config.num_query_groups
            )
            query = query[:, :, idx * size : (idx + 1) * size, :]

        if self.q_layernorm is not None:
            query = apply_module(self.q_layernorm)(query)
        if self.k_layernorm is not None:
            key = apply_module(self.k_layernorm)(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, raw_key

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states=None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ):
        if self.is_kv_shared_layer:
            if not split_qkv or output_gate:
                return super().get_query_key_value_tensors(
                    hidden_states, key_value_states, output_gate, split_qkv
                )
            query, _k, _v = super().get_query_key_value_tensors(
                hidden_states, key_value_states, False, True
            )
            kv_source = self._kv_source_ref() if self._kv_source_ref is not None else None
            if kv_source is not None and kv_source._stored_kv is not None:
                key, value = kv_source._stored_kv
                key = key.to(query.device)
                value = value.to(query.device)
            else:
                key, value = _k, _v
                value = self._v_norm(value)
            return query, key, value

        if self.attention_k_eq_v and split_qkv and not output_gate:
            query, key, value = self._get_k_eq_v_query_key_value_tensors(
                hidden_states,
                key_value_states,
            )
        else:
            result = super().get_query_key_value_tensors(
                hidden_states, key_value_states, output_gate, split_qkv
            )
            if not split_qkv:
                return result
            if output_gate:
                query, key, value, gate = result
                if self.attention_k_eq_v:
                    value = key
            else:
                query, key, value = result

        value = self._v_norm(value)

        if self.store_full_length_kv:
            self._stored_kv = (key, value)

        if output_gate:
            return query, key, value, gate
        return query, key, value


# ---------------------------------------------------------------------------
# Gemma4DenseTransformerLayer: 4-norm + dual-RoPE + PLE + optional local MoE
# ---------------------------------------------------------------------------


class Gemma4DenseTransformerLayer(TransformerLayer):
    """Transformer layer implementing Gemma-4 Dense 4-norm residual structure.

    Differences from the standard TransformerLayer:
    * post_self_attn_layernorm: applied to attention output before residual add.
    * post_mlp_layernorm: applied to MLP output before residual add.
    * Dual RoPE: selects sliding or full-attention embedding per layer.
    * PLE: per-layer embedding residual block after attention + MLP.
    * Optional local MoE block (Step 5, enabled by enable_moe_block=True).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Gemma4DenseTransformerLayerSubmodules,
        layer_number: int = 1,
        **kwargs,
    ):
        super().__init__(config, submodules, layer_number=layer_number, **kwargs)

        self.post_self_attn_layernorm = submodules.post_self_attn_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_mlp_layernorm = submodules.post_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        _ple_dim = getattr(config, 'per_layer_embed_dim', 0)
        self.register_buffer('layer_scalar', torch.ones(1), persistent=True)
        if _ple_dim > 0:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, _ple_dim, bias=False)
            self.per_layer_projection = nn.Linear(_ple_dim, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = submodules.post_per_layer_input_norm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        _enable_moe = getattr(config, 'enable_moe_block', False)
        if _enable_moe:
            self.moe_router = Gemma4MoERouter(config)
            self.moe_experts = Gemma4MoEExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                config, config.hidden_size, eps=config.layernorm_epsilon
            )
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                config, config.hidden_size, eps=config.layernorm_epsilon
            )
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
                config, config.hidden_size, eps=config.layernorm_epsilon
            )
        else:
            self.moe_router = None
            self.moe_experts = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

    def forward(self, *args, **kwargs):
        per_layer_input = kwargs.pop('per_layer_input', None)

        hidden_states, context = self._forward_attention(*args, **kwargs)
        hidden_states = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            residual = hidden_states
            h = F.gelu(self.per_layer_input_gate(hidden_states), approximate='tanh')
            h = h * per_layer_input
            h = self.per_layer_projection(h)
            h = self.post_per_layer_input_norm(h)
            hidden_states = residual + h

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, context

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb=None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin=None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
        inference_params=None,
        **kwargs,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)

        if isinstance(rotary_pos_emb, tuple) and len(rotary_pos_emb) == 2:
            if _is_gemma4_sliding_layer(self.config, self.layer_number):
                rotary_pos_emb = rotary_pos_emb[0]
            else:
                rotary_pos_emb = rotary_pos_emb[1]

        input_layernorm_output = self.input_layernorm(hidden_states)
        if isinstance(input_layernorm_output, tuple):
            input_layernorm_output, residual = input_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()

        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        if isinstance(attention_output_with_bias, tuple):
            attn_out, attn_bias = attention_output_with_bias[0], attention_output_with_bias[1]
            attn_out = self.post_self_attn_layernorm(attn_out)
            attention_output_with_bias = (attn_out, attn_bias)
        else:
            attention_output_with_bias = self.post_self_attn_layernorm(attention_output_with_bias)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, None

    def _forward_mlp(
        self,
        hidden_states: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)
        if isinstance(pre_mlp_layernorm_output, tuple):
            pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()

        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)

        if self.moe_router is not None:
            mlp_out = (
                mlp_output_with_bias[0]
                if isinstance(mlp_output_with_bias, tuple)
                else mlp_output_with_bias
            )
            dense_out = self.post_feedforward_layernorm_1(mlp_out)

            orig_shape = residual.shape
            hidden_flat = residual.reshape(-1, orig_shape[-1])
            _, top_k_weights, top_k_index = self.moe_router(hidden_flat)
            expert_in = self.pre_feedforward_layernorm_2(hidden_flat)
            expert_out = self.moe_experts(expert_in, top_k_index, top_k_weights)
            expert_out = expert_out.reshape(orig_shape)
            expert_out = self.post_feedforward_layernorm_2(expert_out)

            combined = dense_out + expert_out
            if isinstance(mlp_output_with_bias, tuple):
                mlp_output_with_bias = (combined, mlp_output_with_bias[1])
            else:
                mlp_output_with_bias = combined

        if isinstance(mlp_output_with_bias, tuple):
            mlp_out, mlp_bias = mlp_output_with_bias[0], mlp_output_with_bias[1]
            mlp_out = self.post_mlp_layernorm(mlp_out)
            mlp_output_with_bias = (mlp_out, mlp_bias)
        else:
            mlp_output_with_bias = self.post_mlp_layernorm(mlp_output_with_bias)

        with self.bias_dropout_add_exec_handler():
            output = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        return output


# ---------------------------------------------------------------------------
# Shared-KV wiring
# ---------------------------------------------------------------------------


def wire_gemma4_kv_sharing(model: nn.Module) -> None:
    """Wire shared-KV source references between Gemma4DenseSelfAttention layers.

    Must be called once after the model is fully constructed.
    """
    attn_by_layer: dict = {}
    for module in model.modules():
        if isinstance(module, Gemma4DenseSelfAttention):
            idx = module.layer_number - 1
            attn_by_layer[idx] = module

    for attn in attn_by_layer.values():
        if attn.is_kv_shared_layer and attn.kv_shared_layer_index is not None:
            source = attn_by_layer.get(attn.kv_shared_layer_index)
            if source is not None:
                attn._kv_source_ref = weakref.ref(source)


# ---------------------------------------------------------------------------
# Dense layer spec factory
# ---------------------------------------------------------------------------


def get_gemma4_layer_spec(config: Optional[TransformerConfig] = None) -> ModuleSpec:
    """Return a ModuleSpec for a Gemma-4 Dense transformer layer (local/non-TE)."""
    backend = LocalSpecProvider()

    submodules = Gemma4DenseTransformerLayerSubmodules(
        input_layernorm=RMSNorm,
        self_attention=ModuleSpec(
            module=Gemma4DenseSelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=RMSNorm,
                k_layernorm=RMSNorm,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        post_self_attn_layernorm=RMSNorm,
        pre_mlp_layernorm=RMSNorm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
            ),
        ),
        mlp_bda=get_bias_dropout_add,
        post_mlp_layernorm=RMSNorm,
        post_per_layer_input_norm=RMSNorm,
    )

    return ModuleSpec(module=Gemma4DenseTransformerLayer, submodules=submodules)


gemma4_layer_spec = get_gemma4_layer_spec()


# ---------------------------------------------------------------------------
# Gemma-4 Dense Rotary Positional Embeddings
# ---------------------------------------------------------------------------


class _Gemma4ProportionalRotaryEmbedding(RotaryEmbedding):
    """Gemma-4 full-attention RoPE with proportional partial rotation."""

    def __init__(
        self,
        kv_channels: int,
        partial_rotary_factor: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: float = 1000000.0,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        nn.Module.__init__(self)

        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()

        head_dim = kv_channels
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        nope_angles = head_dim // 2 - rope_angles
        rotated = 1.0 / (
            rotary_base
            ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device) / head_dim)
        )
        non_rotated = torch.zeros(nope_angles, dtype=torch.float32, device=device)
        self.inv_freq = torch.cat([rotated, non_rotated], dim=0)
        self.cp_group = (
            cp_group
            if cp_group is not None
            else parallel_state.get_context_parallel_group(check_initialized=False)
        )


class Gemma4DenseRotaryEmbedding(nn.Module):
    """Dual-theta RoPE for Gemma-4 Dense (sliding θ=10000, global θ=1000000 partial)."""

    def __init__(
        self,
        config: TransformerConfig,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        sliding_base = getattr(config, 'sliding_window_rope_base', 10000.0) or 10000.0
        full_base = getattr(config, 'full_attention_rope_base', 1000000.0) or 1000000.0
        partial_factor = getattr(config, 'full_attention_rope_partial_factor', 1.0)
        sliding_kv_channels = config.kv_channels
        full_kv_channels = getattr(config, 'global_kv_channels', None) or config.kv_channels

        shared = dict(
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            use_cpu_initialization=use_cpu_initialization,
            cp_group=cp_group,
        )
        self.rope_sliding = RotaryEmbedding(
            kv_channels=sliding_kv_channels,
            rotary_percent=rotary_percent,
            rotary_base=sliding_base,
            **shared,
        )
        self.rope_full = _Gemma4ProportionalRotaryEmbedding(
            kv_channels=full_kv_channels,
            partial_rotary_factor=partial_factor,
            rotary_base=full_base,
            **shared,
        )

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """Return ``(emb_sliding, emb_full)``."""
        emb_sliding = self.rope_sliding(
            max_seq_len, offset=offset, packed_seq=packed_seq, cp_group=cp_group
        )
        emb_full = self.rope_full(
            max_seq_len, offset=offset, packed_seq=packed_seq, cp_group=cp_group
        )
        return (emb_sliding, emb_full)

    def get_rotary_seq_len(self, *args, **kwargs) -> int:
        return self.rope_sliding.get_rotary_seq_len(*args, **kwargs)

    def get_cos_sin(self, max_seq_len: int, offset: int = 0):
        return (
            self.rope_sliding.get_cos_sin(max_seq_len, offset),
            self.rope_full.get_cos_sin(max_seq_len, offset),
        )


# ---------------------------------------------------------------------------
# Gemma-4 Dense Provider
# ---------------------------------------------------------------------------


@dataclass
class Gemma4DenseProvider(GPTModelProvider):
    """Gemma-4 Dense (3.8B) model provider for clean Megatron-Core.

    All Gemma4-specific settings are encoded here as dataclass fields so that
    no Gemma4-specific CLI arguments are required.
    """

    num_layers: int = 42
    hidden_size: int = 2560
    ffn_hidden_size: int = 10240
    num_attention_heads: int = 8
    num_query_groups: int = 2
    kv_channels: int = 256
    seq_length: int = 131072
    vocab_size: int = 262143
    make_vocab_size_divisible_by: int = 128

    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = field(
        default_factory=lambda: partial(F.gelu, approximate="tanh")
    )

    scale_embeddings_by_hidden_size: bool = True
    share_embeddings_and_output_weights: bool = True
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    window_size: Optional[Tuple[int, int]] = (511, 0)
    window_attn_skip_freq: Union[int, List[int]] = 6

    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16
    use_cpu_initialization: bool = False

    global_kv_channels: int = 512
    num_global_query_groups: int = 2
    sliding_window_rope_base: float = 10000.0
    full_attention_rope_base: float = 1000000.0
    full_attention_rope_partial_factor: float = 0.25
    num_kv_shared_layers: int = 18
    per_layer_embed_vocab_size: int = 262144
    per_layer_embed_dim: int = 256

    num_moe_experts: int = 128
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 704

    def finalize(self) -> None:
        super().finalize()
        self._gemma4_dense_finalized = True

    def _ensure_finalized(self) -> None:
        if not getattr(self, "_gemma4_dense_finalized", False):
            self.finalize()

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> "torch.nn.Module":
        if vp_stage is not None or getattr(self, "pipeline_model_parallel_size", 1) != 1:
            raise NotImplementedError("Gemma4DenseProvider currently supports PP=1 only.")

        return self.build(
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
        )

    def build(
        self,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> "torch.nn.Module":
        """Build a Gemma-4 Dense GPTModel and attach Bridge-specific components."""
        from megatron.core.models.gpt import GPTModel

        self._ensure_finalized()
        config = self

        padded_vocab = (
            (self.vocab_size + self.make_vocab_size_divisible_by - 1)
            // self.make_vocab_size_divisible_by
            * self.make_vocab_size_divisible_by
        )

        dual_rope_attrs = {
            "sliding_window_rope_base": self.sliding_window_rope_base,
            "full_attention_rope_base": self.full_attention_rope_base,
            "full_attention_rope_partial_factor": self.full_attention_rope_partial_factor,
        }
        for attr in dual_rope_attrs:
            setattr(config, attr, None)
        try:
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gemma4_layer_spec(config),
                vocab_size=padded_vocab,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=getattr(self, "_pg_collection", None),
            )
        finally:
            for attr, value in dual_rope_attrs.items():
                setattr(config, attr, value)

        model.rotary_pos_emb = Gemma4DenseRotaryEmbedding(config)

        if pre_process:
            _attach_ple_modules(model, config, self)
        wire_gemma4_kv_sharing(model)
        _install_ple_forward(model)

        return model


def _attach_ple_modules(
    model: "torch.nn.Module",
    config: "TransformerConfig",
    provider: Gemma4DenseProvider,
) -> None:
    """Add PLE embedding / projection / norm modules to a GPTModel instance."""
    import megatron.core.tensor_parallel as tp

    n_layers = provider.num_layers
    ple_dim = provider.per_layer_embed_dim
    ple_vocab = provider.per_layer_embed_vocab_size
    if ple_dim <= 0 or ple_vocab <= 0:
        return

    model.per_layer_embedding = tp.VocabParallelEmbedding(
        ple_vocab,
        n_layers * ple_dim,
        config=config,
        init_method=config.init_method,
    )
    model.per_layer_model_proj = tp.ColumnParallelLinear(
        provider.hidden_size,
        n_layers * ple_dim,
        config=config,
        init_method=config.init_method,
        bias=False,
        gather_output=True,
    )
    model.per_layer_proj_norm = Gemma4RMSNorm(
        config, ple_dim, eps=provider.layernorm_epsilon
    )


def _compute_per_layer_inputs(
    model: "torch.nn.Module",
    input_ids: "torch.Tensor",
    decoder_input: "torch.Tensor",
) -> "Optional[torch.Tensor]":
    """Compute per_layer_inputs of shape [b, s_local, num_layers, ple_dim], or None."""
    if not hasattr(model, "per_layer_embedding") or model.per_layer_embedding is None:
        return None
    if input_ids is None or decoder_input is None:
        return None

    ple_dim: int = model.config.per_layer_embed_dim
    n_layers: int = model.config.num_layers
    b: int = input_ids.shape[0]

    tok_emb = model.per_layer_embedding(input_ids) * (ple_dim ** 0.5)

    if getattr(model.config, "sequence_parallel", False):
        from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region as _scatter
        tok_emb = _scatter(tok_emb.transpose(0, 1)).transpose(0, 1)

    s_local: int = tok_emb.shape[1]
    tok_emb = tok_emb.view(b, s_local, n_layers, ple_dim)

    mdl_proj, _ = model.per_layer_model_proj(decoder_input.transpose(0, 1))
    mdl_proj = mdl_proj * (model.config.hidden_size ** -0.5)
    mdl_proj = mdl_proj.view(b, s_local, n_layers, ple_dim)
    mdl_proj = model.per_layer_proj_norm(mdl_proj)

    return (mdl_proj + tok_emb) * (2.0 ** -0.5)


def _install_ple_forward(model: "torch.nn.Module") -> None:
    """Patch model.forward() to compute PLE and inject as per_layer_inputs."""
    _orig_class_forward = type(model).forward

    def _ple_forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        **kwargs,
    ):
        if decoder_input is None and getattr(self, "pre_process", True):
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
            if getattr(self.config, "scale_embeddings_by_hidden_size", False):
                decoder_input = decoder_input * (self.config.hidden_size ** 0.5)

        per_layer_inputs = _compute_per_layer_inputs(self, input_ids, decoder_input)
        if per_layer_inputs is not None:
            extra_block_kwargs = {
                **(extra_block_kwargs or {}),
                "per_layer_inputs": per_layer_inputs,
            }

        return _orig_class_forward(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            **kwargs,
        )

    model.forward = types.MethodType(_ple_forward, model)


# ---------------------------------------------------------------------------
# Gemma 4 Vision-Language model
# ---------------------------------------------------------------------------


class Gemma4VLModel(MegatronModule):
    """Gemma 4 Vision-Language-Audio model.

    Wraps HF vision/audio towers + multimodal projectors with a Megatron-Core
    GPT language model (Dense or MoE).

    Forward flow:
        1. Embed text tokens via language model embedding
        2. If pixel_values: vision_tower → embed_vision → scatter at image_token_id positions
        3. If input_features: audio_tower → embed_audio → scatter at audio_token_id positions
        4. Forward through language model decoder
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        if pre_process:
            # Vision encoder
            self.vision_tower = AutoModel.from_config(config.vision_config)
            self._init_embed_vision(config)
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)

            # Audio encoder (optional — only when audio_config is provided)
            if getattr(config, "audio_config", None) is not None:
                self.audio_tower = AutoModel.from_config(config.audio_config)
                self._init_embed_audio(config)
                hook_hf_module_setattr_for_tp_grad_sync(self.audio_tower)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def _init_embed_vision(self, config):
        """Initialize the multimodal embedder (vision → language projection)."""
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
        except (ImportError, AttributeError):
            vision_hidden = config.vision_config.hidden_size
            text_hidden = config.text_config.hidden_size
            eps = config.vision_config.rms_norm_eps

            class _SimpleVisionEmbedder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)
                    self._eps = eps

                def forward(self, x):
                    rms = x.float().pow(2).mean(-1, keepdim=True).add(self._eps).sqrt()
                    x = (x.float() / rms).to(x.dtype)
                    return self.embedding_projection(x)

            self.embed_vision = _SimpleVisionEmbedder()

    def _init_embed_audio(self, config):
        """Initialize the audio projector (audio encoder output → language space).

        Gemma4's embed_audio mirrors embed_vision: parameter-free RMSNorm followed
        by a linear projection from audio_config.output_proj_dims to text hidden_size.
        """
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioEmbedder

            self.embed_audio = Gemma4AudioEmbedder(config.audio_config, config.text_config)
        except (ImportError, AttributeError):
            audio_proj_dim = config.audio_config.output_proj_dims
            text_hidden = config.text_config.hidden_size
            eps = getattr(config.audio_config, "rms_norm_eps", 1e-6)

            class _SimpleAudioEmbedder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding_projection = nn.Linear(audio_proj_dim, text_hidden, bias=False)
                    self._eps = eps

                def forward(self, x):
                    rms = x.float().pow(2).mean(-1, keepdim=True).add(self._eps).sqrt()
                    x = (x.float() / rms).to(x.dtype)
                    return self.embedding_projection(x)

            self.embed_audio = _SimpleAudioEmbedder()

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        """Extract and project image features using HF vision tower + embedder."""
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            **kwargs,
        )
        return self.embed_vision(vision_outputs.last_hidden_state)

    def get_audio_features(self, input_features, **kwargs):
        """Extract and project audio features using HF audio tower + embedder."""
        audio_outputs = self.audio_tower(input_features=input_features, **kwargs)
        return self.embed_audio(audio_outputs.last_hidden_state)

    def _scatter_modality_features(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        features: torch.Tensor,
        token_id: int,
        modality_name: str,
    ) -> torch.Tensor:
        """Scatter projected modality features into the embedding at special token positions."""
        mask = (input_ids == token_id).unsqueeze(-1)
        mask = mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        n_slots = mask[:, :, 0].sum().item()
        n_feats = features.numel() // inputs_embeds.shape[-1]
        if n_slots != n_feats:
            raise ValueError(
                f"{modality_name} token count mismatch: "
                f"{n_slots} {modality_name}_token_id positions vs "
                f"{n_feats} tokens from {modality_name} encoder."
            )
        return inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.device, inputs_embeds.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass combining HF vision/audio encoders with Megatron language model."""
        lm_input_ids = input_ids
        if self.pre_process:
            if input_ids is not None:
                multimodal_mask = input_ids == self.config.image_token_id
                if hasattr(self.config, "audio_token_id"):
                    multimodal_mask = torch.logical_or(
                        multimodal_mask,
                        input_ids == self.config.audio_token_id,
                    )
                if multimodal_mask.any():
                    lm_input_ids = input_ids.clone()
                    lm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id

            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=lm_input_ids, position_ids=None
                )
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [B, S, H]
                if getattr(self.language_model.config, "scale_embeddings_by_hidden_size", False):
                    inputs_embeds = inputs_embeds * (self.language_model.config.hidden_size ** 0.5)

            # Vision: scatter image features at image_token_id positions
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values, image_position_ids=image_position_ids)
                inputs_embeds = self._scatter_modality_features(
                    inputs_embeds, input_ids, image_features,
                    self.config.image_token_id, "image",
                )

            # Audio: scatter audio features at audio_token_id positions
            if input_features is not None and hasattr(self, "audio_tower"):
                audio_features = self.get_audio_features(input_features)
                inputs_embeds = self._scatter_modality_features(
                    inputs_embeds, input_ids, audio_features,
                    self.config.audio_token_id, "audio",
                )

            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [S, B, H]

        attention_mask = self._compute_attention_mask(input_ids)

        pg_coll = getattr(self.config, "_pg_collection", None)
        if pg_coll is not None:
            inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                pg_collection=pg_coll,
            )

        if self.config.sequence_parallel and inputs_embeds is not None:
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        outputs = self.language_model.forward(
            input_ids=lm_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
        return (outputs, loss_mask)

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_audio_model: bool = False,
        freeze_audio_projection: bool = False,
    ):
        """Freeze model modules for fine-tuning."""
        pairs = [
            (freeze_language_model, "language_model"),
            (freeze_vision_model, "vision_tower"),
            (freeze_vision_projection, "embed_vision"),
            (freeze_audio_model, "audio_tower"),
            (freeze_audio_projection, "embed_audio"),
        ]
        for should_freeze, attr in pairs:
            if should_freeze and hasattr(self, attr):
                for param in getattr(self, attr).parameters():
                    param.requires_grad = False

    def _compute_attention_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute attention mask: causal, with bidirectional image groups."""
        if not self.pre_process:
            return None
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(
            torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=input_ids.device)
        )

        def _bidirectional_block_mask(token_mask: torch.Tensor) -> torch.Tensor:
            padded = F.pad(token_mask, (1, 0), value=0)
            boundary = padded[:, 1:] > padded[:, :-1]
            block_ids = token_mask * torch.cumsum(boundary, dim=-1)
            return torch.logical_and(
                block_ids[:, None, :] == block_ids.unsqueeze(-1),
                block_ids.unsqueeze(-1) > 0,
            )

        bidir = _bidirectional_block_mask(input_ids == self.config.image_token_id)

        return ~torch.logical_or(causal_mask, bidir.unsqueeze(1))
