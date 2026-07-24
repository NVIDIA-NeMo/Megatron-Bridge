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
Gemma 4 Dense and MoE layer specs, attention, positional embeddings, and helpers.

Dense (E2B, E4B, and 31B) layer specification:
- 4-norm transformer structure (input, post-attn, pre-MLP, post-MLP)
- Dual RoPE (sliding θ=10000, global θ=1000000 with partial rotation)
- Per-Layer Embeddings (PLE)
- Shared KV cache (last N layers)

MoE layer specification:
- TE-based transformer layer with per-layer output scaling
- Dual RoPE with separate local/global embeddings
- Heterogeneous sliding/global attention with independent head dims
"""

import copy
import inspect
import types
import weakref
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, te_checkpoint
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
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

from megatron.bridge.models.gemma.gemma3_provider import (
    TERowParallelLinearLayerNorm,
    _is_local_attn_layer,
)
from megatron.bridge.utils.import_utils import safe_import_from


if TYPE_CHECKING:
    from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider


HAVE_TE = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")[1]
TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")
TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")


# ---------------------------------------------------------------------------
# Dense LM Components
# ---------------------------------------------------------------------------


def _gemma4_rms_norm(hidden_states: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """Apply the exact RMSNorm expression used by Hugging Face Gemma 4."""
    normed_output = hidden_states.float() * torch.pow(
        hidden_states.float().pow(2).mean(-1, keepdim=True) + eps,
        -0.5,
    )
    if weight is not None:
        normed_output = normed_output * weight.float()
    return normed_output.type_as(hidden_states)


def _mark_sequence_parallel_parameter(parameter: nn.Parameter, config: TransformerConfig) -> None:
    """Mark replicated parameters whose gradients span sequence-parallel shards."""
    setattr(parameter, "sequence_parallel", bool(getattr(config, "sequence_parallel", False)))


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
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=getattr(config, "params_dtype", torch.float32)))
            _mark_sequence_parallel_parameter(self.weight, config)
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        weight = self.weight if self.with_scale else None
        return _gemma4_rms_norm(hidden_states, weight, self.eps)


RMSNorm = Gemma4RMSNorm


class Gemma4DenseMLP(MLP):
    """Keep both MCore projections on Gemma 4's layer-specific FFN width.

    MCore's FC1 accepts an explicit width, while FC2 reads it from the config.
    Gemma 4 E2B doubles the final shared layers, so a shallow config copy keeps
    both projections consistent without replacing MCore's optimized kernels.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        ffn_hidden_size: int | None = None,
        **kwargs: object,
    ) -> None:
        ffn_hidden_size = config.ffn_hidden_size if ffn_hidden_size is None else ffn_hidden_size
        layer_config = copy.copy(config)
        layer_config.ffn_hidden_size = ffn_hidden_size
        super().__init__(
            config=layer_config,
            submodules=submodules,
            ffn_hidden_size=ffn_hidden_size,
            **kwargs,
        )


def _gemma4_dense_ffn_hidden_size(config: TransformerConfig, layer_number: int) -> int:
    """Return the HF FFN width for a one-indexed dense Gemma 4 layer."""
    base_width = int(config.ffn_hidden_size)
    first_shared_layer = int(config.num_layers) - int(getattr(config, "num_kv_shared_layers", 0))
    is_shared_layer = first_shared_layer > 0 and layer_number - 1 >= first_shared_layer
    if getattr(config, "use_double_wide_mlp", False) and is_shared_layer:
        return 2 * base_width
    return base_width


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
        num_experts = getattr(config, "num_experts", 1)
        eps = getattr(config, "layernorm_epsilon", 1e-6)
        top_k = getattr(config, "top_k_experts", 1)

        self.hidden_size = hidden_size
        self.scalar_root_size = hidden_size**-0.5
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
        num_experts = getattr(config, "num_experts", 1)
        hidden_size = config.hidden_size
        moe_intermediate_size = getattr(config, "moe_intermediate_size", hidden_size)

        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))
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
            cur_out = F.gelu(gate, approximate="tanh") * up
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
            if getattr(config, "global_kv_channels", None) is not None:
                attention_config.kv_channels = config.global_kv_channels
            if getattr(config, "num_global_query_groups", None) is not None:
                attention_config.num_query_groups = config.num_global_query_groups

        super().__init__(attention_config, submodules, layer_number, *args, **kwargs)
        self.original_config = config
        self.is_gemma4_sliding_layer = is_sliding

        self.attention_k_eq_v = getattr(config, "attention_k_eq_v", False) and not is_sliding

        layer_idx = layer_number - 1
        num_layers = getattr(config, "num_layers", 0)
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared_idx = num_layers - num_kv_shared

        self.is_kv_shared_layer = (num_kv_shared > 0) and (layer_idx >= first_kv_shared_idx)
        self.store_full_length_kv = False
        self.kv_shared_layer_index: Optional[int] = None

        if num_kv_shared > 0:
            skip_freq = getattr(config, "window_attn_skip_freq", None)
            if isinstance(skip_freq, list):
                layer_is_sliding = [
                    x == "sliding_attention" if isinstance(x, str) else bool(x) for x in skip_freq[:num_layers]
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
            1
            for layer_idx in range(1, total_layers + 1)
            if _is_gemma4_sliding_layer(self.original_config, layer_idx) == is_sliding
        )
        type_rank = sum(
            1
            for layer_idx in range(1, self.layer_number)
            if _is_gemma4_sliding_layer(self.original_config, layer_idx) == is_sliding
        )

        def _remap(obj):
            if isinstance(obj, _ShardedTensor):
                if obj.prepend_axis_num <= 0 or obj.global_shape[0] != total_layers:
                    return obj
                new_axis_fragmentations = (
                    (type_total,) + obj.axis_fragmentations[1:] if obj.axis_fragmentations is not None else None
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
            idx = get_pg_rank(self.pg_collection.tp) % (self.world_size // self.config.num_query_groups)
            size = self.num_attention_heads_per_partition // (self.world_size // self.config.num_query_groups)
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
                return super().get_query_key_value_tensors(hidden_states, key_value_states, output_gate, split_qkv)
            query, _k, _v = super().get_query_key_value_tensors(hidden_states, key_value_states, False, True)
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
            result = super().get_query_key_value_tensors(hidden_states, key_value_states, output_gate, split_qkv)
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

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, *args, **kwargs):
        if isinstance(attention_mask, dict):
            mask_key = "sliding_attention" if self.is_gemma4_sliding_layer else "full_attention"
            attention_mask = attention_mask[mask_key]
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            *args,
            **kwargs,
        )


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
        **kwargs: object,
    ) -> None:
        submodules = copy.deepcopy(submodules)
        mlp_submodules = get_submodules(submodules.mlp)
        ffn_hidden_size = _gemma4_dense_ffn_hidden_size(config, layer_number)
        submodules.mlp = partial(
            Gemma4DenseMLP.as_mlp_submodule,
            submodules=mlp_submodules,
            ffn_hidden_size=ffn_hidden_size,
        )
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

        _ple_dim = getattr(config, "per_layer_embed_dim", 0)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
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

        _enable_moe = getattr(config, "enable_moe_block", False)
        if _enable_moe:
            self.moe_router = Gemma4MoERouter(config)
            self.moe_experts = Gemma4MoEExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.moe_router = None
            self.moe_experts = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

    def forward(self, *args, **kwargs):
        per_layer_input = kwargs.pop("per_layer_input", None)

        hidden_states, context = self._forward_attention(*args, **kwargs)
        hidden_states = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            residual = hidden_states
            h = F.gelu(self.per_layer_input_gate(hidden_states), approximate="tanh")
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
            mlp_out = mlp_output_with_bias[0] if isinstance(mlp_output_with_bias, tuple) else mlp_output_with_bias
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
        device = "cpu" if use_cpu_initialization else torch.cuda.current_device()

        head_dim = kv_channels
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        nope_angles = head_dim // 2 - rope_angles
        rotated = 1.0 / (
            rotary_base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device) / head_dim)
        )
        non_rotated = torch.zeros(nope_angles, dtype=torch.float32, device=device)
        self.inv_freq = torch.cat([rotated, non_rotated], dim=0)
        self.cp_group = (
            cp_group if cp_group is not None else parallel_state.get_context_parallel_group(check_initialized=False)
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

        sliding_base = getattr(config, "sliding_window_rope_base", 10000.0) or 10000.0
        full_base = getattr(config, "full_attention_rope_base", 1000000.0) or 1000000.0
        partial_factor = getattr(config, "full_attention_rope_partial_factor", 1.0)
        sliding_kv_channels = config.kv_channels
        full_kv_channels = getattr(config, "global_kv_channels", None) or config.kv_channels

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
        emb_sliding = self.rope_sliding(max_seq_len, offset=offset, packed_seq=packed_seq, cp_group=cp_group)
        emb_full = self.rope_full(max_seq_len, offset=offset, packed_seq=packed_seq, cp_group=cp_group)
        return (emb_sliding, emb_full)

    def get_rotary_seq_len(self, *args, **kwargs) -> int:
        return self.rope_sliding.get_rotary_seq_len(*args, **kwargs)

    def get_cos_sin(self, max_seq_len: int, offset: int = 0):
        return (
            self.rope_sliding.get_cos_sin(max_seq_len, offset),
            self.rope_full.get_cos_sin(max_seq_len, offset),
        )


# ---------------------------------------------------------------------------
# Per-Layer Embedding (PLE) helpers
# ---------------------------------------------------------------------------


def _attach_ple_modules(
    model: "torch.nn.Module",
    config: "TransformerConfig",
    provider: "Gemma4DenseProvider",
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
    model.per_layer_proj_norm = Gemma4RMSNorm(config, ple_dim, eps=provider.layernorm_epsilon)


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

    tok_emb = model.per_layer_embedding(input_ids) * (ple_dim**0.5)

    if getattr(model.config, "sequence_parallel", False):
        from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region as _scatter

        tok_emb = _scatter(tok_emb.transpose(0, 1)).transpose(0, 1)

    s_local: int = tok_emb.shape[1]
    tok_emb = tok_emb.view(b, s_local, n_layers, ple_dim)

    mdl_proj, _ = model.per_layer_model_proj(decoder_input.transpose(0, 1))
    mdl_proj = mdl_proj * (model.config.hidden_size**-0.5)
    mdl_proj = mdl_proj.view(b, s_local, n_layers, ple_dim)
    mdl_proj = model.per_layer_proj_norm(mdl_proj)

    return (mdl_proj + tok_emb) * (2.0**-0.5)


def _gemma4_layer_input(
    per_layer_inputs: "Optional[torch.Tensor]",
    layer: "torch.nn.Module",
) -> "Optional[torch.Tensor]":
    if per_layer_inputs is None:
        return None
    global_layer_idx = layer.layer_number - 1
    return per_layer_inputs[:, :, global_layer_idx, :].transpose(0, 1)


def _gemma4_layer_accepts_input_ids(layer: "torch.nn.Module") -> bool:
    forward_attention = getattr(layer, "_forward_attention", None)
    if forward_attention is None:
        return False
    try:
        parameters = inspect.signature(forward_attention).parameters
    except (TypeError, ValueError):
        return False
    return "input_ids" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _gemma4_checkpointed_forward(
    self: "torch.nn.Module",
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: "Optional[Tensor]",
    context_mask: "Optional[Tensor]",
    rotary_pos_emb: Tensor,
    attention_bias: "Optional[Tensor]",
    packed_seq_params: PackedSeqParams,
    use_inner_quantization_context: bool,
    padding_mask: "Optional[Tensor]" = None,
    extract_layer_indices: "Optional[set[int]]" = None,
    layer_offset: int = 0,
    input_ids: "Optional[Tensor]" = None,
    per_layer_inputs: "Optional[Tensor]" = None,
):
    """MCore recompute helper variant that carries Gemma4 PLE through checkpoint args."""
    from contextlib import nullcontext

    from megatron.core import tensor_parallel
    from megatron.core.extensions.transformer_engine import HAVE_TE as _HAVE_TE
    from megatron.core.fp4_utils import get_fp4_context
    from megatron.core.fp8_utils import get_fp8_context

    te_checkpoint = None
    if _HAVE_TE:
        from megatron.core.extensions.transformer_engine import te_checkpoint

    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states = []

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            padding_mask=None,
            per_layer_inputs=None,
        ):
            for index in range(start, end):
                layer = self.layers[index]

                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                    per_layer_input=_gemma4_layer_input(per_layer_inputs, layer),
                )
                if (
                    input_ids is not None
                    and isinstance(layer, TransformerLayer)
                    and _gemma4_layer_accepts_input_ids(layer)
                ):
                    layer_kwargs["input_ids"] = input_ids
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    else:
                        for k in ("context", "context_mask", "attention_bias", "padding_mask", "per_layer_input"):
                            layer_kwargs.pop(k, None)
                        hidden_states = layer(**layer_kwargs)
                        context = None

                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            return hidden_states, context

        return custom_forward

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states, context
        cf = custom(start, end)
        args = (hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask, per_layer_inputs)
        if use_checkpoint:
            if self.config.fp8 or self.config.fp4:
                hidden_states, context = te_checkpoint(
                    cf,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    *args,
                )
            else:
                hidden_states, context = tensor_parallel.checkpoint(
                    cf, self.config.distribute_saved_activations, *args
                )
        else:
            hidden_states, context = cf(*args)

        if self.config.recompute_method == "uniform":
            if (end - 1 + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)
        else:
            if (start + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)

    if self.config.recompute_method == "uniform":
        layer_idx = 0
        while layer_idx < self.num_layers_per_pipeline_rank:
            chunk_end = min(layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank)
            chunk_runner(layer_idx, chunk_end, True)
            layer_idx += self.config.recompute_num_layers
    elif self.config.recompute_method == "block":
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            use_checkpoint = (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            )
            chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
    else:
        raise ValueError("Invalid activation recompute method.")

    if len(extract_layer_indices) > 0:
        return hidden_states, intermediate_hidden_states

    return hidden_states


def _patch_ple_block_threading(decoder: "torch.nn.Module") -> None:
    """Patch one Gemma4 decoder instance to thread PLE inputs through clean MCore.

    Clean Megatron-Core's GPTModel already forwards ``extra_block_kwargs`` to its
    decoder, but TransformerBlock does not know Gemma4's ``per_layer_inputs``.
    This patch is deliberately instance-scoped: it only affects the Gemma4
    decoder created by this provider and leaves the TransformerBlock class
    unchanged.
    """
    if getattr(decoder, "_gemma4_ple_threading_patched", False):
        return

    layers = getattr(decoder, "layers", None)
    if layers is None:
        decoder._gemma4_ple_threading_patched = True
        return

    decoder_ref = weakref.ref(decoder)

    for layer in layers:
        if getattr(layer, "_gemma4_ple_layer_forward_patched", False):
            continue
        orig_layer_forward = layer.forward

        def _layer_forward(self, *args, _orig_forward=orig_layer_forward, **kwargs):
            decoder_obj = decoder_ref()
            if (
                decoder_obj is not None
                and "per_layer_input" not in kwargs
                and getattr(decoder_obj, "_gemma4_current_per_layer_inputs", None) is not None
            ):
                kwargs["per_layer_input"] = _gemma4_layer_input(decoder_obj._gemma4_current_per_layer_inputs, self)
            return _orig_forward(*args, **kwargs)

        layer.forward = types.MethodType(_layer_forward, layer)
        layer._gemma4_ple_layer_forward_patched = True

    orig_decoder_forward = decoder.forward

    def _decoder_forward(self, *args, per_layer_inputs=None, **kwargs):
        from megatron.core.transformer import transformer_block as transformer_block_module

        previous = getattr(self, "_gemma4_current_per_layer_inputs", None)
        had_previous = hasattr(self, "_gemma4_current_per_layer_inputs")
        orig_module_checkpointed_forward = getattr(transformer_block_module, "checkpointed_forward", None)
        orig_instance_checkpointed_forward = getattr(self, "_checkpointed_forward", None)
        had_instance_checkpointed_forward = "_checkpointed_forward" in getattr(self, "__dict__", {})
        patched_module_checkpointed_forward = False
        patched_instance_checkpointed_forward = False
        self._gemma4_current_per_layer_inputs = per_layer_inputs

        def _module_checkpointed_forward_with_ple(block, *cf_args, **cf_kwargs):
            block_per_layer_inputs = getattr(block, "_gemma4_current_per_layer_inputs", None)
            if block is not self or block_per_layer_inputs is None:
                return orig_module_checkpointed_forward(block, *cf_args, **cf_kwargs)
            return _gemma4_checkpointed_forward(
                block,
                *cf_args,
                **cf_kwargs,
                per_layer_inputs=block_per_layer_inputs,
            )

        def _instance_checkpointed_forward_with_ple(block, *cf_args, **cf_kwargs):
            block_per_layer_inputs = getattr(block, "_gemma4_current_per_layer_inputs", None)
            if block_per_layer_inputs is None:
                return orig_instance_checkpointed_forward(*cf_args, **cf_kwargs)
            return _gemma4_checkpointed_forward(
                block,
                *cf_args,
                **cf_kwargs,
                per_layer_inputs=block_per_layer_inputs,
            )

        if per_layer_inputs is not None:
            # TODO: remove this guard when both MCore main and dev call
            # TransformerBlock._checkpointed_forward directly.
            if orig_module_checkpointed_forward is not None:
                transformer_block_module.checkpointed_forward = _module_checkpointed_forward_with_ple
                patched_module_checkpointed_forward = True
            if orig_instance_checkpointed_forward is not None:
                self._checkpointed_forward = types.MethodType(_instance_checkpointed_forward_with_ple, self)
                patched_instance_checkpointed_forward = True
        try:
            return orig_decoder_forward(*args, **kwargs)
        finally:
            if patched_module_checkpointed_forward:
                transformer_block_module.checkpointed_forward = orig_module_checkpointed_forward
            if patched_instance_checkpointed_forward:
                if had_instance_checkpointed_forward:
                    self._checkpointed_forward = orig_instance_checkpointed_forward
                else:
                    delattr(self, "_checkpointed_forward")
            if had_previous:
                self._gemma4_current_per_layer_inputs = previous
            else:
                delattr(self, "_gemma4_current_per_layer_inputs")

    decoder.forward = types.MethodType(_decoder_forward, decoder)

    decoder._gemma4_ple_threading_patched = True


def _install_ple_forward(model: "torch.nn.Module") -> None:
    """Patch model.forward() to compute PLE and inject as per_layer_inputs."""
    _patch_ple_block_threading(model.decoder)
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
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if getattr(self.config, "scale_embeddings_by_hidden_size", False):
                decoder_input = decoder_input * (self.config.hidden_size**0.5)

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
# MoE LM Components
# ---------------------------------------------------------------------------


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma 4 MoE transformer layer with per-layer output scaling and extra post-norms."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        self.register_buffer("layer_scalar", torch.ones(1, dtype=config.params_dtype))
        self.pre_shared_expert_layernorm = Gemma4RMSNorm(
            config,
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_ffn_layernorm = Gemma4RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)

    def _forward_mlp(
        self,
        hidden_states: Tensor,
        inference_context: BaseInferenceContext | None = None,
        padding_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        """Run HF's separate shared-expert, routed-expert, and router inputs.

        ``input_ids`` is accepted for compatibility with Megatron-Core ``dev``'s
        ``TransformerLayer.forward``, which forwards it for hash-based MoE routing;
        Gemma 4's separate-input MoE path does not use it.
        """
        del inference_context, input_ids
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states

        moe_input = residual
        moe_unflatten_mbs = None
        # TODO: remove this guard when MCore dev includes commit e86c262ccd0c and both pins expose
        # TransformerLayer._maybe_unflatten_for_moe.
        if packed_seq_params is not None and hasattr(TransformerLayer, "_maybe_unflatten_for_moe"):
            moe_input, padding_mask, moe_unflatten_mbs = TransformerLayer._maybe_unflatten_for_moe(
                self,
                residual,
                padding_mask,
                packed_seq_params,
            )

        expert_input = _gemma4_rms_norm(
            moe_input,
            self.pre_mlp_layernorm.weight,
            self.config.layernorm_epsilon,
        )
        shared_expert_input = _gemma4_rms_norm(
            moe_input,
            self.pre_shared_expert_layernorm.weight,
            self.config.layernorm_epsilon,
        )
        mlp_output_with_bias = self.mlp.forward_with_separate_inputs(
            expert_input,
            shared_expert_input,
            moe_input,
            padding_mask=padding_mask,
        )

        if moe_unflatten_mbs is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = TransformerLayer._maybe_reflatten_from_moe(
                self,
                mlp_output,
                packed_seq_params,
                moe_unflatten_mbs,
            )
            mlp_output_with_bias = (mlp_output, mlp_bias)

        return self._forward_post_mlp(mlp_output_with_bias, residual)

    def _forward_post_mlp(
        self,
        mlp_output_with_bias: tuple[Tensor, Tensor | None],
        residual: Tensor,
    ) -> Tensor:
        from megatron.core.utils import make_viewless_tensor

        mlp_out = mlp_output_with_bias[0]
        mlp_bias = mlp_output_with_bias[1] if len(mlp_output_with_bias) > 1 else None

        normed = self.post_ffn_layernorm(mlp_out)
        if isinstance(normed, tuple):
            normed = normed[0]

        if mlp_bias is not None:
            normed = normed + mlp_bias
        hidden_states = (residual + normed) * self.layer_scalar

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output


class Gemma4TopKRouter(TopKRouter):
    """Gemma 4 MoE router with per-expert scaling."""

    def __init__(self, config: TransformerConfig, **kwargs: object) -> None:
        super().__init__(config=config, **kwargs)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_moe_experts, dtype=config.params_dtype))
        self.scale = nn.Parameter(torch.ones(config.hidden_size, dtype=config.params_dtype))
        _mark_sequence_parallel_parameter(self.per_expert_scale, config)
        _mark_sequence_parallel_parameter(self.scale, config)
        self.scalar_root_size = config.hidden_size**-0.5

    def gating(self, input: Tensor) -> Tensor:
        """Apply HF's scaleless RMSNorm and learned router scale before projection."""
        input = _gemma4_rms_norm(input, None, self.config.layernorm_epsilon)
        input = input * self.scale * self.scalar_root_size
        return super().gating(input)

    def routing(
        self,
        logits: Tensor,
        padding_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor | None]:
        """Route Gemma 4 tokens with arguments accepted by both MCore refs."""
        del input_ids, kwargs
        routing_probs, routing_map = super().routing(logits, padding_mask=padding_mask)
        if routing_map is not None:
            prob_sums = routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            routing_probs = routing_probs / prob_sums
            routing_probs = routing_probs * self.per_expert_scale.unsqueeze(0)
        return routing_probs, routing_map


class Gemma4MoELayer(MoELayer):
    """Gemma 4 MoE layer with post-routed-expert and post-shared-expert normalization."""

    def __init__(self, config: TransformerConfig, submodules: object, **kwargs: object) -> None:
        super().__init__(config=config, submodules=submodules, **kwargs)
        self.post_moe_layernorm = Gemma4RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.post_shared_expert_layernorm = Gemma4RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)

    def forward_with_separate_inputs(
        self,
        expert_input: Tensor,
        shared_expert_input: Tensor,
        router_input: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the MoE using the three independently normalized HF inputs."""
        if self.shared_expert_overlap:
            raise NotImplementedError("Gemma 4 separate shared-expert normalization requires overlap to be disabled")
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        def custom_forward(
            expert_input: Tensor,
            shared_expert_input: Tensor,
            router_input: Tensor,
        ) -> tuple[Tensor, Tensor | None]:
            shared_expert_output = self.shared_experts_compute(shared_expert_input)
            probs, routing_map = self.route(router_input, padding_mask)
            dispatched_input, probs = self.preprocess(expert_input, probs, routing_map)
            dispatched_input, probs = self.dispatch(dispatched_input, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
            output = self.combine(output)
            output = self.postprocess(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute and self.training:
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    expert_input,
                    shared_expert_input,
                    router_input,
                )
            return tensor_parallel.checkpoint(
                custom_forward,
                False,
                expert_input,
                shared_expert_input,
                router_input,
            )
        return custom_forward(expert_input, shared_expert_input, router_input)

    def postprocess(self, output: Tensor, shared_expert_output: Tensor | None) -> Tensor:
        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)
        output = self.post_moe_layernorm(output)
        if isinstance(output, tuple):
            output = output[0]
        if shared_expert_output is not None:
            normed_shared = self.post_shared_expert_layernorm(shared_expert_output)
            if isinstance(normed_shared, tuple):
                normed_shared = normed_shared[0]
            output = output + normed_shared
        return output


def _logit_softcapping(logits: torch.Tensor, scale: float | None) -> torch.Tensor:
    if not scale:
        return logits
    return scale * torch.tanh(logits / scale)


class Gemma4OutputLayer(torch.nn.Module):
    """Mixin that applies final_logit_softcapping after the output linear layer."""

    def forward(self, *args, **kwargs):
        output, bias = super().forward(*args, **kwargs)
        output = _logit_softcapping(output, self.config.final_logit_softcapping)
        return output, bias


def _install_tied_kv(model: "torch.nn.Module", provider: "Gemma4ModelProvider") -> None:
    """Mark global attention layers that require K=V weight tying."""
    if not getattr(provider, "attention_k_eq_v", False):
        return

    num_global_kv_heads = getattr(provider, "num_global_key_value_heads", None)
    if not num_global_kv_heads:
        return

    pattern = provider.interleaved_attn_pattern
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return

    for layer in decoder.layers:
        if _is_local_attn_layer(layer.layer_number, pattern):
            continue
        attn = getattr(layer, "self_attention", None)
        if attn is None:
            continue
        attn._tied_kv = True


def _gemma4_block_spec(config, use_transformer_engine=True, **kwargs):
    """Build Gemma 4 MoE block spec with patched attention, layer, and MoE modules."""
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_transformer_engine, **kwargs)

    for layer_spec in block_spec.layer_specs:
        layer_spec.module = Gemma4TransformerLayer

        attn_spec = layer_spec.submodules.self_attention
        if isinstance(attn_spec.module, type) and issubclass(attn_spec.module, SelfAttention):
            attn_spec.module = Gemma4SelfAttention
        if hasattr(attn_spec, "submodules") and attn_spec.submodules is not None:
            attn_spec.submodules.core_attention = Gemma4TEDotProductAttention
            if use_transformer_engine:
                attn_spec.submodules.linear_proj = TERowParallelLinearLayerNorm

        mlp_spec = layer_spec.submodules.mlp
        if hasattr(mlp_spec, "module") and isinstance(mlp_spec.module, type) and issubclass(mlp_spec.module, MoELayer):
            mlp_spec.module = Gemma4MoELayer
            if hasattr(mlp_spec, "submodules") and mlp_spec.submodules is not None:
                mlp_spec.submodules.router = Gemma4TopKRouter
        elif isinstance(mlp_spec, partial) and isinstance(mlp_spec.func, type) and issubclass(mlp_spec.func, MoELayer):
            mlp_kwargs = dict(mlp_spec.keywords or {})
            mlp_submodules = copy.deepcopy(mlp_kwargs["submodules"])
            mlp_submodules.router = Gemma4TopKRouter
            mlp_kwargs["submodules"] = mlp_submodules
            layer_spec.submodules.mlp = partial(Gemma4MoELayer, *mlp_spec.args, **mlp_kwargs)

    return block_spec


class Gemma4SelfAttention(SelfAttention):
    """Gemma 4 MoE self attention with heterogeneous sliding/global layers."""

    def __init__(self, config: TransformerConfig, layer_number: int, **kwargs):
        config = copy.deepcopy(config)

        if not _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            config.kv_channels = config.global_head_dim
            if getattr(config, "num_global_key_value_heads", None) is not None:
                config.num_query_groups = config.num_global_key_value_heads

        super().__init__(config=config, layer_number=layer_number, **kwargs)
        self._v_norm_eps = config.layernorm_epsilon

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Override to separate sliding and global layers in the checkpoint."""
        import dataclasses as _dataclasses

        from megatron.core.dist_checkpointing.mapping import ShardedObject as _SO
        from megatron.core.dist_checkpointing.mapping import ShardedTensor as _ST

        is_global = not _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern)
        suffix = "_global" if is_global else "_sliding"
        if prefix.endswith("."):
            storage_prefix = prefix[:-1] + suffix + "."
        else:
            storage_prefix = prefix + suffix

        state_dict = super().sharded_state_dict(prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata)

        def _storage_key(key: str) -> str:
            if key.startswith(prefix):
                return storage_prefix + key[len(prefix) :]
            return key.replace(".self_attention.", f".self_attention{suffix}.", 1)

        pattern = self.config.interleaved_attn_pattern
        total_layers = self.config.num_layers
        if is_global:
            type_total = sum(1 for i in range(1, total_layers + 1) if not _is_local_attn_layer(i, pattern))
            type_rank = sum(1 for i in range(1, self.layer_number) if not _is_local_attn_layer(i, pattern))
        else:
            type_total = sum(1 for i in range(1, total_layers + 1) if _is_local_attn_layer(i, pattern))
            type_rank = sum(1 for i in range(1, self.layer_number) if _is_local_attn_layer(i, pattern))

        def _remap(t):
            if isinstance(t, _ST):
                new_key = _storage_key(t.key)
                if t.prepend_axis_num <= 0 or t.global_shape[0] != total_layers:
                    return _dataclasses.replace(t, key=new_key)
                new_global_shape = (type_total,) + t.global_shape[1:]
                new_global_offset = (type_rank,) + t.global_offset[1:]
                new_frags = (type_total,) + t.axis_fragmentations[1:] if t.axis_fragmentations is not None else None
                return _dataclasses.replace(
                    t,
                    key=new_key,
                    global_shape=new_global_shape,
                    global_offset=new_global_offset,
                    axis_fragmentations=new_frags,
                )
            if isinstance(t, _SO):
                new_key = _storage_key(t.key)
                if not t.global_shape or t.global_shape[0] != total_layers:
                    return _dataclasses.replace(t, key=new_key)
                new_global_shape = (type_total,) + t.global_shape[1:]
                new_global_offset = (type_rank,) + t.global_offset[1:]
                return _dataclasses.replace(
                    t,
                    key=new_key,
                    global_shape=new_global_shape,
                    global_offset=new_global_offset,
                )
            return t

        def _fix(d):
            if isinstance(d, dict):
                return {k: _fix(v) for k, v in d.items()}
            return _remap(d)

        return _fix(state_dict)

    def _get_tied_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return independently normalized K and V from Gemma 4's shared raw projection."""
        mixed_qkv, split_arg_list = super().get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            output_gate=False,
            split_qkv=False,
        )
        query, raw_key, _value = torch.split(mixed_qkv, split_arg_list, dim=3)
        key = raw_key

        query = query.reshape(
            query.size(0),
            query.size(1),
            -1,
            self.hidden_size_per_attention_head,
        )
        if self.config.num_query_groups < self.world_size:
            idx = get_pg_rank(self.pg_collection.tp) % (self.world_size // self.config.num_query_groups)
            size = self.num_attention_heads_per_partition // (self.world_size // self.config.num_query_groups)
            query = query[:, :, idx * size : (idx + 1) * size, :]

        if self.q_layernorm is not None:
            query = apply_module(self.q_layernorm)(query)
        if self.k_layernorm is not None:
            key = apply_module(self.k_layernorm)(key)
        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, raw_key

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        """Override to apply v_norm and enforce K=V tying for global attention."""
        if getattr(self, "_tied_kv", False) and kwargs.get("split_qkv", True) and not kwargs.get("output_gate", False):
            query, key, value = self._get_tied_query_key_value_tensors(hidden_states, key_value_states)
            v_float = value.float()
            rms = v_float.pow(2).mean(-1, keepdim=True).add(self._v_norm_eps).sqrt()
            value = (v_float / rms).to(value.dtype)
            return query, key, value

        result = super().get_query_key_value_tensors(hidden_states, key_value_states, **kwargs)
        if len(result) < 3:
            return result
        query, key, value = result[0], result[1], result[2]
        v_float = value.float()
        rms = v_float.pow(2).mean(-1, keepdim=True).add(self._v_norm_eps).sqrt()
        value = (v_float / rms).to(value.dtype)
        return (query, key, value) + result[3:]

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tuple[Tensor, Tensor]] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert isinstance(rotary_pos_emb, (tuple, list)) and len(rotary_pos_emb) == 2
        assert rotary_pos_cos is None and rotary_pos_sin is None

        is_local = _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern)
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask["sliding_attention" if is_local else "full_attention"]

        if is_local:
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


class Gemma4TEDotProductAttention(TEDotProductAttention):
    """Gemma 4 MoE core attention — switches between sliding and global window."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            config.window_size = (config.window_size - 1, 0)
        else:
            config.window_size = None

        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )


class Gemma4RotaryEmbedding(RotaryEmbedding):
    """Gemma 4 MoE position RoPE — dual local/global embeddings."""

    def __init__(
        self,
        rotary_base: int = 1_000_000,
        rotary_base_local: int = 10_000,
        global_kv_channels: int = 512,
        global_rotary_percent: float = 0.25,
        **kwargs,
    ):
        global_kwargs = {k: v for k, v in kwargs.items() if k not in ("rotary_percent", "kv_channels")}
        super().__init__(
            kv_channels=global_kv_channels,
            rotary_base=rotary_base,
            rotary_percent=1.0,
            **global_kwargs,
        )

        # HF proportional RoPE rotates pairs across the full attention head. Represent the
        # partial rotation with zero-frequency pairs instead of shortening the rotary width.
        rope_angles = int(global_rotary_percent * global_kv_channels // 2)
        nope_angles = global_kv_channels // 2 - rope_angles
        device = self.inv_freq.device
        rotated = 1.0 / (
            rotary_base
            ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device) / global_kv_channels)
        )
        non_rotated = torch.zeros(nope_angles, dtype=torch.float32, device=device)
        self.inv_freq = torch.cat([rotated, non_rotated], dim=0)

        self.rope_local = RotaryEmbedding(
            rotary_base=rotary_base_local,
            rotary_percent=1.0,
            **{k: v for k, v in kwargs.items() if k != "rotary_percent"},
        )

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> tuple[Tensor, Tensor]:
        if cp_group is not None:
            rope_global = super().forward(max_seq_len, offset, packed_seq, cp_group)
            rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq, cp_group)
            return (rope_local, rope_global)
        return self._forward_cached(max_seq_len, offset, packed_seq)

    @lru_cache(maxsize=32)
    def _forward_cached(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
    ) -> tuple[Tensor, Tensor]:
        rope_global = super().forward(max_seq_len, offset, packed_seq, None)
        rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq, None)
        return (rope_local, rope_global)
