# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Gemma-4 layer specification for Megatron-LM.
#
# Gemma-4 uses a 4-norm transformer structure (unlike standard 2-norm):
#   1. input_layernorm          : before self-attention     (pre-norm)
#   2. post_self_attn_layernorm : after  self-attention output, before residual add (post-norm)
#   3. pre_mlp_layernorm        : before MLP               (pre-norm)
#   4. post_mlp_layernorm       : after  MLP output, before residual add           (post-norm)
#
# Phase 3 — Dual RoPE:
#   Sliding-window layers use theta=10 000 (full rotation).
#   Full-attention layers use theta=1 000 000 with partial rotation (25 % of dims).
#   Gemma4RotaryEmbedding emits a (emb_sliding, emb_full) tuple;
#   Gemma4TransformerLayer._forward_attention resolves the correct one per layer.
#
# Phase 4 — Per-Layer Embeddings (PLE):
#   Reference: HF transformers modeling_gemma4.py (Gemma4TextDecoderLayer.forward)
#   per_layer_inputs [b, s, n_layers, ple_dim] computed in gpt_model._preprocess as:
#       (norm(linear(embed)) + embed_lookup) × 1/√2
#   Each layer receives per_layer_input [s, b, ple_dim] and applies:
#       residual = hidden
#       h = gelu(per_layer_input_gate(hidden))  # [s, b, ple_dim]
#       h = h × per_layer_input
#       h = per_layer_projection(h)              # [s, b, hidden_size]
#       h = post_per_layer_input_norm(h)
#       hidden = residual + h
#       hidden = hidden × layer_scalar
#
# Phase B — Attention corrections:
#   v_norm: RMSNorm without learnable scale applied to value states (Gemma4SelfAttention).
#
# Step 3 — Shared KV Cache (num_kv_shared_layers):
#   The last num_kv_shared_layers transformer layers reuse K/V from the last
#   non-shared layer of the same attention type (sliding or full).
#   Call wire_gemma4_kv_sharing(model) after model construction to set up references.
#
# Step 4 — attention_k_eq_v:
#   Full-attention layers (non-sliding) share K and V projections: V = k_proj(x).
#   The V portion of linear_qkv is unused; set to zero in the checkpoint loader.
#
# Step 5 — MoE block (enable_moe_block):
#   Each layer adds a sparse expert branch in parallel with the dense MLP.
#   Router + experts share the same hidden-state input as the dense MLP.
#   Three extra layernorms gate the combination (post_feedforward_1/2, pre_feedforward_2).

import copy
import types
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
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
# Step 5 — MoE router and experts (matching HF Gemma4TextRouter/Experts)
# ---------------------------------------------------------------------------


class Gemma4MoERouter(nn.Module):
    """Token router for Gemma-4 MoE block.

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

        # Scaleless RMSNorm (no learnable weight — matches HF with_scale=False)
        self.norm = Gemma4RMSNorm(config, hidden_size, eps=eps, with_scale=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            hidden_states: [tokens, hidden_size]  (2-D, pre-flattened)

        Returns:
            router_probs:  [tokens, num_experts]
            top_k_weights: [tokens, top_k]
            top_k_index:   [tokens, top_k]
        """
        h = self.norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        expert_scores = self.proj(h)
        router_probs = F.softmax(expert_scores.float(), dim=-1).to(h.dtype)
        top_k_weights, top_k_index = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probs, top_k_weights, top_k_index


class Gemma4MoEExperts(nn.Module):
    """Sparse expert collection for Gemma-4 MoE block.

    Mirrors HF ``Gemma4TextExperts``.  Experts share weight tensors stored as
    3-D parameters (num_experts, …).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        num_experts = getattr(config, 'num_experts', 1)
        hidden_size = config.hidden_size
        moe_intermediate_size = getattr(config, 'moe_intermediate_size', hidden_size)

        self.num_experts = num_experts
        # Gate+Up fused; split into halves inside forward (matches HF gate_up_proj)
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
        """
        Args:
            hidden_states: [tokens, hidden_size]
            top_k_index:   [tokens, top_k]
            top_k_weights: [tokens, top_k]

        Returns:
            Tensor [tokens, hidden_size]
        """
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
# Extended submodule dataclass
# ---------------------------------------------------------------------------


@dataclass
class Gemma4TransformerLayerSubmodules(TransformerLayerSubmodules):
    """TransformerLayerSubmodules extended with Gemma-4's extra post-sublayer norms.

    Inherits all standard fields from TransformerLayerSubmodules and adds:
      post_self_attn_layernorm   : applied to attention output before the residual add.
      post_mlp_layernorm         : applied to MLP output before the residual add.
      post_per_layer_input_norm  : applied to PLE output before the residual add (Phase 4).
    """

    post_self_attn_layernorm: LayerNormBuilder = IdentityOp
    post_mlp_layernorm: LayerNormBuilder = IdentityOp
    post_per_layer_input_norm: LayerNormBuilder = IdentityOp


# ---------------------------------------------------------------------------
# Gemma4SelfAttention: v_norm + Step 3 (shared KV) + Step 4 (k_eq_v)
# ---------------------------------------------------------------------------


class Gemma4SelfAttention(SelfAttention):
    """SelfAttention subclass for Gemma-4.

    Extends SelfAttention with:
    - v_norm: scaleless RMSNorm on value states (Phase B)
    - attention_k_eq_v: full-attention layers reuse K projection for V (Step 4)
    - Shared KV cache: last N layers reuse K/V from the last non-shared layer of
      the same attention type (Step 3).  Call wire_gemma4_kv_sharing(model) after
      model construction to complete the setup.
    """

    def __init__(self, config: TransformerConfig, submodules, layer_number: int, *args, **kwargs):
        attention_config = copy.copy(config)
        attention_config.softmax_scale = 1.0 if config.softmax_scale is None else config.softmax_scale
        # Gemma4 always uses per-head Q/K normalization; signal this so SelfAttention.__init__
        # accepts q_layernorm/k_layernorm in the submodule spec without raising an error.
        attention_config.qk_layernorm = True

        is_sliding = is_layer_window_attention(
            config.window_size, config.window_attn_skip_freq, layer_number
        )
        if not is_sliding:
            if getattr(config, 'global_kv_channels', None) is not None:
                attention_config.kv_channels = config.global_kv_channels
            if getattr(config, 'num_global_query_groups', None) is not None:
                attention_config.num_query_groups = config.num_global_query_groups

        super().__init__(attention_config, submodules, layer_number, *args, **kwargs)
        self.original_config = config
        self.is_gemma4_sliding_layer = is_sliding

        # Step 4: attention_k_eq_v — full-attention layers use K proj for V as well
        self.attention_k_eq_v = (
            getattr(config, 'attention_k_eq_v', False) and not is_sliding
        )

        # Step 3: Shared KV cache setup
        layer_idx = layer_number - 1  # 0-based
        num_layers = getattr(config, 'num_layers', 0)
        num_kv_shared = getattr(config, 'num_kv_shared_layers', 0)
        first_kv_shared_idx = num_layers - num_kv_shared  # first shared layer (0-based)

        self.is_kv_shared_layer = (num_kv_shared > 0) and (layer_idx >= first_kv_shared_idx)
        self.store_full_length_kv = False
        self.kv_shared_layer_index: Optional[int] = None  # 0-based source layer index

        if num_kv_shared > 0:
            skip_freq = getattr(config, 'window_attn_skip_freq', None)
            if isinstance(skip_freq, list):
                layer_is_sliding = [bool(x) for x in skip_freq[:num_layers]]
            elif isinstance(skip_freq, int) and skip_freq > 0:
                layer_is_sliding = [(i + 1) % skip_freq != 0 for i in range(num_layers)]
            else:
                layer_is_sliding = [False] * num_layers

            this_is_sliding = is_sliding

            if self.is_kv_shared_layer:
                # Find the last non-shared layer of the same attention type
                prev_types = layer_is_sliding[:first_kv_shared_idx]
                for i in range(len(prev_types) - 1, -1, -1):
                    if prev_types[i] == this_is_sliding:
                        self.kv_shared_layer_index = i
                        break
            else:
                # Mark this as a KV store layer if it's the LAST non-shared layer
                # of its attention type (its KV will be reused by shared layers)
                is_last_of_type = layer_idx < first_kv_shared_idx
                for i in range(layer_idx + 1, first_kv_shared_idx):
                    if layer_is_sliding[i] == this_is_sliding:
                        is_last_of_type = False
                        break
                self.store_full_length_kv = is_last_of_type

        # Runtime KV state (populated during forward pass)
        self._stored_kv: Optional[Tuple[Tensor, Tensor]] = None
        # Reference to source layer (set by wire_gemma4_kv_sharing)
        self._kv_source: Optional['Gemma4SelfAttention'] = None

    def _v_norm(self, value: Tensor) -> Tensor:
        vf = value.float()
        return (vf * torch.pow(vf.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)).to(value)

    def _get_k_eq_v_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Q/K/V extraction for HF-compatible ``attention_k_eq_v``.

        HF uses the raw K projection as V, then applies k_norm only to the key
        path and v_norm only to the value path.  Megatron's base implementation
        applies k_norm before returning K, so use the unsplit QKV path here to
        keep the raw K tensor available for the value path.
        """
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
        # ---- Shared-KV path -----------------------------------------------
        # This layer reuses K/V from a source layer; only Q is computed fresh.
        if self.is_kv_shared_layer:
            if not split_qkv or output_gate:
                # Fallback to normal computation for unsupported call patterns
                return super().get_query_key_value_tensors(
                    hidden_states, key_value_states, output_gate, split_qkv
                )
            # Compute Q (and ignore K/V from linear_qkv — their weights are zero)
            query, _k, _v = super().get_query_key_value_tensors(
                hidden_states, key_value_states, False, True
            )
            if self._kv_source is not None and self._kv_source._stored_kv is not None:
                key, value = self._kv_source._stored_kv
                key = key.to(query.device)
                value = value.to(query.device)
            else:
                # Source not wired yet — fall back to computed K/V with v_norm
                key, value = _k, _v
                value = self._v_norm(value)
            return query, key, value

        # ---- Normal path ---------------------------------------------------
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

        # v_norm: scaleless RMSNorm on head_dim axis (Phase B)
        value = self._v_norm(value)

        # Step 3: store K/V for shared layers that will reference this layer
        if self.store_full_length_kv:
            self._stored_kv = (key, value)

        if output_gate:
            return query, key, value, gate
        return query, key, value


# ---------------------------------------------------------------------------
# Custom TransformerLayer: 4-norm structure + dual-RoPE + PLE + MoE (Step 5)
# ---------------------------------------------------------------------------


class Gemma4TransformerLayer(TransformerLayer):
    """Transformer layer implementing Gemma-4's 4-norm residual structure.

    Differences from the standard TransformerLayer:
    * After self-attention output (before residual add): post_self_attn_layernorm.
    * After MLP output (before residual add): post_mlp_layernorm.

    Phase 3 — Dual RoPE:
      When rotary_pos_emb is a (emb_sliding, emb_full) tuple (from Gemma4RotaryEmbedding),
      _forward_attention selects the correct embedding for this layer based on
      window_attn_skip_freq.

    Phase 4 — Per-Layer Embeddings:
      After attention + MLP, applies:
        hidden = hidden + norm(proj(gelu(gate(hidden)) × per_layer_input))
      followed by hidden *= layer_scalar.

    Step 5 — MoE block:
      When enable_moe_block=True, the MLP output is combined with a sparse expert
      branch that routes from the pre-MLP residual state.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Gemma4TransformerLayerSubmodules,
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

        # Phase 4 — PLE modules (gate / projection / norm) + layer_scalar
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

        # Step 5 — MoE block (optional, enabled by config.enable_moe_block)
        _enable_moe = getattr(config, 'enable_moe_block', False)
        if _enable_moe:
            self.moe_router = Gemma4MoERouter(config)
            self.moe_experts = Gemma4MoEExperts(config)
            # Three extra norms used by the MoE combination path
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

    # ------------------------------------------------------------------
    # forward: intercept per_layer_input, apply PLE+scalar after MLP
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        per_layer_input = kwargs.pop('per_layer_input', None)

        hidden_states, context = self._forward_attention(*args, **kwargs)
        hidden_states = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )

        # Phase 4: PLE residual block (after attention + MLP)
        # Matches HF: gelu(gate(h)) × per_layer_input → proj → norm → residual
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            residual = hidden_states
            h = F.gelu(self.per_layer_input_gate(hidden_states), approximate='tanh')
            h = h * per_layer_input                     # [s, b, ple_dim]
            h = self.per_layer_projection(h)            # [s, b, hidden_size]
            h = self.post_per_layer_input_norm(h)
            hidden_states = residual + h

        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, context

    # ------------------------------------------------------------------
    # _forward_attention: dual-RoPE selection + 4-norm attention block
    # ------------------------------------------------------------------

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

        # Phase 3: resolve dual-RoPE tuple to single embedding for this layer
        if isinstance(rotary_pos_emb, tuple) and len(rotary_pos_emb) == 2:
            if is_layer_window_attention(
                self.config.window_size, self.config.window_attn_skip_freq, self.layer_number
            ):
                rotary_pos_emb = rotary_pos_emb[0]  # sliding-window embedding
            else:
                rotary_pos_emb = rotary_pos_emb[1]  # full-attention embedding

        # 1. Input layernorm
        input_layernorm_output = self.input_layernorm(hidden_states)
        if isinstance(input_layernorm_output, tuple):
            input_layernorm_output, residual = input_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()

        # 2. Self-attention
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

        # 3. post_self_attn_layernorm (before residual add)
        if isinstance(attention_output_with_bias, tuple):
            attn_out, attn_bias = attention_output_with_bias[0], attention_output_with_bias[1]
            attn_out = self.post_self_attn_layernorm(attn_out)
            attention_output_with_bias = (attn_out, attn_bias)
        else:
            attention_output_with_bias = self.post_self_attn_layernorm(attention_output_with_bias)

        # 4. Bias-dropout-add (residual connection)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, None  # Gemma-4 is decoder-only (no cross-attention)

    # ------------------------------------------------------------------
    # _forward_mlp: post_mlp_layernorm + optional Step 5 MoE combination
    # ------------------------------------------------------------------

    def _forward_mlp(
        self,
        hidden_states: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # 1. Pre-MLP layernorm; capture residual (= hidden_states before norm)
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)
        if isinstance(pre_mlp_layernorm_output, tuple):
            pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()

        # 2. Dense MLP
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)

        # 3. Step 5 — MoE: combine dense MLP output with sparse expert output
        if self.moe_router is not None:
            mlp_out = (
                mlp_output_with_bias[0]
                if isinstance(mlp_output_with_bias, tuple)
                else mlp_output_with_bias
            )

            # Dense branch: norm the MLP output
            dense_out = self.post_feedforward_layernorm_1(mlp_out)

            # Expert branch: route from pre-MLP residual (= hidden_states input)
            # [s, b, h] → [s*b, h] for token-level routing
            orig_shape = residual.shape
            hidden_flat = residual.reshape(-1, orig_shape[-1])

            _, top_k_weights, top_k_index = self.moe_router(hidden_flat)
            expert_in = self.pre_feedforward_layernorm_2(hidden_flat)
            expert_out = self.moe_experts(expert_in, top_k_index, top_k_weights)
            expert_out = expert_out.reshape(orig_shape)
            expert_out = self.post_feedforward_layernorm_2(expert_out)

            # Combine dense + expert outputs
            combined = dense_out + expert_out
            if isinstance(mlp_output_with_bias, tuple):
                mlp_output_with_bias = (combined, mlp_output_with_bias[1])
            else:
                mlp_output_with_bias = combined

        # 4. post_mlp_layernorm (before residual add)
        if isinstance(mlp_output_with_bias, tuple):
            mlp_out, mlp_bias = mlp_output_with_bias[0], mlp_output_with_bias[1]
            mlp_out = self.post_mlp_layernorm(mlp_out)
            mlp_output_with_bias = (mlp_out, mlp_bias)
        else:
            mlp_output_with_bias = self.post_mlp_layernorm(mlp_output_with_bias)

        # 5. Bias-dropout-add (residual connection)
        with self.bias_dropout_add_exec_handler():
            output = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        return output


# ---------------------------------------------------------------------------
# Step 3 helper: wire shared-KV source references after model construction
# ---------------------------------------------------------------------------


def wire_gemma4_kv_sharing(model: nn.Module) -> None:
    """Wire up shared-KV source references between Gemma4SelfAttention layers.

    Must be called once after the model is fully constructed.  Scans all
    ``Gemma4SelfAttention`` modules and links each shared layer to the
    attention module it should borrow K/V from.

    Args:
        model: The GPTModel (or any nn.Module containing Gemma4SelfAttention).
    """
    # Collect {0-based layer index → attention module}
    attn_by_layer: dict = {}
    for module in model.modules():
        if isinstance(module, Gemma4SelfAttention):
            idx = module.layer_number - 1  # convert 1-based to 0-based
            attn_by_layer[idx] = module

    for attn in attn_by_layer.values():
        if attn.is_kv_shared_layer and attn.kv_shared_layer_index is not None:
            source = attn_by_layer.get(attn.kv_shared_layer_index)
            if source is not None:
                attn._kv_source = source


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------


def get_gemma4_layer_spec(config: Optional[TransformerConfig] = None) -> ModuleSpec:
    """Return a ModuleSpec for a Gemma-4 transformer layer (local / non-TE implementation).

    Usage in training script:
        --spec megatron.bridge.models.gemma.gemma4_layer_specs gemma4_layer_spec

    Architecture:
        - GQA with qk_layernorm (q_norm, k_norm per head group) + v_norm (no scale)
        - Sliding-window causal attention (--window-size / --window-attn-skip-freq)
        - GEGLU MLP (--geglu)
        - 4-norm residual structure (see Gemma4TransformerLayer)

    Phase 3 (Dual RoPE):
        Enabled when --sliding-window-rope-base and --full-attention-rope-base are set.
        Gemma4TransformerLayer selects the correct embedding per layer at runtime.

    Phase 4 (Per-Layer Embeddings):
        Enabled when --per-layer-embed-vocab-size > 0.
        Applied to hidden states after attention + MLP (matches HF reference).

    Step 3 (Shared KV):
        Enabled when config.num_kv_shared_layers > 0.
        Call wire_gemma4_kv_sharing(model) after construction.

    Step 4 (attention_k_eq_v):
        Enabled when config.attention_k_eq_v=True.
        Full-attention layers use K projection for V; V weights in loader set to zero.

    Step 5 (MoE block):
        Enabled when config.enable_moe_block=True.
        Requires config.num_experts, config.moe_intermediate_size, config.top_k_experts.
    """
    backend = LocalSpecProvider()

    submodules = Gemma4TransformerLayerSubmodules(
        # Pre-attention norm
        input_layernorm=RMSNorm,

        # Self-attention: Gemma4SelfAttention adds v_norm + k_eq_v + shared-KV
        self_attention=ModuleSpec(
            module=Gemma4SelfAttention,
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

        # Post-attention norm (Gemma-4 specific)
        post_self_attn_layernorm=RMSNorm,

        # Pre-MLP norm
        pre_mlp_layernorm=RMSNorm,

        # MLP (gate + up projection via gated_linear_unit=True in config)
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
            ),
        ),
        mlp_bda=get_bias_dropout_add,

        # Post-MLP norm (Gemma-4 specific)
        post_mlp_layernorm=RMSNorm,

        # Post-PLE norm (Phase 4, applied to hidden_size output of per_layer_projection)
        post_per_layer_input_norm=RMSNorm,
    )

    return ModuleSpec(module=Gemma4TransformerLayer, submodules=submodules)


gemma4_layer_spec = get_gemma4_layer_spec()


# ---------------------------------------------------------------------------
# Gemma-4 Rotary Positional Embeddings
# ---------------------------------------------------------------------------


class _Gemma4ProportionalRotaryEmbedding(RotaryEmbedding):
    """Gemma-4 full-attention RoPE.

    Keeps the embedding width equal to the full attention head dimension.
    Only the first ``partial_rotary_factor`` portion receives non-zero
    frequencies; the remaining dimensions get zero frequency.
    The exponent denominator is the full head dimension, not the rotated subset.
    """

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


class Gemma4RotaryEmbedding(nn.Module):
    """Dual-theta Rotary Positional Embedding for Gemma-4.

    Gemma-4 uses two different RoPE configurations:
      - Sliding-window attention layers: theta = ``sliding_window_rope_base`` (10 000),
        full head-dim rotation.
      - Full-attention layers: theta = ``full_attention_rope_base`` (1 000 000),
        partial rotation controlled by ``full_attention_rope_partial_factor`` (0.25).

    ``forward()`` returns a ``(emb_sliding, emb_full)`` 2-tuple.
    ``Gemma4TransformerLayer._forward_attention`` selects the correct embedding for
    each layer based on ``config.window_attn_skip_freq`` and the layer number.
    """

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
        """Return ``(emb_sliding, emb_full)`` — one tensor per attention type."""
        emb_sliding = self.rope_sliding(
            max_seq_len, offset=offset, packed_seq=packed_seq, cp_group=cp_group
        )
        emb_full = self.rope_full(
            max_seq_len, offset=offset, packed_seq=packed_seq, cp_group=cp_group
        )
        return (emb_sliding, emb_full)

    def get_rotary_seq_len(self, *args, **kwargs) -> int:
        """Delegate to the sliding-window sub-embedding."""
        return self.rope_sliding.get_rotary_seq_len(*args, **kwargs)

    def get_cos_sin(self, max_seq_len: int, offset: int = 0):
        """Return ``((cos_s, sin_s), (cos_f, sin_f))``."""
        return (
            self.rope_sliding.get_cos_sin(max_seq_len, offset),
            self.rope_full.get_cos_sin(max_seq_len, offset),
        )


# ---------------------------------------------------------------------------
# Gemma-4 E4B Provider  (clean-MCore compatible: no Gemma4 CLI args needed)
# ---------------------------------------------------------------------------


@dataclass
class Gemma4E4BProvider:
    """Gemma-4 E4B (3.8B dense text) model provider for clean Megatron-Core.

    All Gemma4-specific settings are encoded here as dataclass fields so that
    no Gemma4-specific CLI arguments are required.  The provider builds a
    standard MCore GPTModel and then attaches PLE modules, wires shared-KV
    source references, and patches forward() to compute per-layer inputs.

    Usage in parity_check_e4b.py::

        provider = Gemma4E4BProvider()
        model = provider.build(pre_process=True, post_process=True)
        load_checkpoint([model], None, None)
    """

    # ---- Architecture (E4B defaults) ------------------------------------
    num_layers: int = 42
    hidden_size: int = 2560
    ffn_hidden_size: int = 10240
    num_attention_heads: int = 8
    num_query_groups: int = 2          # KV heads (both sliding and global layers)
    kv_channels: int = 256             # head_dim for sliding layers
    seq_length: int = 131072
    vocab_size: int = 262143
    make_vocab_size_divisible_by: int = 128

    # ---- Norms & activations --------------------------------------------
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    # geglu-tanh: matches HF gelu_pytorch_tanh
    activation_func: Callable = field(
        default_factory=lambda: partial(F.gelu, approximate="tanh")
    )

    # ---- Embeddings ------------------------------------------------------
    scale_embeddings_by_hidden_size: bool = True
    share_embeddings_and_output_weights: bool = True

    # ---- Dropout ---------------------------------------------------------
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # ---- Window attention (kept in clean MCore) --------------------------
    window_size: Optional[Tuple[int, int]] = (511, 0)
    window_attn_skip_freq: int = 6

    # ---- dtype -----------------------------------------------------------
    bf16: bool = True
    fp16: bool = False

    # ---- Gemma4-specific (read by gemma4_layer_specs via getattr) --------
    global_kv_channels: int = 512
    num_global_query_groups: int = 2
    sliding_window_rope_base: float = 10000.0
    full_attention_rope_base: float = 1000000.0
    full_attention_rope_partial_factor: float = 0.25
    num_kv_shared_layers: int = 18
    per_layer_embed_vocab_size: int = 262144
    per_layer_embed_dim: int = 256

    def build(
        self,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> "torch.nn.Module":
        """Build a Gemma-4 E4B GPTModel and attach Bridge-specific components.

        Steps:
          1. Build TransformerConfig from this provider's fields.
          2. Instantiate MCore GPTModel with get_gemma4_layer_spec.
          3. Attach PLE modules (per_layer_embedding / proj / norm).
          4. Wire shared-KV layer references.
          5. Patch model.forward() to compute per_layer_inputs.
        """
        from megatron.core.models.gpt import GPTModel
        from megatron.core.transformer.transformer_config import TransformerConfig

        # Build a TransformerConfig with all standard fields
        config_kwargs = {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_query_groups,
            "kv_channels": self.kv_channels,
            "normalization": self.normalization,
            "layernorm_epsilon": self.layernorm_epsilon,
            "gated_linear_unit": self.gated_linear_unit,
            "add_bias_linear": self.add_bias_linear,
            "activation_func": self.activation_func,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "window_size": self.window_size,
            "window_attn_skip_freq": self.window_attn_skip_freq,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "scale_embeddings_by_hidden_size": self.scale_embeddings_by_hidden_size,
        }
        config = TransformerConfig(**config_kwargs)

        # Inject Gemma4-specific fields needed during GPTModel.__init__()
        # (read by Gemma4SelfAttention / Gemma4TransformerLayer constructors via getattr)
        # NOTE: sliding_window_rope_base / full_attention_rope_base are intentionally
        # omitted here because clean MCore GPTModel.__init__() raises ValueError when
        # it detects those attributes.  They are injected AFTER model construction.
        for attr in (
            "global_kv_channels",
            "num_global_query_groups",
            "num_kv_shared_layers",
            "per_layer_embed_vocab_size",
            "per_layer_embed_dim",
        ):
            setattr(config, attr, getattr(self, attr))

        padded_vocab = (
            (self.vocab_size + self.make_vocab_size_divisible_by - 1)
            // self.make_vocab_size_divisible_by
            * self.make_vocab_size_divisible_by
        )

        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gemma4_layer_spec(config),
            vocab_size=padded_vocab,
            max_sequence_length=self.seq_length,
            position_embedding_type="rope",
            rotary_percent=1.0,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            pre_process=pre_process,
            post_process=post_process,
        )

        # Inject dual-RoPE attrs now that GPTModel.__init__() is complete
        setattr(config, "sliding_window_rope_base", self.sliding_window_rope_base)
        setattr(config, "full_attention_rope_base", self.full_attention_rope_base)
        setattr(config, "full_attention_rope_partial_factor", self.full_attention_rope_partial_factor)

        # Replace standard RoPE with Gemma4 dual-theta RoPE
        model.rotary_pos_emb = Gemma4RotaryEmbedding(config)

        # Attach PLE modules and wire shared-KV
        if pre_process:
            _attach_ple_modules(model, config, self)
        wire_gemma4_kv_sharing(model)

        # Patch forward to compute PLE before the decoder
        _install_ple_forward(model)

        return model


def _attach_ple_modules(
    model: "torch.nn.Module",
    config: "TransformerConfig",
    provider: Gemma4E4BProvider,
) -> None:
    """Add PLE embedding / projection / norm modules to a GPTModel instance."""
    import megatron.core.tensor_parallel as tp

    n_layers = provider.num_layers
    ple_dim = provider.per_layer_embed_dim
    ple_vocab = provider.per_layer_embed_vocab_size

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
    """Compute per_layer_inputs matching the formula in the pre-split GPTModel.

    Returns tensor of shape [b, s_local, num_layers, ple_dim], or None.
    """
    if not hasattr(model, "per_layer_embedding") or model.per_layer_embedding is None:
        return None
    if input_ids is None or decoder_input is None:
        return None

    ple_dim: int = model.config.per_layer_embed_dim
    n_layers: int = model.config.num_layers
    b: int = input_ids.shape[0]

    # 1. Token embedding: [b, s, n_layers * ple_dim]
    tok_emb = model.per_layer_embedding(input_ids) * (ple_dim ** 0.5)

    if getattr(model.config, "sequence_parallel", False):
        from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
        tok_emb = scatter_to_sequence_parallel_region(
            tok_emb.transpose(0, 1)
        ).transpose(0, 1)

    s_local: int = tok_emb.shape[1]
    tok_emb = tok_emb.view(b, s_local, n_layers, ple_dim)

    # 2. Model projection: decoder_input [s_local, b, h] → [b, s_local, n*ple_dim]
    mdl_proj, _ = model.per_layer_model_proj(decoder_input.transpose(0, 1))
    mdl_proj = mdl_proj * (model.config.hidden_size ** -0.5)
    mdl_proj = mdl_proj.view(b, s_local, n_layers, ple_dim)
    mdl_proj = model.per_layer_proj_norm(mdl_proj)

    # 3. Combine: (norm(proj) + tok_emb) × 1/√2
    return (mdl_proj + tok_emb) * (2.0 ** -0.5)


def _install_ple_forward(model: "torch.nn.Module") -> None:
    """Patch model.forward() to compute PLE and inject as per_layer_inputs.

    The patched forward:
      1. Computes the embedding output once.
      2. Computes PLE using that embedding output.
      3. Passes decoder_input (pre-computed) to GPTModel.forward() so that
         _preprocess() skips the embedding step (no double computation).
      4. Merges PLE into extra_block_kwargs so TransformerBlock threads it
         to each Gemma4TransformerLayer as per_layer_input.
    """
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
        # Compute embedding output (only once; passed to _preprocess to skip re-compute)
        if decoder_input is None and getattr(self, "pre_process", True):
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
            if getattr(self.config, "scale_embeddings_by_hidden_size", False):
                decoder_input = decoder_input * (self.config.hidden_size ** 0.5)

        # Compute PLE and merge into extra_block_kwargs
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
