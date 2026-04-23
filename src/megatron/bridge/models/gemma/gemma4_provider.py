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

"""
Gemma 4 Model Provider for Megatron-Core.

Gemma 4 is a Mixture-of-Experts (MoE) model with hybrid sliding/global attention.
Key differences from Gemma 3:
- MoE: 128 experts, top-k=8, plus a dense MLP path (mapped to shared experts)
- Heterogeneous attention: sliding layers use head_dim=256 / 8 KV heads,
  global layers use global_head_dim=512 / 2 KV heads with partial rotary (0.25)
- K=V sharing on global attention layers (V projection may be omitted)
- Per-layer scaling via ``layer_scalar`` buffer
- Dual pre/post layernorms for dense MLP vs MoE paths
"""

import copy
import math
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Callable, Optional, Tuple, Union

import torch
from megatron.core.activations import fast_gelu
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch import Tensor

from megatron.bridge.models.gemma.gemma3_provider import (
    Gemma3LanguageModelEmbedding,
    TERowParallelLinearLayerNorm,
    _is_local_attn_layer,
)
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.import_utils import safe_import_from


HAVE_TE = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")[1]
TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")
TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")


@dataclass
class Gemma4ModelProvider(GPTModelProvider):
    """Configuration and provider for Megatron Core Gemma 4 models.

    Gemma 4 is a MoE model with hybrid sliding/global attention. The dense MLP
    path is mapped to Megatron-Core's shared expert mechanism.
    """

    seq_length: int = 262_144

    # Embedding
    position_embedding_type: str = "rope"
    rotary_base: tuple = (10_000, 1_000_000)  # (local/sliding, global/full)
    share_embeddings_and_output_weights: bool = True

    # Norm — Gemma 4 uses STANDARD RMSNorm (x * w / rms(x)), NOT zero-centered gamma.
    # This differs from Gemma 1/2/3 which use zero-centered gamma (x * (1+w) / rms(x)).
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1e-6

    # Attention — base values are for sliding layers (majority)
    kv_channels: int = 256  # head_dim for sliding layers
    num_query_groups: int = 8  # num_kv_heads for sliding layers
    window_size: int = 1024
    interleaved_attn_pattern: tuple = (5, 1)  # (sliding, global)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_backend: AttnBackend = AttnBackend.auto
    softmax_scale: float = 1.0  # Gemma 4 uses QK norm; no 1/sqrt(d) scaling
    qk_layernorm: bool = True

    # Global attention overrides (applied per-layer in custom SelfAttention)
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    global_rotary_percent: float = 0.25

    # MLP / Activation
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = fast_gelu

    # MoE — dense MLP maps to shared experts (None for dense/non-MoE models)
    num_moe_experts: Optional[int] = 128
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 704
    moe_shared_expert_intermediate_size: int = 2112  # dense MLP intermediate
    moe_shared_expert_overlap: bool = False  # Must be False: Gemma4 uses separate pre/post norms
    moe_shared_expert_gate: bool = False  # no gate on shared expert, just sum
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "aux_loss"
    moe_router_pre_softmax: bool = True  # HF does softmax before topk
    moe_router_dtype: str = "fp32"
    moe_aux_loss_coeff: float = 0.001
    moe_permute_fusion: bool = True
    moe_layer_freq: int = 1  # all layers are MoE (dense path via shared expert)

    # Logit softcapping
    final_logit_softcapping: float = 30.0

    # Do not change
    flash_decode: bool = False
    transformer_layer_spec: Union[Callable, object] = field(
        default_factory=lambda: partial(
            _gemma4_block_spec, use_transformer_engine=HAVE_TE
        )
    )
    scatter_embedding_sequence_parallel: bool = True

    # Data type settings
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma 4 model.

        Replaces the model's embedding and RoPE with customized Gemma 4 variants
        that handle embedding scaling and dual local/global RoPE.
        """
        rotary_base_local, rotary_base_global = self.rotary_base
        # Trick megatron's RotaryEmbedding to initialize the model successfully
        self.rotary_base = rotary_base_local
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        self.rotary_base = (rotary_base_local, rotary_base_global)

        # Replace embedding with Gemma-style scaling (sqrt(hidden_size))
        if hasattr(model, "embedding"):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=self,
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )

        # Replace RoPE with dual local/global variant
        model.rotary_pos_emb = Gemma4RotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            use_cpu_initialization=self.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
            global_kv_channels=self.global_head_dim,
            global_rotary_percent=self.global_rotary_percent,
        )

        # Apply final_logit_softcapping to output layer
        if hasattr(model, "output_layer") and self.final_logit_softcapping:
            extend_instance(model.output_layer, Gemma4OutputLayer)

        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()

        # Tie K=V in global attention layers so fine-tuning preserves the
        # K=V constraint that the HF checkpoint relies on.
        _install_tied_kv(model, self)

        return model


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma 4 transformer layer with per-layer output scaling and extra post-norms.

    Gemma 4 has architectural features not present in standard MCore:
    - ``layer_scalar``: per-layer scaling applied to the full hidden state after residual add.
    - ``post_ffn_layernorm``: norm applied to the combined dense+MoE output before residual add
      (HF's ``post_feedforward_layernorm``).
    - ``post_moe_layernorm``: norm applied to routed expert output before combining with dense
      (HF's ``post_feedforward_layernorm_2``). Applied via a forward hook on the MoE layer.
    """

    def __init__(self, config, submodules, layer_number=1, **kwargs):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        self.register_buffer("layer_scalar", torch.ones(1, dtype=config.params_dtype))
        # HF pre_feedforward_layernorm (dense/shared-expert pre-norm) has no MCore
        # counterpart — stored as an inert buffer so it round-trips through export.
        self.register_buffer("pffl_weight", torch.ones(config.hidden_size, dtype=config.params_dtype))

        # Post-feedforward layernorm: applied to combined dense+MoE output before residual add
        # (HF: post_feedforward_layernorm)
        NormImpl = TENorm if HAVE_TE else torch.nn.Identity
        self.post_ffn_layernorm = NormImpl(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

    def _forward_post_mlp(self, mlp_output_with_bias, residual):
        """Override to apply post_ffn_layernorm before residual add, then layer_scalar."""
        from megatron.core.utils import make_viewless_tensor

        # Apply post_ffn_layernorm to the MLP output before residual add
        mlp_out = mlp_output_with_bias[0]
        mlp_bias = mlp_output_with_bias[1] if len(mlp_output_with_bias) > 1 else None

        # Post-feedforward norm (HF: post_feedforward_layernorm)
        normed = self.post_ffn_layernorm(mlp_out)
        if isinstance(normed, tuple):
            normed = normed[0]

        # Residual add then per-layer scaling:
        # HF: hidden_states = (residual + post_ffn_norm(mlp_out)) * layer_scalar
        if mlp_bias is not None:
            normed = normed + mlp_bias
        hidden_states = (residual + normed) * self.layer_scalar

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output


class Gemma4TopKRouter(TopKRouter):
    """Gemma 4 MoE router with per-expert scaling.

    Applies ``per_expert_scale`` to the routing probs after standard routing.
    Also renormalizes top-k weights before scaling (matching HF behavior).

    The router's input preprocessing (parameter-free RMSNorm + ``scale * scalar_root_size``)
    is fused into the router weight at load time in the bridge.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.register_buffer(
            "per_expert_scale",
            torch.ones(config.num_moe_experts, dtype=config.params_dtype),
        )
        # HF router.scale (per-channel input scaling, fused into router weight on import)
        # — stored as an inert buffer so it round-trips through export.
        self.register_buffer(
            "scale",
            torch.ones(config.hidden_size, dtype=config.params_dtype),
        )

    def routing(self, logits, padding_mask=None):
        """Apply standard routing, then renormalize and scale by per_expert_scale."""
        routing_probs, routing_map = super().routing(logits, padding_mask=padding_mask)
        # routing_probs: [num_tokens, num_experts] sparse — non-zero at selected experts
        # routing_map: [num_tokens, num_experts] boolean mask
        #
        # HF does: top_k_weights /= top_k_weights.sum(); top_k_weights *= per_expert_scale
        # In MCore sparse format, renormalize selected probs and apply per_expert_scale
        if routing_map is not None:
            # Renormalize: divide each token's selected probs by their sum
            prob_sums = routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            routing_probs = routing_probs / prob_sums
            # Apply per-expert scale element-wise (broadcasting over tokens)
            routing_probs = routing_probs * self.per_expert_scale.unsqueeze(0)
        return routing_probs, routing_map


class Gemma4MoELayer(MoELayer):
    """Gemma 4 MoE layer with post-routed-expert and post-shared-expert normalization.

    Applies ``post_feedforward_layernorm_2`` (pffl_ln2) to routed expert output and
    ``post_feedforward_layernorm_1`` (pffl_ln1) to shared expert output before combining.
    Standard MCore MoELayer simply sums routed + shared outputs without any intermediate norms.
    """

    def __init__(self, config, submodules, **kwargs):
        super().__init__(config=config, submodules=submodules, **kwargs)
        NormImpl = TENorm if HAVE_TE else torch.nn.Identity
        # HF: post_feedforward_layernorm_2 — applied to routed expert output
        self.post_moe_layernorm = NormImpl(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        # HF: post_feedforward_layernorm_1 — applied to shared expert (dense MLP) output
        self.post_shared_expert_layernorm = NormImpl(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

    def postprocess(self, output, shared_expert_output):
        """Apply post-MoE norms to routed and shared expert outputs, then combine."""
        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)
        # Norm routed expert output (HF: post_feedforward_layernorm_2)
        output = self.post_moe_layernorm(output)
        if isinstance(output, tuple):
            output = output[0]
        if shared_expert_output is not None:
            # Norm shared expert output (HF: post_feedforward_layernorm_1)
            normed_shared = self.post_shared_expert_layernorm(shared_expert_output)
            if isinstance(normed_shared, tuple):
                normed_shared = normed_shared[0]
            output = output + normed_shared
        return output


def _logit_softcapping(logits: torch.Tensor, scale: float | None) -> torch.Tensor:
    """Prevents logits from growing excessively: scale * tanh(logits / scale)."""
    if not scale:
        return logits
    return scale * torch.tanh(logits / scale)


class Gemma4OutputLayer(torch.nn.Module):
    """Mixin that applies final_logit_softcapping after the output linear layer."""

    def forward(self, *args, **kwargs):
        output, bias = super().forward(*args, **kwargs)
        output = _logit_softcapping(output, self.config.final_logit_softcapping)
        return output, bias


class Gemma4TiedKVMixin(torch.nn.Module):
    """Mixin for linear_qkv in Gemma4 global attention layers that enforces K=V tying.

    In Gemma4, global attention layers share K and V projections (``v_proj``
    absent in the HF checkpoint).  At import time the bridge copies K rows into
    the V rows of ``linear_qkv.weight``.  Without this mixin, K and V diverge
    during fine-tuning, and the trained V rows are silently discarded on export
    (since ``v_proj`` is absent in the HF state dict).

    This mixin wraps the ``forward`` pass of ``linear_qkv`` so that:

    1. The raw Q+K output is computed by the underlying TE linear.
    2. K is duplicated into the V slot (same tensor object, not a copy).
    3. ``cat([Q, K, K])`` is returned as if ``linear_qkv`` had V=K rows.

    **Gradient effect**:

    - K rows receive the accumulated gradient ``dL/dK + dL/dV`` from both the
      key *and* value attention paths — the correct combined gradient.
    - V rows receive zero gradient (the second K tensor is used in their place)
      and stay frozen at their initial K=V values throughout training.

    On export the existing bridge logic already drops ``v_proj`` for K=V layers,
    so no additional changes are needed for checkpoint round-trips.

    **TP handling**: per-rank Q and KV sizes are inferred from the actual
    output dimension at runtime, supporting both uniform sharding and the case
    where KV heads are replicated (``num_kv_heads < tp_size``).

    Usage (applied by ``_install_tied_kv``)::

        linear_qkv._tied_kv_q_size = q_total   # e.g. num_heads * global_head_dim
        linear_qkv._tied_kv_kv_size = kv_total  # e.g. num_global_kv_heads * global_head_dim
        extend_instance(linear_qkv, Gemma4TiedKVMixin)
    """

    def forward(self, *args, **kwargs):
        from megatron.core import parallel_state

        output = super().forward(*args, **kwargs)

        # Unpack the (potentially nested) tuple produced by TE column-parallel linears.
        # adapter_wrapper.base_linear_forward expects one of four patterns:
        #   1. tensor                        — no bias, no layernorm_output
        #   2. (tensor, bias)                — plain ColumnParallelLinear
        #   3. ((tensor, ln_out), bias)      — LayerNormColumnParallelLinear
        #   4. (tensor, bias, ln_out)        — fused with both
        if isinstance(output, tuple):
            first, rest = output[0], output[1:]
        else:
            first, rest = output, None

        # Pattern 3: first element is itself a tuple (fused layernorm output)
        if isinstance(first, tuple):
            qkv, ln_out = first[0], first[1:]
        else:
            qkv, ln_out = first, None

        out_dim = qkv.size(-1)
        tp = (
            parallel_state.get_tensor_model_parallel_world_size()
            if parallel_state.is_initialized()
            else 1
        )

        # Derive per-rank Q and KV output sizes.
        # Case (a): uniform sharding — each TP rank gets (q+kv+kv)/tp rows.
        # Case (b): KV replicated   — num_kv_heads < tp, so KV is not sharded.
        q_rank = self._tied_kv_q_size // tp
        kv_rank = self._tied_kv_kv_size // tp
        if q_rank + 2 * kv_rank != out_dim:
            # Fall back to KV-replicated layout
            kv_rank = self._tied_kv_kv_size
            q_rank = out_dim - 2 * kv_rank

        q = qkv[..., :q_rank]
        k = qkv[..., q_rank : q_rank + kv_rank]
        # Reference k twice so CatBackward accumulates dL/dk from both K and V paths
        tied = torch.cat([q, k, k], dim=-1)

        # Reconstruct the output in the original format, replacing the QKV tensor
        # with the tied version while preserving bias and layernorm_output slots.
        first_out = (tied,) + tuple(ln_out) if ln_out is not None else tied
        if rest is not None:
            return (first_out,) + tuple(rest)
        return first_out


def _install_tied_kv(model: "torch.nn.Module", provider: "Gemma4ModelProvider") -> None:
    """Apply :class:`Gemma4TiedKVMixin` to ``linear_qkv`` in every global attention layer.

    Skips dense models (``provider.num_moe_experts is None``) where K=V sharing
    has not been verified.  Must be called after model construction so that the
    attention modules and their ``linear_qkv`` weights are already built.
    """
    # Only confirmed for MoE models (26B-A4B family); skip dense variants
    if getattr(provider, "num_moe_experts", None) is None:
        return

    num_global_kv_heads = getattr(provider, "num_global_key_value_heads", None)
    if not num_global_kv_heads:
        return  # No global KV heads configured

    pattern = provider.interleaved_attn_pattern
    q_total = provider.num_attention_heads * provider.global_head_dim
    kv_total = num_global_kv_heads * provider.global_head_dim

    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return

    for layer in decoder.layers:
        if _is_local_attn_layer(layer.layer_number, pattern):
            continue  # Sliding layers share nothing — skip
        attn = getattr(layer, "self_attention", None)
        if attn is None:
            continue
        linear_qkv = getattr(attn, "linear_qkv", None)
        if linear_qkv is None:
            continue
        # Tag the instance with the sizes needed by the mixin's forward()
        linear_qkv._tied_kv_q_size = q_total
        linear_qkv._tied_kv_kv_size = kv_total
        extend_instance(linear_qkv, Gemma4TiedKVMixin)


def _gemma4_block_spec(config, use_transformer_engine=True, **kwargs):
    """Build Gemma 4 block spec: MoE or dense layer specs with patched attention.

    Uses ``get_gpt_decoder_block_spec`` to build standard specs, then patches
    each layer spec:
    - Attention module → Gemma4SelfAttention (heterogeneous head dims)
    - Core attention → Gemma4TEDotProductAttention (sliding/global window)
    - linear_proj → TERowParallelLinearLayerNorm (post-attention RMSNorm)
    - MoE models only: MoE layer → Gemma4MoELayer, router → Gemma4TopKRouter
    """
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_transformer_engine, **kwargs)

    for layer_spec in block_spec.layer_specs:
        # Replace layer module with Gemma4 variant (adds layer_scalar)
        layer_spec.module = Gemma4TransformerLayer

        attn_spec = layer_spec.submodules.self_attention
        # Replace attention module with Gemma4 variant (handles per-layer head_dim)
        if isinstance(attn_spec.module, type) and issubclass(attn_spec.module, SelfAttention):
            attn_spec.module = Gemma4SelfAttention
        # Replace core attention with Gemma4 variant (handles sliding/global window)
        if hasattr(attn_spec, "submodules") and attn_spec.submodules is not None:
            attn_spec.submodules.core_attention = Gemma4TEDotProductAttention
            # Post-attention RMSNorm (maps to HF post_attention_layernorm)
            if use_transformer_engine:
                attn_spec.submodules.linear_proj = TERowParallelLinearLayerNorm

        # MoE layer: only patch when the spec is an MoE layer (not dense MLP)
        mlp_spec = layer_spec.submodules.mlp
        if hasattr(mlp_spec, "module") and isinstance(mlp_spec.module, type) and issubclass(mlp_spec.module, MoELayer):
            mlp_spec.module = Gemma4MoELayer

            if hasattr(mlp_spec, "submodules") and mlp_spec.submodules is not None:
                # Replace router with Gemma4 variant (per_expert_scale + renormalization)
                mlp_spec.submodules.router = Gemma4TopKRouter

    return block_spec


class Gemma4SelfAttention(SelfAttention):
    """Gemma 4 self attention with heterogeneous sliding/global layers.

    - Sliding layers: head_dim=256, num_kv_heads=8, full rotary, local window
    - Global layers: head_dim=512, num_kv_heads=2, partial rotary (0.25), full attention
    - Value normalization: parameter-free RMSNorm applied to V after projection

    The config is deep-copied and overridden per-layer so that the QKV linear
    is constructed with the correct dimensions.
    """

    def __init__(self, config: TransformerConfig, layer_number: int, **kwargs):
        # Deep-copy config so per-layer overrides don't affect other layers
        config = copy.deepcopy(config)

        if not _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            # Global layer: override kv_channels; override num_query_groups only when
            # num_global_key_value_heads is explicitly set (non-MoE models may omit it
            # and reuse the same num_query_groups as sliding layers).
            config.kv_channels = config.global_head_dim
            if getattr(config, "num_global_key_value_heads", None) is not None:
                config.num_query_groups = config.num_global_key_value_heads

        super().__init__(config=config, layer_number=layer_number, **kwargs)
        self._v_norm_eps = config.layernorm_epsilon

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Override to separate sliding and global layers in the checkpoint.

        Sliding layers (head_dim=256) and global layers (head_dim=512) produce
        linear_qkv, linear_proj, q_layernorm, k_layernorm tensors with different
        shapes. dist_checkpointing validates two things per key group:
        1. Uniform global_shape — fails because sliding/global shapes differ.
        2. Full coverage of the global tensor — fails if only a subset of layers
           fill the group (e.g. 25 sliding layers can't cover a 30-slot group).

        Fix: append '_sliding'/'_global' suffix to create per-type groups AND
        remap the prepended layer axis in ShardedTensors so global_shape[0],
        global_offset[0], and axis_fragmentations[0] reflect per-type layer
        counts rather than the total layer count.

        Example:
            'decoder.layers.0.self_attention.'
          → 'decoder.layers.0.self_attention_sliding.'  (or _global)
        Loading works automatically because the same class produces the same
        suffixed keys on load.
        """
        import dataclasses as _dataclasses

        from megatron.core.dist_checkpointing.mapping import ShardedObject as _SO
        from megatron.core.dist_checkpointing.mapping import ShardedTensor as _ST

        is_global = not _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern)
        suffix = "_global" if is_global else "_sliding"
        # Insert suffix before the trailing dot (prefix always ends with '.')
        if prefix.endswith("."):
            modified_prefix = prefix[:-1] + suffix + "."
        else:
            modified_prefix = prefix + suffix

        state_dict = super().sharded_state_dict(
            prefix=modified_prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

        # Compute per-type layer count and this layer's rank within its type.
        # layer_number is 1-indexed in MCore.
        pattern = self.config.interleaved_attn_pattern
        total_layers = self.config.num_layers
        if is_global:
            type_total = sum(
                1 for i in range(1, total_layers + 1)
                if not _is_local_attn_layer(i, pattern)
            )
            type_rank = sum(
                1 for i in range(1, self.layer_number)
                if not _is_local_attn_layer(i, pattern)
            )
        else:
            type_total = sum(
                1 for i in range(1, total_layers + 1)
                if _is_local_attn_layer(i, pattern)
            )
            type_rank = sum(
                1 for i in range(1, self.layer_number)
                if _is_local_attn_layer(i, pattern)
            )

        def _remap(t):
            if isinstance(t, _ST):
                # Only remap the prepended layer axis (axis 0 when prepend_axis_num > 0)
                if t.prepend_axis_num <= 0 or t.global_shape[0] != total_layers:
                    return t
                new_global_shape = (type_total,) + t.global_shape[1:]
                new_global_offset = (type_rank,) + t.global_offset[1:]
                new_frags = (
                    (type_total,) + t.axis_fragmentations[1:]
                    if t.axis_fragmentations is not None
                    else None
                )
                return _dataclasses.replace(
                    t,
                    global_shape=new_global_shape,
                    global_offset=new_global_offset,
                    axis_fragmentations=new_frags,
                )
            if isinstance(t, _SO):
                # ShardedObject (e.g. TE _extra_state): remap first axis if it matches total layers.
                # These have no prepend_axis_num — their global_shape IS the layer axis directly.
                if not t.global_shape or t.global_shape[0] != total_layers:
                    return t
                new_global_shape = (type_total,) + t.global_shape[1:]
                new_global_offset = (type_rank,) + t.global_offset[1:]
                return _dataclasses.replace(
                    t,
                    global_shape=new_global_shape,
                    global_offset=new_global_offset,
                )
            return t

        def _fix(d):
            if isinstance(d, dict):
                return {k: _fix(v) for k, v in d.items()}
            return _remap(d)

        return _fix(state_dict)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        """Override to apply parameter-free RMSNorm to V after QKV split.

        HF Gemma4 applies ``v_norm = Gemma4RMSNorm(head_dim, with_scale=False)``
        to the value states. This is a parameter-free normalization: ``v / rms(v)``.
        """
        result = super().get_query_key_value_tensors(hidden_states, key_value_states, **kwargs)
        query, key, value = result[0], result[1], result[2]
        # Parameter-free RMSNorm on V: v / sqrt(mean(v^2) + eps)
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
        """Switch to either local or global RoPE embedding before forward."""
        assert isinstance(rotary_pos_emb, (tuple, list)) and len(rotary_pos_emb) == 2
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


class Gemma4TEDotProductAttention(TEDotProductAttention):
    """Gemma 4 core attention.

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
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            # Local sliding window attention, (left_window, right_window)
            config.window_size = (config.window_size - 1, 0)
        else:
            # Global full attention
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
    """Gemma 4 position RoPE embedding.

    Computes RoPE embeddings for both local (sliding) and global (full) attention layers.
    Local layers use full rotary with theta=10000.
    Global layers use **proportional** partial rotary (0.25) with theta=1000000.

    HF's proportional RoPE formula differs from standard partial rotary:
    - Standard:     inv_freq = 1/(base^(arange(0, dim, 2) / dim))       where dim = head_dim * percent
    - Proportional: inv_freq = 1/(base^(arange(0, dim, 2) / head_dim))  denominator is full head_dim

    This gives slower-decaying frequencies (spread across the full head_dim range).
    """

    def __init__(
        self,
        rotary_base: int = 1_000_000,
        rotary_base_local: int = 10_000,
        global_kv_channels: int = 512,
        global_rotary_percent: float = 0.25,
        **kwargs,
    ):
        # Global RoPE: proportional partial rotary with high theta
        global_kwargs = {k: v for k, v in kwargs.items() if k not in ("rotary_percent", "kv_channels")}
        super().__init__(
            kv_channels=global_kv_channels,
            rotary_base=rotary_base,
            rotary_percent=global_rotary_percent,
            **global_kwargs,
        )

        # Fix global inv_freq to match HF's proportional RoPE formula.
        # HF proportional: inv_freq = 1/(base^(arange / head_dim)) not 1/(base^(arange / dim))
        # where dim = int(head_dim * percent) and head_dim = global_kv_channels
        dim = int(global_kv_channels * global_rotary_percent)  # 128
        device = self.inv_freq.device
        self.inv_freq = 1.0 / (
            rotary_base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / global_kv_channels)
        )

        # Local RoPE: full rotary with low theta
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
        """Get (local_rope, global_rope) tuple.

        Local and global RoPE have different dimensions (e.g. 256 vs 64),
        so they cannot be stacked into a single tensor.
        """
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
        """Cached forward for hashable parameters only."""
        rope_global = super().forward(max_seq_len, offset, packed_seq, None)
        rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq, None)
        return (rope_local, rope_global)
