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

    # MoE — dense MLP maps to shared experts
    num_moe_experts: int = 128
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
    """Gemma 4 MoE layer with post-routed-expert normalization.

    Applies ``post_feedforward_layernorm_2`` to routed expert output before
    combining with shared expert (dense MLP) output. Standard MCore MoELayer
    simply sums routed + shared outputs without any intermediate norms.
    """

    def __init__(self, config, submodules, **kwargs):
        super().__init__(config=config, submodules=submodules, **kwargs)
        NormImpl = TENorm if HAVE_TE else torch.nn.Identity
        self.post_moe_layernorm = NormImpl(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

    def postprocess(self, output, shared_expert_output):
        """Apply post-MoE norm to routed expert output, then add shared expert."""
        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)
        # Norm routed expert output (HF: post_feedforward_layernorm_2)
        output = self.post_moe_layernorm(output)
        if isinstance(output, tuple):
            output = output[0]
        if shared_expert_output is not None:
            output = output + shared_expert_output
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


def _gemma4_block_spec(config, use_transformer_engine=True, **kwargs):
    """Build Gemma 4 block spec: MoE layer specs with patched attention.

    Uses ``get_gpt_decoder_block_spec`` to build standard MoE specs (including
    shared experts), then patches each layer spec:
    - Attention module → Gemma4SelfAttention (heterogeneous head dims)
    - Core attention → Gemma4TEDotProductAttention (sliding/global window)
    - linear_proj → TERowParallelLinearLayerNorm (post-attention RMSNorm)
    - shared_experts.linear_fc2 → TERowParallelLinearLayerNorm (post-dense-MLP RMSNorm)
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

        # MoE layer: replace with Gemma4MoELayer (adds post-MoE norm)
        mlp_spec = layer_spec.submodules.mlp
        if hasattr(mlp_spec, "module"):
            mlp_spec.module = Gemma4MoELayer

        # Post-dense-MLP RMSNorm on shared expert (maps to HF post_feedforward_layernorm_1)
        if hasattr(mlp_spec, "submodules") and mlp_spec.submodules is not None:
            # Replace router with Gemma4 variant (per_expert_scale + renormalization)
            mlp_spec.submodules.router = Gemma4TopKRouter
            shared_spec = getattr(mlp_spec.submodules, "shared_experts", None)
            if shared_spec is not None and hasattr(shared_spec, "submodules") and shared_spec.submodules is not None:
                if use_transformer_engine:
                    shared_spec.submodules.linear_fc2 = TERowParallelLinearLayerNorm

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
            # Global layer: override kv_channels and num_query_groups
            config.kv_channels = config.global_head_dim
            config.num_query_groups = config.num_global_key_value_heads

        super().__init__(config=config, layer_number=layer_number, **kwargs)
        self._v_norm_eps = config.layernorm_epsilon

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
