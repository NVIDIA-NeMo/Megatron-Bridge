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

"""Gemma 4 text-only model providers.

Gemma4DenseProvider: Dense (E2B, E4B, and 31B) — builds GPTModel with local spec,
    dual RoPE, PLE, and shared KV.
Gemma4ModelProvider: MoE (26B-A4B and similar) — extends GPTModelProvider
    with TE-based layer spec, dual RoPE, and softcapped output layer.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from megatron.core.activations import fast_gelu
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gemma.gemma3_provider import Gemma3LanguageModelEmbedding
from megatron.bridge.models.gemma.modeling_gemma4 import (
    HAVE_TE,
    Gemma4DenseRotaryEmbedding,
    Gemma4OutputLayer,
    Gemma4RotaryEmbedding,
    _attach_ple_modules,
    _gemma4_block_spec,
    _install_ple_forward,
    _install_tied_kv,
    get_gemma4_layer_spec,
    wire_gemma4_kv_sharing,
)
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider


def _validate_gemma4_moe_orchestration(provider: GPTModelProvider) -> None:
    """Reject MCore execution modes bypassed by Gemma 4's custom MoE forward."""
    unsupported = []
    if provider.transformer_impl == "inference_optimized":
        unsupported.append("transformer_impl='inference_optimized'")
    if provider.cuda_graph_impl != "none":
        unsupported.append(f"cuda_graph_impl={provider.cuda_graph_impl!r}")
    if provider.moe_shared_expert_overlap:
        unsupported.append("moe_shared_expert_overlap=True")
    if provider.mlp_chunks_for_prefill > 1:
        unsupported.append("mlp_chunks_for_prefill > 1")
    if provider.mlp_chunks_for_training > 1:
        unsupported.append("mlp_chunks_for_training > 1")
    if provider.inference_fuse_tp_communication:
        unsupported.append("inference_fuse_tp_communication=True")
    recompute_modules = set(provider.recompute_modules or [])
    if provider.recompute_granularity == "selective" and "layernorm" in recompute_modules:
        unsupported.append("selective layernorm recompute")
    offload_modules = set(provider.offload_modules or [])
    if provider.fine_grained_activation_offloading and "mlp_norm" in offload_modules:
        unsupported.append("MLP norm activation offloading")
    if unsupported:
        raise ValueError(
            "Gemma 4 MoE's separate router/shared/routed inputs do not yet support: " + ", ".join(unsupported)
        )


def _install_gemma4_dense_load_state_aliases(model: torch.nn.Module) -> None:
    """Translate Gemma4 Dense checkpoint attention aliases before load_state_dict.

    Gemma4 Dense saves sliding/global attention tensors under separate names in
    dist-checkpoints because the two layer types have different sharded shapes.
    After dist-checkpoint load materializes a regular state_dict, PyTorch module
    loading expects the real module attribute name, ``self_attention``.
    """

    if getattr(model, "_gemma4_dense_load_state_aliases_installed", False):
        return

    def _load_state_dict_pre_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs

        for key in list(state_dict.keys()):
            if prefix and not key.startswith(prefix):
                continue

            canonical_key = None
            if ".self_attention_sliding." in key:
                canonical_key = key.replace(".self_attention_sliding.", ".self_attention.")
            elif ".self_attention_global." in key:
                canonical_key = key.replace(".self_attention_global.", ".self_attention.")

            if canonical_key is None:
                continue

            state_dict.setdefault(canonical_key, state_dict[key])
            state_dict.pop(key)

    model._register_load_state_dict_pre_hook(_load_state_dict_pre_hook)
    model._gemma4_dense_load_state_aliases_installed = True


# ---------------------------------------------------------------------------
# Dense provider
# ---------------------------------------------------------------------------


@dataclass
class Gemma4DenseProvider(GPTModelProvider):
    """Gemma 4 dense E2B, E4B, and 31B provider for clean Megatron-Core.

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
    # fast_gelu == gelu(x, approximate='tanh'), already registered in ACTIVATION_FUNC_MAP
    # as "gelu_pytorch_tanh" — required for HF export to recognise the activation.
    activation_func: Callable = field(default_factory=lambda: fast_gelu)

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
    attention_k_eq_v: bool = False
    sliding_window_rope_base: float = 10000.0
    full_attention_rope_base: float = 1000000.0
    full_attention_rope_partial_factor: float = 0.25
    num_kv_shared_layers: int = 18
    use_double_wide_mlp: bool = False
    per_layer_embed_vocab_size: int = 262144
    per_layer_embed_dim: int = 256
    final_logit_softcapping: float | None = 30.0

    num_moe_experts: Optional[int] = None
    moe_router_topk: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None

    def finalize(self) -> None:
        if self.use_double_wide_mlp:
            self.hetereogenous_dist_checkpoint = True
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
        _install_gemma4_dense_load_state_aliases(model)

        if (
            hasattr(model, "output_layer")
            and self.final_logit_softcapping is not None
            and not isinstance(model.output_layer, Gemma4OutputLayer)
        ):
            extend_instance(model.output_layer, Gemma4OutputLayer)

        return model


# ---------------------------------------------------------------------------
# MoE provider
# ---------------------------------------------------------------------------


@dataclass
class Gemma4ModelProvider(GPTModelProvider):
    """Configuration and provider for Megatron Core Gemma 4 MoE models."""

    seq_length: int = 262_144

    position_embedding_type: str = "rope"
    rotary_base: tuple = (10_000, 1_000_000)
    share_embeddings_and_output_weights: bool = True

    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1e-6

    kv_channels: int = 256
    num_query_groups: int = 8
    window_size: int = 1024
    interleaved_attn_pattern: tuple = (5, 1)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_backend: AttnBackend = AttnBackend.auto
    softmax_scale: float = 1.0
    qk_layernorm: bool = True
    attention_k_eq_v: bool = False

    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    global_rotary_percent: float = 0.25

    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = fast_gelu

    num_moe_experts: Optional[int] = 128
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 704
    moe_shared_expert_intermediate_size: int = 2112
    moe_shared_expert_overlap: bool = False
    moe_shared_expert_gate: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "aux_loss"
    moe_router_pre_softmax: bool = True
    moe_router_dtype: str = "fp32"
    moe_aux_loss_coeff: float = 0.001
    moe_permute_fusion: bool = True
    moe_layer_freq: int = 1

    final_logit_softcapping: float | None = 30.0

    flash_decode: bool = False
    transformer_layer_spec: Union[Callable, object] = field(
        default_factory=lambda: partial(_gemma4_block_spec, use_transformer_engine=HAVE_TE)
    )
    scatter_embedding_sequence_parallel: bool = True

    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def finalize(self) -> None:
        _validate_gemma4_moe_orchestration(self)
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma 4 MoE model."""
        rotary_base_local, rotary_base_global = self.rotary_base
        self.rotary_base = rotary_base_local
        try:
            model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        finally:
            self.rotary_base = (rotary_base_local, rotary_base_global)

        if hasattr(model, "embedding"):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=self,
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )

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

        if (
            hasattr(model, "output_layer")
            and self.final_logit_softcapping is not None
            and not isinstance(model.output_layer, Gemma4OutputLayer)
        ):
            extend_instance(model.output_layer, Gemma4OutputLayer)

        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()

        _install_tied_kv(model, self)

        return model
<<<<<<< HEAD


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

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
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

    def routing(self, logits, padding_mask=None, input_ids=None):
        """Apply standard routing, then renormalize and scale by per_expert_scale."""
        routing_probs, routing_map = super().routing(logits, padding_mask=padding_mask, input_ids=input_ids)
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


def _install_tied_kv(model: "torch.nn.Module", provider: "Gemma4ModelProvider") -> None:
    """Mark global attention layers that require K=V weight tying.

    In Gemma4, global attention layers share K and V projections (``v_proj``
    absent in the HF checkpoint).  At import time the bridge copies K rows into
    the V rows of ``linear_qkv.weight``.  This function marks each global
    ``Gemma4SelfAttention`` module with ``_tied_kv = True`` so that
    :meth:`Gemma4SelfAttention.get_query_key_value_tensors` can enforce V=K in
    the forward pass.

    K-V sharing is decided based on attention_k_eq_v field.
    Must be called after model construction so that the
    attention modules are already built.

    Note on gradient routing for LoRA: since V-rows = K-rows in the loaded
    checkpoint, the forward pass is numerically correct without any further
    modification.  Full gradient routing (accumulating dL/dV into K-rows) is
    left as a future improvement.
    """
    if not getattr(provider, "attention_k_eq_v", False):
        return

    num_global_kv_heads = getattr(provider, "num_global_key_value_heads", None)
    if not num_global_kv_heads:
        return  # No global KV heads configured

    pattern = provider.interleaved_attn_pattern

    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return

    for layer in decoder.layers:
        if _is_local_attn_layer(layer.layer_number, pattern):
            continue  # Sliding layers — skip
        attn = getattr(layer, "self_attention", None)
        if attn is None:
            continue
        # Mark this attention module so get_query_key_value_tensors knows to tie K=V.
        attn._tied_kv = True


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

        Fix: append '_sliding'/'_global' to the checkpoint storage keys to
        create per-type groups AND remap the prepended layer axis in
        ShardedTensors so global_shape[0], global_offset[0], and
        axis_fragmentations[0] reflect per-type layer counts rather than the
        total layer count.

        Example:
            state dict key: 'decoder.layers.0.self_attention.linear_qkv.weight'
            storage key:    'decoder.layers.0.self_attention_sliding.linear_qkv.weight'

        The returned state dict keys must stay unsuffixed so
        ``module.load_state_dict`` can load the tensors into the normal module
        hierarchy.
        """
        import dataclasses as _dataclasses

        from megatron.core.dist_checkpointing.mapping import ShardedObject as _SO
        from megatron.core.dist_checkpointing.mapping import ShardedTensor as _ST

        is_global = not _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern)
        suffix = "_global" if is_global else "_sliding"
        # Insert suffix before the trailing dot (prefix normally ends with '.')
        if prefix.endswith("."):
            storage_prefix = prefix[:-1] + suffix + "."
        else:
            storage_prefix = prefix + suffix

        state_dict = super().sharded_state_dict(prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata)

        def _storage_key(key: str) -> str:
            if key.startswith(prefix):
                return storage_prefix + key[len(prefix) :]
            return key.replace(".self_attention.", f".self_attention{suffix}.", 1)

        # Compute per-type layer count and this layer's rank within its type.
        # layer_number is 1-indexed in MCore.
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
                # Only remap the prepended layer axis (axis 0 when prepend_axis_num > 0)
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
                # ShardedObject (e.g. TE _extra_state): remap first axis if it matches total layers.
                # These have no prepend_axis_num — their global_shape IS the layer axis directly.
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

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        """Override to apply parameter-free RMSNorm to V after QKV split.

        HF Gemma4 applies ``v_norm = Gemma4RMSNorm(head_dim, with_scale=False)``
        to the value states. This is a parameter-free normalization: ``v / rms(v)``.

        For global attention layers (``self._tied_kv = True``), K=V tying is enforced
        here after ``super()`` has completed the all-gather for KV-replicated TP layouts.
        This ensures V=K throughout training for all tensor-parallel configs.
        """
        result = super().get_query_key_value_tensors(hidden_states, key_value_states, **kwargs)
        # When split_qkv=False (fused_single_qkv_rope / fused RoPE path), super() returns
        # (mixed_qkv, split_arg_list) — V-norm is not applied in this case.
        if len(result) < 3:
            return result
        query, key, value = result[0], result[1], result[2]
        # For global attention layers K=V tying is required (HF Gemma4 has no v_proj).
        # Enforced here — after the all-gather — so it is TP-safe for all configs.
        if getattr(self, "_tied_kv", False):
            value = key
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
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / global_kv_channels)
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
=======
>>>>>>> upstream/main
