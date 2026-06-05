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

"""Gemma 4 model providers: MoE (Gemma4ModelProvider), Dense (Gemma4DenseProvider),
and their VL variants (Gemma4VLModelProvider, Gemma4DenseVLProvider)."""

import copy
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import torch
from megatron.core.activations import fast_gelu
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
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
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4DenseProvider, Gemma4VLModel
from megatron.bridge.utils.import_utils import safe_import_from


if TYPE_CHECKING:
    pass


HAVE_TE = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")[1]
TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")
TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")


# ---------------------------------------------------------------------------
# Gemma-4 MoE model components
# ---------------------------------------------------------------------------


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma 4 MoE transformer layer with per-layer output scaling and extra post-norms."""

    def __init__(self, config, submodules, layer_number=1, **kwargs):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        self.register_buffer("layer_scalar", torch.ones(1, dtype=config.params_dtype))
        self.register_buffer("pffl_weight", torch.ones(config.hidden_size, dtype=config.params_dtype))

        NormImpl = TENorm if HAVE_TE else torch.nn.Identity
        self.post_ffn_layernorm = NormImpl(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

    def _forward_post_mlp(self, mlp_output_with_bias, residual):
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

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.register_buffer(
            "per_expert_scale",
            torch.ones(config.num_moe_experts, dtype=config.params_dtype),
        )
        self.register_buffer(
            "scale",
            torch.ones(config.hidden_size, dtype=config.params_dtype),
        )

    def routing(self, logits, padding_mask=None, input_ids=None):
        routing_probs, routing_map = super().routing(logits, padding_mask=padding_mask, input_ids=input_ids)
        if routing_map is not None:
            prob_sums = routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            routing_probs = routing_probs / prob_sums
            routing_probs = routing_probs * self.per_expert_scale.unsqueeze(0)
        return routing_probs, routing_map


class Gemma4MoELayer(MoELayer):
    """Gemma 4 MoE layer with post-routed-expert and post-shared-expert normalization."""

    def __init__(self, config, submodules, **kwargs):
        super().__init__(config=config, submodules=submodules, **kwargs)
        NormImpl = TENorm if HAVE_TE else torch.nn.Identity
        self.post_moe_layernorm = NormImpl(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_shared_expert_layernorm = NormImpl(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

    def postprocess(self, output, shared_expert_output):
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
            modified_prefix = prefix[:-1] + suffix + "."
        else:
            modified_prefix = prefix + suffix

        state_dict = super().sharded_state_dict(
            prefix=modified_prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

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
                if t.prepend_axis_num <= 0 or t.global_shape[0] != total_layers:
                    return t
                new_global_shape = (type_total,) + t.global_shape[1:]
                new_global_offset = (type_rank,) + t.global_offset[1:]
                new_frags = (type_total,) + t.axis_fragmentations[1:] if t.axis_fragmentations is not None else None
                return _dataclasses.replace(
                    t,
                    global_shape=new_global_shape,
                    global_offset=new_global_offset,
                    axis_fragmentations=new_frags,
                )
            if isinstance(t, _SO):
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
        """Override to apply v_norm and enforce K=V tying for global attention."""
        result = super().get_query_key_value_tensors(hidden_states, key_value_states, **kwargs)
        if len(result) < 3:
            return result
        query, key, value = result[0], result[1], result[2]
        if getattr(self, "_tied_kv", False):
            value = key
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
            rotary_percent=global_rotary_percent,
            **global_kwargs,
        )

        dim = int(global_kv_channels * global_rotary_percent)
        device = self.inv_freq.device
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / global_kv_channels)
        )

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


# ---------------------------------------------------------------------------
# Gemma-4 MoE Provider
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

    final_logit_softcapping: float = 30.0

    flash_decode: bool = False
    transformer_layer_spec: Union[Callable, object] = field(
        default_factory=lambda: partial(_gemma4_block_spec, use_transformer_engine=HAVE_TE)
    )
    scatter_embedding_sequence_parallel: bool = True

    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma 4 MoE model."""
        rotary_base_local, rotary_base_global = self.rotary_base
        self.rotary_base = rotary_base_local
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
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

        if hasattr(model, "output_layer") and self.final_logit_softcapping:
            extend_instance(model.output_layer, Gemma4OutputLayer)

        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()

        _install_tied_kv(model, self)

        return model


# ---------------------------------------------------------------------------
# VL providers
# ---------------------------------------------------------------------------


@dataclass
class Gemma4VLModelProvider(Gemma4ModelProvider):
    """Model provider for Gemma 4 MoE Vision-Language models."""

    scatter_embedding_sequence_parallel: bool = False

    vision_config: Any = None
    text_config: Any = None
    audio_config: Any = None

    vision_soft_tokens_per_image: int = 280

    bos_token_id: int = 2
    eos_token_id: int = 1
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Gemma4VLModel:
        model = Gemma4VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class Gemma4DenseVLProvider(Gemma4DenseProvider):
    """Model provider for Dense Gemma 4 Vision-Language checkpoints."""

    scatter_embedding_sequence_parallel: bool = False

    vision_config: Any = None
    text_config: Any = None
    audio_config: Any = None

    vision_soft_tokens_per_image: int = 280

    bos_token_id: int = 2
    eos_token_id: int = 1
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Gemma4VLModel:
        model = Gemma4VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
