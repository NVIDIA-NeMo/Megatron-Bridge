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

"""Native MCore composition for the causal and bidirectional D1 backbones."""

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.nn import functional as F

from megatron.bridge.models.d1.modeling_d1.short_conv import D1ShortConv


class D1RMSNorm(MegatronModule):
    """Preserve the release's rounding before the learned affine multiplication.

    Args:
        config: Native precision and initialization settings.
        hidden_size: Width of the final normalized dimension.
        eps: Positive variance stabilizer from the released configuration.
    """

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float) -> None:
        super().__init__(config)
        self.eps = eps
        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
        )
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=config.params_dtype))
        self.weight.allreduce = True
        self.weight.sequence_parallel = False
        self.weight.tensor_model_parallel = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize in float32, round, then apply the released affine weight."""
        values = hidden_states.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(hidden_states.dtype) * self.weight


class D1SelfAttention(SelfAttention):
    """Use native projections with precision-compatible single-rank attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        rotary_pos_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Rotate normalized Q/K in float32 before casting once to their input dtype.

        Args:
            hidden_states: Sequence-first states [length, batch, width].
            attention_mask: Boolean forbidden positions [batch, 1, length, length].
            rotary_pos_emb: Float32 half-split rotary angles.

        Returns:
            Native output projection and its optional deferred bias.
        """
        query, key, value = self.get_query_key_value_tensors(hidden_states)
        query = apply_rotary_pos_emb(
            query.float(), rotary_pos_emb.float(), self.config, cp_group=self.pg_collection.cp
        ).to(query.dtype)
        key = apply_rotary_pos_emb(
            key.float(), rotary_pos_emb.float(), self.config, cp_group=self.pg_collection.cp
        ).to(key.dtype)
        if self.attn_mask_type == AttnMaskType.causal:
            # The release uses PyTorch SDPA for right-padded causal chains.
            # Keep its forward/backward kernel contract while native MCore
            # modules continue to own QKV, norms, projection and conversion.
            # D1Backbone rejects partitioned execution before construction.
            attended = F.scaled_dot_product_attention(
                query.permute(1, 2, 0, 3),
                key.permute(1, 2, 0, 3),
                value.permute(1, 2, 0, 3),
                is_causal=True,
                dropout_p=0.0,
                scale=self.config.kv_channels**-0.5,
                enable_gqa=True,
            )
            context = attended.permute(2, 0, 1, 3).contiguous().flatten(2)
        else:
            context = self.core_attention(query, key, value, attention_mask, attn_mask_type=self.attn_mask_type)
        return self.linear_proj(context)


class D1BackboneLayer(MegatronModule):
    """Compose a native mixer, RMS norms, and gated MCore MLP.

    Args:
        config: Native transformer configuration.
        layer_number: One-based layer index for MCore attention.
        kind: Released schedule entry, conv or full_attention.
        causal: Select causal or bidirectional convolution.
    """

    def __init__(self, config: TransformerConfig, *, layer_number: int, kind: str, causal: bool) -> None:
        super().__init__(config)
        self.is_attention_layer = kind == "full_attention"
        self.operator_norm = D1RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.ffn_norm = D1RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        if self.is_attention_layer:
            self.self_attn = D1SelfAttention(
                config=config,
                layer_number=layer_number,
                attn_mask_type=AttnMaskType.causal if causal else AttnMaskType.arbitrary,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=D1RMSNorm,
                    k_layernorm=D1RMSNorm,
                ),
            )
        elif kind == "conv":
            self.conv = D1ShortConv(config, causal=causal)
        else:
            raise ValueError(f"Unsupported D1 layer kind: {kind}")
        self.feed_forward = MLP(
            config=config,
            ffn_hidden_size=config.ffn_hidden_size,
            submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        valid_tokens: torch.Tensor,
        media_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a residual mixer followed by a residual gated MLP.

        Args:
            hidden_states: Sequence-first states [length, batch, width].
            attention_mask: Boolean forbidden positions [batch, 1, length, length].
            rotary_pos_emb: MCore rotary angles [length, 1, 1, head_dim].
            valid_tokens: Boolean real-token mask [batch, length].
            media_lengths: Long prefix lengths [batch].

        Returns:
            Sequence-first residual states.
        """
        normalized = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            mixed, bias = self.self_attn(normalized, attention_mask, rotary_pos_emb=rotary_pos_emb)
            if bias is not None:
                mixed = mixed + bias
        else:
            mixed = self.conv(normalized, valid_tokens=valid_tokens, media_lengths=media_lengths)
        residual = hidden_states + mixed
        values, bias = self.feed_forward(self.ffn_norm(residual))
        if bias is not None:
            values = values + bias
        return residual + values


class D1Backbone(MegatronModule):
    """D1 language backbone built from native MCore blocks.

    Vision and audio prefixes are supplied by a modality wrapper. This module
    accepts their embeddings and owns language masking and placement semantics.
    Model partitioning and autoregressive inference caches are not implemented.

    Args:
        config: Native RMSNorm, GQA and gated-MLP configuration.
        vocab_size: Unpadded published vocabulary size.
        layer_types: One conv/full_attention entry per language layer.
        causal: Whether each query may only attend to earlier positions.
        rotary_base: Released RoPE base.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        vocab_size: int,
        layer_types: tuple[str, ...],
        causal: bool,
        rotary_base: float,
    ) -> None:
        super().__init__(config)
        if (
            config.tensor_model_parallel_size != 1
            or config.pipeline_model_parallel_size != 1
            or config.context_parallel_size != 1
            or config.sequence_parallel
        ):
            raise ValueError("D1Backbone currently requires TP=PP=CP=1 without sequence parallelism")
        if config.normalization != "RMSNorm" or config.layernorm_zero_centered_gamma:
            raise ValueError("D1 requires standard affine RMSNorm")
        if not config.qk_layernorm or not config.gated_linear_unit or config.activation_func is not F.silu:
            raise ValueError("D1 requires per-head Q/K normalization and a SiLU gated MLP")
        if config.add_bias_linear or config.add_qkv_bias or config.hidden_dropout or config.attention_dropout:
            raise ValueError("D1 requires bias-free language projections and zero dropout")
        if config.apply_query_key_layer_scaling or config.rotary_interleaved:
            raise ValueError("D1 requires unscaled attention and non-interleaved RoPE")
        if len(layer_types) != config.num_layers or any(
            kind not in {"conv", "full_attention"} for kind in layer_types
        ):
            raise ValueError("layer_types must contain one supported entry per language layer")
        if vocab_size <= 0 or rotary_base <= 0:
            raise ValueError("vocab_size and rotary_base must be positive")
        self.causal = causal
        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
        )
        self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size, device=device, dtype=config.params_dtype)
        if config.perform_initialization:
            config.init_method(self.embed_tokens.weight)
        self.embed_tokens.weight.allreduce = True
        self.embed_tokens.weight.tensor_model_parallel = False
        self.layers = nn.ModuleList(
            D1BackboneLayer(config, layer_number=index + 1, kind=kind, causal=causal)
            for index, kind in enumerate(layer_types)
        )
        self.embedding_norm = D1RMSNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.rotary_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            rotary_base=rotary_base,
            use_cpu_initialization=config.use_cpu_initialization,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        valid_tokens: torch.Tensor,
        media_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode an already packed sequence of media and text embeddings.

        Args:
            inputs_embeds: Batch-first states [batch, length, hidden_size].
            valid_tokens: Boolean real-token mask [batch, length].
            media_lengths: Long media-prefix lengths [batch].

        Returns:
            Batch-first normalized language states.
        """
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError("inputs_embeds must have shape [batch, length, hidden_size]")
        batch, length, _ = inputs_embeds.shape
        if length == 0 or valid_tokens.shape != (batch, length) or valid_tokens.dtype != torch.bool:
            raise ValueError("valid_tokens must be boolean with shape [batch, nonempty length]")
        if media_lengths.shape != (batch,) or media_lengths.dtype != torch.long:
            raise ValueError("media_lengths must be long with shape [batch]")
        if valid_tokens.device != inputs_embeds.device or media_lengths.device != inputs_embeds.device:
            raise ValueError("embeddings and masks must share a device")
        if torch.any(~valid_tokens.any(dim=1)):
            raise ValueError("every sequence must contain a real token")
        if torch.any((media_lengths < 0) | (media_lengths > valid_tokens.sum(dim=1))):
            raise ValueError("media_lengths must fit within the real sequence")
        positions = torch.arange(length, device=inputs_embeds.device)
        if self.causal and not torch.equal(valid_tokens, positions[None, :] < valid_tokens.sum(1)[:, None]):
            raise ValueError("causal D1 requires right-padded real tokens")
        mask = (~valid_tokens[:, None, None, :]).expand(batch, 1, length, length)
        if self.causal:
            mask = mask | (positions[None, :] > positions[:, None])[None, None, :, :]
        else:
            media_queries = positions[None, :] < media_lengths[:, None]
            text_keys = positions[None, :] >= media_lengths[:, None]
            mask = mask | (media_queries[:, None, :, None] & text_keys[:, None, None, :])
        rotary = self.rotary_emb(length)
        values = inputs_embeds.transpose(0, 1)
        for layer in self.layers:
            values = layer(
                values,
                attention_mask=mask,
                rotary_pos_emb=rotary,
                valid_tokens=valid_tokens,
                media_lengths=media_lengths,
            )
        return self.embedding_norm(values).transpose(0, 1)
