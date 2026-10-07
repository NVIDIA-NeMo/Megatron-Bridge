# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Conversion-free shared-stack DiffusionGemma text training composition."""

import torch
import torch.nn.functional as F
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.models.gpt import GPTModel
from torch import Tensor, nn

from megatron.bridge.models.gemma.gemma3_provider import _is_local_attn_layer
from megatron.bridge.models.gemma.modeling_gemma4 import Gemma4RMSNorm


KVPair = tuple[Tensor, Tensor]


def build_attention_masks(
    positions: Tensor, canvas_positions: Tensor, valid: Tensor, prefix_lengths: Tensor, window: int
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build SDPA masks (True means allowed) in absolute position coordinates."""
    expected_positions = torch.arange(positions.shape[1], device=positions.device)[None].expand_as(positions)
    if not torch.equal(positions, expected_positions):
        raise ValueError("encoder positions must be arange(S); left padding and position offsets are unsupported")
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("encoder_valid_mask must use right padding")
    if canvas_positions.shape[1] == 0:
        raise ValueError("canvas must contain at least one token")
    expected_canvas = prefix_lengths[:, None] + torch.arange(canvas_positions.shape[1], device=positions.device)[None]
    if not torch.equal(canvas_positions, expected_canvas):
        raise ValueError("prefix_lengths must be the canvas start column and canvas_positions must be contiguous")
    if (prefix_lengths < 0).any() or (prefix_lengths > valid.sum(-1)).any():
        raise ValueError("prefix_lengths must lie within the valid clean encoder sequence")
    if (canvas_positions >= positions.shape[1]).any():
        raise ValueError("canvas positions must lie within the encoder sequence")
    encoder = positions[:, None, :] <= positions[:, :, None]
    encoder = encoder & valid[:, None, :]
    canvas_start = canvas_positions[:, :1]
    clean_prefix = valid & (positions < canvas_start)
    clean_prefix = clean_prefix & (
        torch.arange(positions.shape[1], device=positions.device)[None] < prefix_lengths[:, None]
    )
    decoder = torch.cat(
        (
            clean_prefix[:, None, :].expand(-1, canvas_positions.shape[1], -1),
            torch.ones(
                (*canvas_positions.shape, canvas_positions.shape[1]), dtype=torch.bool, device=positions.device
            ),
        ),
        dim=-1,
    )
    encoder_local = encoder & ((positions[:, :, None] - positions[:, None, :]).abs() < window)
    local_prefix = clean_prefix & (positions >= canvas_start - window + 1)
    decoder_local = torch.cat(
        (
            local_prefix[:, None, :].expand(-1, canvas_positions.shape[1], -1),
            torch.ones(
                (*canvas_positions.shape, canvas_positions.shape[1]), dtype=torch.bool, device=positions.device
            ),
        ),
        dim=-1,
    )
    return (
        {"full_attention": encoder[:, None], "sliding_attention": encoder_local[:, None]},
        {"full_attention": decoder[:, None], "sliding_attention": decoder_local[:, None]},
    )


def prefix_canvas_attention(
    query: Tensor, key: Tensor, value: Tensor, allowed_mask: Tensor, encoder_kv: KVPair | None = None
) -> Tensor:
    """Attend with explicit gradient-bearing KV; tensors use native [S,B,H,D]."""
    if encoder_kv is not None:
        key = torch.cat((encoder_kv[0], key), dim=0)
        value = torch.cat((encoder_kv[1], value), dim=0)
    query, key, value = (tensor.permute(1, 2, 0, 3) for tensor in (query, key, value))
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed_mask, dropout_p=0.0, scale=1.0)
    return output.permute(2, 0, 1, 3).contiguous().flatten(2)


class DiffusionGemmaSelfConditioning(nn.Module):
    """HF AnalogBits gated projection, residual addition, and unscaled RMSNorm."""

    def __init__(self, hidden_size: int, intermediate_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        # Gemma4RMSNorm needs only the native parameter dtype from its config.
        from types import SimpleNamespace

        self.pre_norm = Gemma4RMSNorm(SimpleNamespace(params_dtype=dtype, sequence_parallel=False), hidden_size, eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.post_norm = Gemma4RMSNorm(
            SimpleNamespace(params_dtype=dtype, sequence_parallel=False), hidden_size, eps, with_scale=False
        )

    def forward(self, inputs_embeds: Tensor, signal: Tensor) -> Tensor:
        """Project a soft embedding signal and normalize the combined input."""
        normalized = self.pre_norm(signal)
        projected = self.down_proj(F.gelu(self.gate_proj(normalized), approximate="tanh") * self.up_proj(normalized))
        combined = inputs_embeds + projected
        return self.post_norm(combined)


class DiffusionGemmaModel(GPTModel):
    """Native GPT ownership/checkpointing with three explicit shared-stack passes."""

    def _stack(
        self,
        hidden: Tensor,
        positions: Tensor,
        masks: dict[str, Tensor],
        encoder_kvs: tuple[KVPair, ...] | None = None,
    ) -> tuple[Tensor, tuple[KVPair, ...]]:
        rope = self.rotary_pos_emb(int(positions.max().item()) + 1)
        own_kvs = []
        for index, layer in enumerate(self.decoder.layers):
            attention = layer.self_attention
            local = _is_local_attn_layer(attention.layer_number, self.config.interleaved_attn_pattern)
            kind = "sliding_attention" if local else "full_attention"
            query, key, value = attention.get_query_key_value_tensors(layer.input_layernorm(hidden))
            angles = rope[0 if local else 1][:, 0, 0][positions].transpose(0, 1).unsqueeze(2)
            query = apply_rotary_pos_emb(query, angles, config=attention.config)
            key = apply_rotary_pos_emb(key, angles, config=attention.config)
            own_kvs.append((key, value))
            context = prefix_canvas_attention(
                query, key, value, masks[kind], None if encoder_kvs is None else encoder_kvs[index]
            )
            projected, bias = attention.linear_proj(context)
            if bias is not None:
                projected = projected + bias
            residual = hidden + projected
            expert_input = layer.pre_mlp_layernorm(residual)
            shared_input = layer.pre_shared_expert_layernorm(residual)
            mlp_output, mlp_bias = layer.mlp.forward_with_separate_inputs(expert_input, shared_input, residual)
            normalized = layer.post_ffn_layernorm(mlp_output)
            if mlp_bias is not None:
                normalized = normalized + mlp_bias
            scalar = layer.encoder_layer_scalar if encoder_kvs is None else layer.layer_scalar
            hidden = (residual + normalized) * scalar
        return self.decoder.final_layernorm(hidden), tuple(own_kvs)

    def _logits(self, hidden: Tensor) -> Tensor:
        weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
        logits, _ = self.output_layer(hidden, weight=weight)
        return logits.transpose(0, 1).contiguous()[..., : self.config.vocab_size]

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        *,
        noisy_tokens: Tensor,
        canvas_positions: Tensor,
        encoder_valid_mask: Tensor,
        prefix_lengths: Tensor,
        self_conditioning_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return [B,S,V] encoder and [B,C,V] final decoder logits.

        The preview pass is detached; final decoder gradients flow through the
        original clean encoder KV. All positions are absolute, and prefix_lengths
        give the canvas start column in a right-padded sequence with
        position_ids=arange(S). Left padding and position offsets are unsupported.
        """
        if input_ids.shape != position_ids.shape or encoder_valid_mask.shape != input_ids.shape:
            raise ValueError("input_ids, position_ids and encoder_valid_mask must have identical [B,S] shape")
        if encoder_valid_mask.dtype != torch.bool or noisy_tokens.shape != canvas_positions.shape:
            raise ValueError("encoder_valid_mask must be bool; canvas tensors must have identical [B,C] shape")
        if prefix_lengths.shape != (input_ids.shape[0],):
            raise ValueError("prefix_lengths must have shape [B]")
        if self_conditioning_mask is None:
            self_conditioning_mask = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if self_conditioning_mask.dtype != torch.bool or self_conditioning_mask.shape != (input_ids.shape[0],):
            raise ValueError("self_conditioning_mask must be bool [B]")
        encoder_masks, decoder_masks = build_attention_masks(
            position_ids, canvas_positions, encoder_valid_mask, prefix_lengths, self.config.window_size
        )
        encoder_hidden, encoder_kvs = self._stack(self.embedding(input_ids, position_ids), position_ids, encoder_masks)
        encoder_logits = self._logits(encoder_hidden)
        noisy_embedding = self.embedding(noisy_tokens, canvas_positions)
        with torch.no_grad():
            preview_input = self.self_conditioning(noisy_embedding, torch.zeros_like(noisy_embedding))
            preview_hidden, _ = self._stack(preview_input, canvas_positions, decoder_masks, encoder_kvs)
            preview_logits = self._logits(preview_hidden)
            probabilities = preview_logits.softmax(-1, dtype=torch.float32)
        embedding_weight = self.embedding.word_embeddings.weight[: self.config.vocab_size]
        soft = probabilities.to(embedding_weight.dtype) @ embedding_weight
        soft = soft * self.config.hidden_size**0.5
        soft = soft * self_conditioning_mask[:, None, None].to(soft.dtype)
        decoder_input = self.self_conditioning(noisy_embedding, soft.transpose(0, 1))
        decoder_hidden, _ = self._stack(decoder_input, canvas_positions, decoder_masks, encoder_kvs)
        return encoder_logits, self._logits(decoder_hidden)
