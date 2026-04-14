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
Custom SelfAttention for ERNIE 4.5 VL vision encoder.

Overrides the standard MCore SelfAttention to apply absolute 2D RoPE
positional embeddings instead of the standard relative RoPE.

ERNIE ViT uses non-interleaved RoPE (rotate_half style, splitting at
the midpoint: [-x2, x1]), corresponding to ``rotary_interleaved=False``
in MCore.  The RoPE frequencies are pre-computed as absolute position
embeddings based on 2D (height, width) grid coordinates.

This approach mirrors Qwen3VLSelfAttention but with ERNIE-specific
non-interleaved rotation.
"""

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import (
    BaseInferenceContext,
    SelfAttention,
    deprecate_inference_params,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor

from typing import Optional, Tuple, Union


def _apply_rotary_pos_emb_thd_absolute(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """Apply RoPE to ``thd`` (packed) format tensors using absolute position embeddings.

    Args:
        t: Input tensor of shape [total_tokens, num_heads, head_dim].
        cu_seqlens: Cumulative sequence lengths (currently unused, kept for API consistency).
        freqs: Rotary embedding frequencies of shape [total_tokens, 1, 1, head_dim].
        rotary_interleaved: Whether to use interleaved rotation.

    Returns:
        Tensor of shape [total_tokens, num_heads, head_dim] with RoPE applied.
    """
    # Unsqueeze to [total_tokens, 1, num_heads, head_dim] for bshd RoPE, then squeeze back
    return _apply_rotary_pos_emb_bshd(
        t[:, None], freqs, rotary_interleaved=rotary_interleaved
    ).squeeze(1)


def apply_rotary_pos_emb_absolute(
    t: Tensor,
    freqs: Tensor,
    config,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor:
    """Apply absolute RoPE, routing to bshd or thd format as appropriate.

    For ERNIE ViT, the freqs tensor has shape [total_tokens, 1, 1, head_dim]
    (absolute position embeddings, where the raw frequencies of shape
    [head_dim//2] are tiled 2x to cover the full head_dim), unlike standard
    relative RoPE where freqs is [max_seqlen, 1, 1, rotary_dim].

    Args:
        t: Input tensor (Q or K).
        freqs: Pre-computed RoPE frequencies.
        config: TransformerConfig (used for rotary_interleaved flag).
        cu_seqlens: If provided, indicates packed sequence (thd) format.

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    orig_dtype = t.dtype
    # Compute RoPE in fp32 for numerical stability
    t = t.float()

    if cu_seqlens is None:
        result = _apply_rotary_pos_emb_bshd(
            t, freqs, rotary_interleaved=config.rotary_interleaved
        )
    else:
        result = _apply_rotary_pos_emb_thd_absolute(
            t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved
        )

    return result.to(orig_dtype)


class ErnieVLSelfAttention(SelfAttention):
    """SelfAttention with absolute 2D RoPE for ERNIE ViT.

    Overrides the standard MCore SelfAttention.forward() to apply
    ``apply_rotary_pos_emb_absolute`` instead of the standard
    ``apply_rotary_pos_emb`` which expects relative position embeddings.

    This is necessary because ERNIE ViT pre-computes absolute 2D (H, W)
    position embeddings and passes them as rotary_pos_emb through the
    TransformerBlock, rather than using the standard MCore RoPE infrastructure
    that computes frequencies from sequential position IDs.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with absolute 2D RoPE for vision encoder.

        The main difference from the parent class is in the RoPE application
        section: we use ``apply_rotary_pos_emb_absolute`` which handles
        absolute position embeddings properly for both bshd and thd formats.

        Args:
            hidden_states: Input tensor [seq_len, batch, hidden_size].
            attention_mask: Attention mask (typically None for ViT).
            rotary_pos_emb: Pre-computed absolute 2D RoPE frequencies.
            packed_seq_params: Parameters for per-image packed sequence attention.
            (other args): See parent class SelfAttention.

        Returns:
            Tuple of (output, bias) where output is [seq_len, batch, hidden_size].
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # For self attention, duplicate the rotary_pos_emb if it isn't already a tuple
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        nvtx_range_push(suffix="qkv")
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        nvtx_range_pop(suffix="qkv")

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        nvtx_range_push(suffix="adjust_key_value")
        query, key, value, rotary_pos_emb, attn_mask_type, _block_table = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # Apply absolute 2D RoPE (the key difference)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb_absolute(
                    query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb_absolute(
                    key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
                )
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # Core attention computation
        # ==================================
        nvtx_range_push(suffix="core_attention")
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            # Reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        # =================
        # Output. [sq, b, h]
        # =================
        nvtx_range_push(suffix="linear_proj")
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix="linear_proj")

        return output, bias
