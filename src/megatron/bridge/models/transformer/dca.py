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

"""Dual Chunk Attention modules for long-context transformer training.

The implementation follows the DCA algorithm proposed in "Training-Free
Long-Context Scaling of Large Language Models" and the reference Megatron-LM
implementation in NVIDIA/Megatron-LM#4048.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig
from megatron.core.typed_torch import apply_module
from torch import Tensor

from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.import_utils import safe_import_from


flash_attn_func, HAVE_FLASH_ATTN = safe_import_from("flash_attn", "flash_attn_func")


@dataclass(kw_only=True)
class DualChunkTransformerConfig(TransformerConfig):
    """Transformer configuration for Dual Chunk Attention.

    Args:
        dca_chunk_size: Maximum relative position used by successive-chunk
            attention.
        dca_local_size: Overlap reserved for local context. The effective
            chunk length is ``dca_chunk_size - dca_local_size``.
        yarn_rotary_scaling_factor: Scaling factor used by YARN RoPE.
        yarn_original_max_position_embeddings: Original YARN context length.
        yarn_beta_fast: YARN high-frequency correction parameter.
        yarn_beta_slow: YARN low-frequency correction parameter.
        yarn_mscale: Optional YARN attention concentration parameter.
        yarn_mscale_all_dim: Optional all-dimension YARN concentration parameter.
        yarn_correction_range_round_to_int: Whether YARN correction bounds are
            rounded to integers.
    """

    dca_chunk_size: int
    dca_local_size: int
    yarn_rotary_scaling_factor: float | None = None
    yarn_original_max_position_embeddings: int | None = None
    yarn_beta_fast: float | None = None
    yarn_beta_slow: float | None = None
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_correction_range_round_to_int: bool | None = None

    def finalize(self) -> None:
        """Finalize the base transformer config and validate DCA settings."""
        super().finalize()
        validate_dual_chunk_transformer_config(self)


def is_dual_chunk_attention_config(config: object) -> bool:
    """Return whether a model or transformer config enables DCA."""
    transformer_config = getattr(config, "transformer", config)
    return hasattr(transformer_config, "dca_chunk_size") and hasattr(transformer_config, "dca_local_size")


def _get_dca_sizes(config: MCoreTransformerConfig) -> tuple[int, int]:
    chunk_size = getattr(config, "dca_chunk_size", None)
    local_size = getattr(config, "dca_local_size", None)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("dca_chunk_size must be an integer.")
    if not isinstance(local_size, int) or isinstance(local_size, bool):
        raise ValueError("dca_local_size must be an integer.")
    return chunk_size, local_size


def validate_dual_chunk_transformer_config(config: MCoreTransformerConfig) -> None:
    """Validate transformer-level settings required by DCA.

    Args:
        config: Transformer configuration carrying ``dca_chunk_size`` and
            ``dca_local_size``.

    Raises:
        ValueError: If DCA parameters are invalid or an unsupported transformer
            feature is enabled.
    """
    chunk_size, local_size = _get_dca_sizes(config)
    if chunk_size <= 0:
        raise ValueError("dca_chunk_size must be positive.")
    if local_size < 0:
        raise ValueError("dca_local_size must be non-negative.")
    if chunk_size <= local_size:
        raise ValueError("dca_chunk_size must be greater than dca_local_size.")

    if getattr(config, "apply_rope_fusion", False):
        raise ValueError("DCA requires apply_rope_fusion=False.")
    if getattr(config, "fused_single_qkv_rope", False):
        raise ValueError("DCA requires fused_single_qkv_rope=False.")
    if getattr(config, "context_parallel_size", 1) != 1:
        raise ValueError("DCA does not support context_parallel_size > 1 yet.")
    if getattr(config, "attention_output_gate", False):
        raise ValueError("DCA does not support attention_output_gate yet.")
    if getattr(config, "fine_grained_activation_offloading", False):
        raise ValueError("DCA does not support fine_grained_activation_offloading yet.")
    if getattr(config, "cuda_graph_impl", "none") != "none":
        raise ValueError("DCA does not support CUDA graphs yet.")
    if getattr(config, "flash_decode", False):
        raise ValueError("DCA does not support flash decoding yet.")
    if getattr(config, "transformer_impl", "local") == "inference_optimized":
        raise ValueError("DCA does not support transformer_impl='inference_optimized'.")
    if getattr(config, "multi_latent_attention", False):
        raise ValueError("DCA does not support multi-latent attention.")
    if getattr(config, "experimental_attention_variant", None) is not None:
        raise ValueError("DCA cannot be combined with experimental_attention_variant.")
    if getattr(config, "mtp_num_layers", None) not in (None, 0):
        raise ValueError("DCA does not support MTP yet.")
    if getattr(config, "no_rope_freq", None) is not None:
        raise ValueError("DCA does not support disabling RoPE on selected layers.")
    if getattr(config, "softmax_type", "vanilla") != "vanilla":
        raise ValueError("DCA currently supports only softmax_type='vanilla'.")

    window_size = getattr(config, "window_size", None)
    if window_size not in (None, (-1, -1)):
        raise ValueError("DCA cannot be combined with sliding-window attention.")


def _merge_chunk_attention_outputs(outputs: list[Tensor], logsumexps: list[Tensor]) -> Tensor:
    """Merge partial attention outputs with differentiable LSE weights.

    Args:
        outputs: Partial outputs with shape ``[batch, heads, query, head_dim]``.
        logsumexps: Matching log-sum-exp tensors with shape
            ``[batch, heads, query, 1]``.

    Returns:
        The globally normalized attention output.

    Raises:
        ValueError: If the input lists are empty or have different lengths.
    """
    if len(outputs) != len(logsumexps):
        raise ValueError("outputs and logsumexps must have the same length.")
    if not outputs:
        raise ValueError("at least one partial output is required.")
    if len(outputs) == 1:
        return outputs[0]

    stacked_outputs = torch.stack(outputs, dim=0)
    stacked_lse = torch.stack(logsumexps, dim=0)
    output_dtype = stacked_outputs.dtype
    accumulation_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
    weights = torch.softmax(stacked_lse.to(accumulation_dtype), dim=0)
    merged = (stacked_outputs.to(accumulation_dtype) * weights).sum(dim=0)
    return merged.to(output_dtype)


def _required_rotary_length(seq_len: int, chunk_size: int, local_size: int) -> int:
    """Return the RoPE table length required by DCA position remapping."""
    chunk_len = chunk_size - local_size
    if seq_len <= chunk_len:
        return seq_len
    max_dca_position = min(2 * chunk_len - 1, chunk_size)
    return max(seq_len, max_dca_position + 1)


def _extend_rotary_frequencies(freqs: Tensor, required_len: int) -> Tensor:
    """Extend a linear RoPE frequency table to ``required_len`` positions."""
    if freqs.size(0) >= required_len:
        return freqs
    if freqs.size(0) < 2:
        raise ValueError("DCA needs at least two rotary positions to extend the frequency table.")

    step = freqs[1:2] - freqs[0:1]
    positions_shape = (required_len - freqs.size(0),) + (1,) * (freqs.ndim - 1)
    positions = torch.arange(
        freqs.size(0),
        required_len,
        dtype=freqs.dtype,
        device=freqs.device,
    ).reshape(positions_shape)
    extension = freqs[0:1] + positions * step
    return torch.cat((freqs, extension), dim=0)


class DualChunkAttention(MegatronModule):
    """Core Dual Chunk Attention operating on pre-RoPE Q/K/V tensors.

    Long sequences are split into effective chunks and evaluated through
    intra-chunk, successive-chunk, and inter-chunk attention. Their partial
    softmax results are merged with global log-sum-exp normalization.

    Args:
        config: Megatron transformer configuration.
        layer_number: One-indexed transformer layer number.
        attn_mask_type: Attention mask type. DCA requires causal attention.
        attention_type: Attention type. DCA supports self-attention only.
        attention_dropout: Optional attention dropout override.
        softmax_scale: Optional query-key scaling override.
        cp_comm_type: Context-parallel communication type.
        pg_collection: Process groups used by the parent attention module.
        dca_chunk_size: DCA chunk size. Defaults to the config field.
        dca_local_size: DCA local overlap size. Defaults to the config field.
    """

    def __init__(
        self,
        config: MCoreTransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        dca_chunk_size: int | None = None,
        dca_local_size: int | None = None,
    ) -> None:
        super().__init__(config=config)
        del cp_comm_type

        if attn_mask_type != AttnMaskType.causal:
            raise ValueError("DCA only supports causal attention masks.")
        if attention_type != "self":
            raise ValueError("DCA only supports self-attention.")

        config_chunk_size, config_local_size = _get_dca_sizes(config)
        self.chunk_size = dca_chunk_size if dca_chunk_size is not None else config_chunk_size
        self.local_size = dca_local_size if dca_local_size is not None else config_local_size
        if self.chunk_size <= 0:
            raise ValueError("dca_chunk_size must be positive.")
        if self.local_size < 0:
            raise ValueError("dca_local_size must be non-negative.")
        if self.chunk_size <= self.local_size:
            raise ValueError("dca_chunk_size must be greater than dca_local_size.")

        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.chunk_len = self.chunk_size - self.local_size
        self.cp_group = pg_collection.cp if pg_collection is not None else None

        head_dim = config.kv_channels
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        self.attention_dropout = attention_dropout if attention_dropout is not None else config.attention_dropout
        self.mscale = (
            _yarn_get_concentration_factor_from_config(config)
            if getattr(config, "yarn_rotary_scaling_factor", None) is not None
            else 1.0
        )

    def _apply_rope(self, tensor: Tensor, freqs: Tensor) -> Tensor:
        return apply_rotary_pos_emb(
            tensor,
            freqs,
            config=self.config,
            mscale=self.mscale,
            cp_group=self.cp_group,
        )

    def _compute_dca_freqs(
        self,
        rotary_pos_emb: tuple[Tensor, Tensor],
        seq_len: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        required_len = _required_rotary_length(seq_len, self.chunk_size, self.local_size)
        q_pos_emb = _extend_rotary_frequencies(q_pos_emb, required_len)
        k_pos_emb = _extend_rotary_frequencies(k_pos_emb, required_len)

        positions = torch.arange(seq_len, device=device)
        local_positions = positions % self.chunk_len
        key_freqs = k_pos_emb[local_positions]
        q_intra_freqs = q_pos_emb[local_positions]
        successor_positions = (local_positions + self.chunk_len).clamp(max=self.chunk_size)
        q_successor_freqs = q_pos_emb[successor_positions]
        inter_position = min(2 * self.chunk_len - 1, self.chunk_size)
        q_inter_freqs = q_pos_emb[inter_position : inter_position + 1].expand(seq_len, -1, -1, -1)
        return key_freqs, q_intra_freqs, q_successor_freqs, q_inter_freqs

    def _expand_gqa(self, key: Tensor, value: Tensor, num_heads: int) -> tuple[Tensor, Tensor]:
        num_kv_heads = key.size(2)
        if num_kv_heads == num_heads:
            return key, value
        if num_heads % num_kv_heads != 0:
            raise ValueError("DCA requires num_attention_heads to be divisible by num_query_groups.")
        repeat_factor = num_heads // num_kv_heads
        return key.repeat_interleave(repeat_factor, dim=2), value.repeat_interleave(repeat_factor, dim=2)

    def _unfused_attention_with_lse(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        causal: bool,
    ) -> tuple[Tensor, Tensor]:
        q_len = query.size(0)
        kv_len = key.size(0)
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        compute_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
        scores = torch.matmul(q.to(compute_dtype), k.to(compute_dtype).transpose(-2, -1)) * self.softmax_scale

        if causal and q_len > 1:
            kv_offset = kv_len - q_len
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), dtype=scores.dtype, device=scores.device),
                diagonal=1 + kv_offset,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
        output = torch.matmul(probabilities, v.to(compute_dtype)).to(query.dtype)
        return output, lse

    def _use_flash_attention(self, query: Tensor, *, requires_lse_gradient: bool = True) -> bool:
        if not HAVE_FLASH_ATTN or not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
            return False
        needs_lse_gradient = requires_lse_gradient and torch.is_grad_enabled()
        return query.size(-1) <= 256 and not (needs_lse_gradient and query.size(-1) == 256)

    def _attach_flash_lse_gradient(
        self,
        query: Tensor,
        key: Tensor,
        softmax_lse: Tensor,
        *,
        causal: bool,
    ) -> Tensor:
        """Attach exact Q/K gradients to a FlashAttention log-sum-exp tensor.

        FlashAttention 2.x returns the forward LSE but ignores its gradient. A
        second attention pass prepends a dummy key whose detached score matches
        the raw LSE. The dummy probability stays near one half and reconstructs
        a differentiable LSE without materializing the attention score matrix.
        """
        reference_lse = softmax_lse.detach().permute(0, 2, 1)
        query_extra = (reference_lse / self.softmax_scale).to(query.dtype).unsqueeze(-1)
        query_augmented = torch.cat((query, query_extra), dim=-1)
        key_augmented = torch.cat((key, torch.zeros_like(key[..., :1])), dim=-1)
        dummy_key = torch.zeros_like(key_augmented[:, :1])
        dummy_key[..., -1] = 1.0
        key_augmented = torch.cat((dummy_key, key_augmented), dim=1)

        probability_value = torch.zeros_like(key_augmented)
        probability_value[:, 0, :, -1] = 1.0
        probability_output = flash_attn_func(
            query_augmented,
            key_augmented,
            probability_value,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )
        dummy_probability = probability_output[..., -1].float()
        dummy_score = query_extra.squeeze(-1).float() * self.softmax_scale
        surrogate_lse = dummy_score + torch.log1p(-dummy_probability) - torch.log(dummy_probability)
        surrogate_lse = surrogate_lse.permute(0, 2, 1)

        return surrogate_lse + (softmax_lse - surrogate_lse).detach()

    def _flash_attention_with_lse(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        causal: bool,
        requires_lse_gradient: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if not HAVE_FLASH_ATTN:
            raise RuntimeError("flash-attn is not available.")

        q = query.permute(1, 0, 2, 3).contiguous()
        k = key.permute(1, 0, 2, 3).contiguous()
        v = value.permute(1, 0, 2, 3).contiguous()
        dropout_p = self.attention_dropout if self.training else 0.0
        output, softmax_lse, *_ = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=self.softmax_scale,
            causal=causal,
            return_attn_probs=True,
        )
        if requires_lse_gradient and torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
            softmax_lse = self._attach_flash_lse_gradient(q, k, softmax_lse, causal=causal)
        return output.permute(0, 2, 1, 3).contiguous(), softmax_lse.unsqueeze(-1)

    def _standard_attention_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        seq_len, batch_size, num_heads, head_dim = query.shape
        if self._use_flash_attention(query, requires_lse_gradient=False):
            output, _ = self._flash_attention_with_lse(
                query,
                key,
                value,
                causal=True,
                requires_lse_gradient=False,
            )
        else:
            key, value = self._expand_gqa(key, value, num_heads)
            output, _ = self._unfused_attention_with_lse(query, key, value, causal=True)
        output = output.permute(2, 0, 1, 3).contiguous()
        return output.reshape(seq_len, batch_size, num_heads * head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        /,
        *,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Apply DCA to pre-RoPE query, key, and value tensors.

        Args:
            query: Query tensor in ``[sequence, batch, heads, head_dim]`` layout.
            key: Key tensor in ``[sequence, batch, kv_heads, head_dim]`` layout.
            value: Value tensor matching ``key``.
            attention_mask: Explicit masks are not supported; causal masking is
                generated internally.
            attn_mask_type: Optional runtime mask override.
            attention_bias: Optional attention bias, currently unsupported.
            packed_seq_params: Packed-sequence metadata, currently unsupported.
            rotary_pos_emb: RoPE frequency table or query/key table pair.

        Returns:
            Attention output in ``[sequence, batch, hidden]`` layout.
        """
        if attention_mask is not None:
            raise ValueError("DCA does not support explicit attention masks yet.")
        if attn_mask_type is not None and attn_mask_type != AttnMaskType.causal:
            raise ValueError("DCA only supports causal attention masks.")
        if attention_bias is not None:
            raise ValueError("DCA does not support attention_bias yet.")
        if packed_seq_params is not None:
            raise ValueError("DCA does not support packed_seq_params yet.")

        seq_len, batch_size, num_heads, head_dim = query.shape
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        if seq_len <= self.chunk_len:
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                query = self._apply_rope(query, q_pos_emb[:seq_len])
                key = self._apply_rope(key, k_pos_emb[:seq_len])
            return self._standard_attention_forward(query, key, value)

        if rotary_pos_emb is None:
            raise ValueError("rotary_pos_emb is required when sequence length exceeds the DCA chunk length.")

        key_freqs, q_intra_freqs, q_successor_freqs, q_inter_freqs = self._compute_dca_freqs(
            rotary_pos_emb,
            seq_len,
            query.device,
        )
        key_rope = self._apply_rope(key, key_freqs)
        query_intra = self._apply_rope(query, q_intra_freqs)
        query_successor = self._apply_rope(query, q_successor_freqs)
        query_inter = self._apply_rope(query, q_inter_freqs)

        use_flash_attention = self._use_flash_attention(query)
        attention_with_lse = (
            self._flash_attention_with_lse if use_flash_attention else self._unfused_attention_with_lse
        )
        if not use_flash_attention:
            key_rope, value = self._expand_gqa(key_rope, value, num_heads)

        output_chunks: list[Tensor] = []
        num_chunks = (seq_len + self.chunk_len - 1) // self.chunk_len
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_len
            chunk_end = min(chunk_start + self.chunk_len, seq_len)
            chunk_outputs: list[Tensor] = []
            chunk_lses: list[Tensor] = []

            output_intra, lse_intra = attention_with_lse(
                query_intra[chunk_start:chunk_end],
                key_rope[chunk_start:chunk_end],
                value[chunk_start:chunk_end],
                causal=True,
            )
            chunk_outputs.append(output_intra)
            chunk_lses.append(lse_intra)

            if chunk_idx >= 1:
                previous_start = (chunk_idx - 1) * self.chunk_len
                output_successor, lse_successor = attention_with_lse(
                    query_successor[chunk_start:chunk_end],
                    key_rope[previous_start:chunk_start],
                    value[previous_start:chunk_start],
                    causal=False,
                )
                chunk_outputs.append(output_successor)
                chunk_lses.append(lse_successor)

            if chunk_idx >= 2:
                inter_end = (chunk_idx - 1) * self.chunk_len
                output_inter, lse_inter = attention_with_lse(
                    query_inter[chunk_start:chunk_end],
                    key_rope[:inter_end],
                    value[:inter_end],
                    causal=False,
                )
                chunk_outputs.append(output_inter)
                chunk_lses.append(lse_inter)

            merged = _merge_chunk_attention_outputs(chunk_outputs, chunk_lses)
            output_chunks.append(merged.permute(2, 0, 1, 3).contiguous())

        output = torch.cat(output_chunks, dim=0)
        return output.reshape(seq_len, batch_size, num_heads * head_dim)


class DualChunkSelfAttention(SelfAttention):
    """Training-only SelfAttention wrapper that delegates RoPE to DCA."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        key_value_states: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Project Q/K/V, run DCA with raw RoPE frequencies, and project output."""
        if key_value_states is not None:
            raise ValueError("DCA self-attention does not support cross-attention inputs.")
        if inference_context is not None or inference_params is not None:
            raise ValueError("DCA does not support inference or KV cache yet.")
        if rotary_pos_cos is not None or rotary_pos_sin is not None or rotary_pos_cos_sin is not None:
            raise ValueError("DCA requires rotary_pos_emb instead of precomputed rotary cos/sin tensors.")
        if attention_bias is not None:
            raise ValueError("DCA does not support attention_bias yet.")
        if packed_seq_params is not None:
            raise ValueError("DCA does not support packed_seq_params yet.")
        if sequence_len_offset is not None:
            raise ValueError("DCA does not support sequence_len_offset yet.")

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        qkv_output = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            output_gate=False,
            split_qkv=True,
        )
        query, key, value = qkv_output
        if self.checkpoint_core_attention and self.training:
            core_attention_output = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=None,
                packed_seq_params=None,
                core_attention_extra_kwargs={"rotary_pos_emb": rotary_pos_emb},
            )
        else:
            core_attention_output = apply_module(self.core_attention)(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=None,
                packed_seq_params=None,
                rotary_pos_emb=rotary_pos_emb,
            )
        return apply_module(self.linear_proj)(core_attention_output)
