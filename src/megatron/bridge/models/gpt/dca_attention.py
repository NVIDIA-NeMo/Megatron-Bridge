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

"""Bridge-local Dual Chunk Attention modules."""

import math

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from torch import Tensor

from megatron.bridge.utils.import_utils import safe_import_from


flash_attn_func, HAVE_FLASH_ATTN = safe_import_from("flash_attn", "flash_attn_func")


def _get_transformer_config(config):
    return getattr(config, "transformer", config)


def _get_dca_value(config, name: str, default):
    if hasattr(config, name):
        return getattr(config, name)
    transformer_config = _get_transformer_config(config)
    return getattr(transformer_config, name, default)


def _merge_chunk_attention_outputs(outputs: list[Tensor], logsumexps: list[Tensor]) -> Tensor:
    """Merge partial attention outputs using log-sum-exp renormalization."""
    if len(outputs) != len(logsumexps):
        raise ValueError("outputs and logsumexps must have the same length.")
    if len(outputs) == 0:
        raise ValueError("at least one partial output is required.")
    if len(outputs) == 1:
        return outputs[0]

    stacked_outputs = torch.stack(outputs, dim=0)
    stacked_lse = torch.stack(logsumexps, dim=0)
    max_lse = stacked_lse.max(dim=0).values
    weights = torch.exp(stacked_lse - max_lse).detach()
    weights = weights / weights.sum(dim=0)
    return (stacked_outputs * weights).sum(dim=0)


def _required_rotary_length(seq_len: int, chunk_size: int, local_size: int) -> int:
    """Return the minimum RoPE table length needed by DCA position remapping."""
    chunk_len = chunk_size - local_size
    if seq_len <= chunk_len:
        return seq_len
    max_dca_position = min(2 * chunk_len - 1, chunk_size)
    return max(seq_len, max_dca_position + 1)


def validate_dual_chunk_attention_config(config) -> None:
    """Validate a GPT config before enabling DCA."""
    if not getattr(config, "use_dual_chunk_attention", False):
        return

    transformer_config = _get_transformer_config(config)
    chunk_size = _get_dca_value(config, "dca_chunk_size", 8192)
    local_size = _get_dca_value(config, "dca_local_size", 1024)

    if chunk_size <= 0:
        raise ValueError("dca_chunk_size must be positive when DCA is enabled.")
    if local_size < 0:
        raise ValueError("dca_local_size must be non-negative when DCA is enabled.")
    if chunk_size <= local_size:
        raise ValueError("dca_chunk_size must be greater than dca_local_size when DCA is enabled.")

    position_embedding_type = getattr(config, "position_embedding_type", "rope")
    if position_embedding_type not in {"rope", "yarn"}:
        raise ValueError("DCA requires position_embedding_type to be 'rope' or 'yarn'.")

    if getattr(transformer_config, "apply_rope_fusion", False):
        raise ValueError("DCA requires apply_rope_fusion=False.")
    if getattr(transformer_config, "fused_single_qkv_rope", False):
        raise ValueError("DCA requires fused_single_qkv_rope=False.")
    if getattr(transformer_config, "context_parallel_size", 1) != 1:
        raise ValueError("DCA does not support context_parallel_size > 1 yet.")
    if getattr(config, "attention_backend", None) == AttnBackend.local:
        raise ValueError("DCA is not compatible with attention_backend=local.")
    if getattr(config, "use_transformer_engine_full_layer_spec", False):
        raise ValueError("DCA does not support use_transformer_engine_full_layer_spec yet.")
    if getattr(config, "restore_modelopt_state", False):
        raise ValueError("DCA does not support restore_modelopt_state yet.")
    if getattr(transformer_config, "mtp_num_layers", None) is not None or getattr(config, "mtp_enabled", False):
        raise ValueError("DCA does not support MTP yet.")
    if getattr(transformer_config, "cuda_graph_impl", "none") != "none":
        raise ValueError("DCA does not support CUDA graphs yet.")
    if getattr(transformer_config, "attention_output_gate", False):
        raise ValueError("DCA does not support attention_output_gate yet.")
    recompute_modules = getattr(transformer_config, "recompute_modules", None) or []
    if getattr(transformer_config, "recompute_granularity", None) == "selective" and "core_attn" in recompute_modules:
        raise ValueError("DCA does not support selective core attention recompute yet.")


class DualChunkAttention(MegatronModule):
    """Dual Chunk Attention core module.

    The input Q/K tensors are expected to be pre-RoPE. DCA applies its own RoPE
    remapping internally.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        dca_chunk_size: int | None = None,
        dca_local_size: int | None = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        if self.attn_mask_type != AttnMaskType.causal:
            raise ValueError("DCA only supports causal attention masks.")
        self.chunk_size = dca_chunk_size if dca_chunk_size is not None else getattr(config, "dca_chunk_size", 8192)
        self.local_size = dca_local_size if dca_local_size is not None else getattr(config, "dca_local_size", 1024)
        if self.chunk_size <= self.local_size:
            raise ValueError("dca_chunk_size must be greater than dca_local_size.")
        self.chunk_len = self.chunk_size - self.local_size

        head_dim = getattr(config, "kv_channels", None)
        if head_dim is None:
            head_dim = getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1)
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        self.attention_dropout = (
            attention_dropout if attention_dropout is not None else getattr(config, "attention_dropout", 0.0)
        )
        self.mscale = _yarn_get_concentration_factor_from_config(config)

    def _apply_rope(self, tensor: Tensor, freqs: Tensor) -> Tensor:
        return apply_rotary_pos_emb(tensor, freqs, config=self.config, mscale=self.mscale)

    def _compute_dca_freqs(
        self, rotary_pos_emb: tuple[Tensor, Tensor], seq_len: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        required_len = _required_rotary_length(seq_len, self.chunk_size, self.local_size)
        if q_pos_emb.size(0) < required_len or k_pos_emb.size(0) < required_len:
            raise ValueError(
                "DCA rotary_pos_emb is too short: "
                f"need at least {required_len}, got q={q_pos_emb.size(0)} k={k_pos_emb.size(0)}."
            )

        positions = torch.arange(seq_len, device=device)
        local_positions = positions % self.chunk_len
        key_freqs = k_pos_emb[local_positions]
        q_intra_freqs = q_pos_emb[local_positions]
        succ_positions = (local_positions + self.chunk_len).clamp(max=self.chunk_size)
        q_succ_freqs = q_pos_emb[succ_positions]
        inter_pos = min(2 * self.chunk_len - 1, self.chunk_size)
        q_inter_freqs = q_pos_emb[inter_pos : inter_pos + 1].expand(seq_len, -1, -1, -1)
        return key_freqs, q_intra_freqs, q_succ_freqs, q_inter_freqs

    def _expand_gqa(self, key: Tensor, value: Tensor, num_heads: int) -> tuple[Tensor, Tensor]:
        num_kv_heads = key.size(2)
        if num_kv_heads == num_heads:
            return key, value
        if num_heads % num_kv_heads != 0:
            raise ValueError("DCA requires num_attention_heads to be divisible by num_query_groups.")
        repeat_factor = num_heads // num_kv_heads
        return key.repeat_interleave(repeat_factor, dim=2), value.repeat_interleave(repeat_factor, dim=2)

    def _unfused_attention_with_lse(
        self, query: Tensor, key: Tensor, value: Tensor, *, causal: bool
    ) -> tuple[Tensor, Tensor]:
        q_len = query.size(0)
        kv_len = key.size(0)
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        if causal and q_len > 1:
            kv_offset = kv_len - q_len
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), dtype=scores.dtype, device=scores.device),
                diagonal=1 + kv_offset,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probs, v), lse

    def _use_flash_attention(self, query: Tensor) -> bool:
        return bool(HAVE_FLASH_ATTN and query.is_cuda and query.dtype in (torch.float16, torch.bfloat16))

    def _flash_attention_with_lse(
        self, query: Tensor, key: Tensor, value: Tensor, *, causal: bool
    ) -> tuple[Tensor, Tensor]:
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
        return output.permute(0, 2, 1, 3).contiguous(), softmax_lse.unsqueeze(-1)

    def _standard_attention_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        seq_len, batch_size, num_heads, head_dim = query.shape
        if self._use_flash_attention(query):
            output, _ = self._flash_attention_with_lse(query, key, value, causal=True)
            output = output.permute(2, 0, 1, 3).contiguous()
            return output.reshape(seq_len, batch_size, num_heads * head_dim)

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
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Run DCA on pre-RoPE Q/K/V tensors."""
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
            raise ValueError("rotary_pos_emb is required for DCA when seq_len > dca_chunk_size - dca_local_size.")

        key_freqs, q_intra_freqs, q_succ_freqs, q_inter_freqs = self._compute_dca_freqs(
            rotary_pos_emb, seq_len, query.device
        )
        key_rope = self._apply_rope(key, key_freqs)
        query_intra = self._apply_rope(query, q_intra_freqs)
        query_succ = self._apply_rope(query, q_succ_freqs)
        query_inter = self._apply_rope(query, q_inter_freqs)

        use_flash_attention = self._use_flash_attention(query)
        attn_with_lse = self._flash_attention_with_lse if use_flash_attention else self._unfused_attention_with_lse
        if not use_flash_attention:
            key_rope, value = self._expand_gqa(key_rope, value, num_heads)
        output_chunks = []
        num_chunks = (seq_len + self.chunk_len - 1) // self.chunk_len

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_len
            chunk_end = min(chunk_start + self.chunk_len, seq_len)
            chunk_outputs = []
            chunk_lses = []

            out_intra, lse_intra = attn_with_lse(
                query_intra[chunk_start:chunk_end],
                key_rope[chunk_start:chunk_end],
                value[chunk_start:chunk_end],
                causal=True,
            )
            chunk_outputs.append(out_intra)
            chunk_lses.append(lse_intra)

            if chunk_idx >= 1:
                prev_start = (chunk_idx - 1) * self.chunk_len
                prev_end = chunk_start
                out_succ, lse_succ = attn_with_lse(
                    query_succ[chunk_start:chunk_end],
                    key_rope[prev_start:prev_end],
                    value[prev_start:prev_end],
                    causal=False,
                )
                chunk_outputs.append(out_succ)
                chunk_lses.append(lse_succ)

            if chunk_idx >= 2:
                inter_end = (chunk_idx - 1) * self.chunk_len
                out_inter, lse_inter = attn_with_lse(
                    query_inter[chunk_start:chunk_end],
                    key_rope[:inter_end],
                    value[:inter_end],
                    causal=False,
                )
                chunk_outputs.append(out_inter)
                chunk_lses.append(lse_inter)

            merged = _merge_chunk_attention_outputs(chunk_outputs, chunk_lses)
            output_chunks.append(merged.permute(2, 0, 1, 3).contiguous())

        output = torch.cat(output_chunks, dim=0)
        return output.reshape(seq_len, batch_size, num_heads * head_dim)


class DualChunkSelfAttention(SelfAttention):
    """SelfAttention wrapper that lets DCA own RoPE application."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
        dca_chunk_size: int | None = None,
        dca_local_size: int | None = None,
    ):
        self.dca_chunk_size = dca_chunk_size
        self.dca_local_size = dca_local_size
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

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
        if key_value_states is not None:
            raise ValueError("DCA self attention does not support cross-attention inputs.")
        if inference_context is not None or inference_params is not None:
            raise ValueError("DCA does not support inference/KV cache yet.")
        if rotary_pos_cos is not None or rotary_pos_sin is not None or rotary_pos_cos_sin is not None:
            raise ValueError("DCA requires rotary_pos_emb, not precomputed rotary cos/sin tensors.")
        if attention_bias is not None:
            raise ValueError("DCA does not support attention_bias yet.")
        if packed_seq_params is not None:
            raise ValueError("DCA does not support packed_seq_params yet.")
        if sequence_len_offset is not None:
            raise ValueError("DCA does not support sequence_len_offset yet.")
        if self.config.attention_output_gate:
            raise ValueError("DCA does not support attention_output_gate yet.")
        if self.checkpoint_core_attention and self.training:
            raise ValueError("DCA does not support selective core attention recompute yet.")

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        core_attn_out = apply_module(self.core_attention)(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=self.attn_mask_type,
            attention_bias=None,
            packed_seq_params=None,
            rotary_pos_emb=rotary_pos_emb,
        )
        return apply_module(self.linear_proj)(core_attn_out)
