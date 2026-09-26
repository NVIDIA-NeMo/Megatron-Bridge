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

"""Context-parallel attention for dense Gemma 4 global-attention layers."""

import copy
from typing import Callable

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_cp_partition_indices


_FLEX_ATTENTION: Callable | None = None


def _load_flex_attention() -> Callable:
    """Load and compile FlexAttention lazily to avoid import-time CUDA work."""
    global _FLEX_ATTENTION
    if _FLEX_ATTENTION is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True, fullgraph=True)
    return _FLEX_ATTENTION


def _nonpacked_local_positions(
    local_seq_len: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> Tensor:
    """Return positions owned by one rank in MCore's zigzag CP layout."""
    if cp_size == 1:
        return torch.arange(local_seq_len, device=device, dtype=torch.int64)
    if local_seq_len % 2 != 0:
        raise ValueError(
            f"Gemma 4 context parallelism requires an even rank-local sequence length, got {local_seq_len}."
        )

    chunk_len = local_seq_len // 2
    front_start = cp_rank * chunk_len
    back_start = (2 * cp_size - cp_rank - 1) * chunk_len
    return torch.cat(
        (
            torch.arange(front_start, front_start + chunk_len, device=device),
            torch.arange(back_start, back_start + chunk_len, device=device),
        )
    )


def _packed_local_positions(
    packed_seq_params: PackedSeqParams,
    local_seq_len: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    cp_group: torch.distributed.ProcessGroup,
) -> Tensor:
    """Return global packed-token positions owned by one CP rank."""
    positions = get_packed_seq_cp_partition_indices(
        packed_seq_params,
        total_tokens=local_seq_len * cp_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=device,
        cp_group=cp_group,
    )
    if positions.numel() != local_seq_len:
        raise ValueError(
            "Packed Gemma 4 CP partition metadata produced an unexpected local token count: "
            f"expected {local_seq_len}, got {positions.numel()}."
        )
    return positions


def _gather_rank_major_positions(local_positions: Tensor, cp_group: torch.distributed.ProcessGroup) -> Tensor:
    """Gather positions in the same rank-major order as gathered K/V tensors."""
    cp_size = torch.distributed.get_world_size(cp_group)
    gathered = [torch.empty_like(local_positions) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, local_positions.contiguous(), group=cp_group)
    return torch.cat(gathered)


def _nonpacked_rank_major_positions(
    local_seq_len: int,
    cp_size: int,
    device: torch.device,
) -> Tensor:
    """Return all zigzag positions in rank-major gather order without communication."""
    return torch.cat([_nonpacked_local_positions(local_seq_len, cp_size, rank, device) for rank in range(cp_size)])


def _packed_document_metadata(
    packed_seq_params: PackedSeqParams,
    query_positions: Tensor,
    key_positions: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return document ids and token validity in global packed coordinates."""
    cu_seqlens = packed_seq_params.cu_seqlens_q
    if cu_seqlens is None:
        raise ValueError("Packed Gemma 4 CP requires cu_seqlens_q.")
    cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens

    device = query_positions.device
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int64).flatten()
    cu_seqlens_padded = cu_seqlens_padded.to(device=device, dtype=torch.int64).flatten()
    if cu_seqlens.numel() != cu_seqlens_padded.numel() or cu_seqlens.numel() < 2:
        raise ValueError("Packed Gemma 4 CP requires matching, non-empty padded and unpadded cu_seqlens.")

    document_starts = cu_seqlens_padded[:-1]
    real_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    num_documents = document_starts.numel()

    def _document_ids(positions: Tensor) -> Tensor:
        ids = torch.searchsorted(cu_seqlens_padded, positions, right=True) - 1
        return ids.clamp(min=0, max=num_documents - 1)

    query_document_ids = _document_ids(query_positions)
    key_document_ids = _document_ids(key_positions)
    valid_queries = query_positions - document_starts[query_document_ids] < real_lengths[query_document_ids]
    valid_keys = key_positions - document_starts[key_document_ids] < real_lengths[key_document_ids]
    return query_document_ids, key_document_ids, valid_queries, valid_keys


def _position_extrema(
    positions: Tensor,
    block_size: int,
    valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return per-block min/max without a token-pair mask."""
    maximum = torch.iinfo(positions.dtype).max
    if valid is not None:
        low_values = torch.where(valid, positions, maximum)
        high_values = torch.where(valid, positions, -1)
    else:
        low_values = positions
        high_values = positions
    padding = (-positions.numel()) % block_size
    low_values = torch.cat(
        (low_values, torch.full((padding,), maximum, device=positions.device, dtype=positions.dtype))
    )
    high_values = torch.cat((high_values, torch.full((padding,), -1, device=positions.device, dtype=positions.dtype)))
    return (
        low_values.reshape(-1, block_size).min(dim=-1).values,
        high_values.reshape(-1, block_size).max(dim=-1).values,
    )


def _sparse_block_mask(
    query_positions: Tensor,
    key_positions: Tensor,
    query_document_ids: Tensor | None,
    key_document_ids: Tensor | None,
    valid_keys: Tensor | None,
    mask_mod: Callable,
    block_size: int = 128,
    window_left: int | None = None,
):
    """Construct FlexAttention metadata at block granularity.

    The active-block predicate is a conservative superset of the token mask;
    ``mask_mod`` handles partially occupied blocks exactly. This avoids the
    Q-by-K boolean tensor created internally by ``create_block_mask``.
    """
    from torch.nn.attention.flex_attention import BlockMask

    query_min, query_max = _position_extrema(query_positions, block_size)
    key_min, key_max = _position_extrema(key_positions, block_size, valid_keys)
    active = query_max[:, None] >= key_min[None, :]
    full = query_min[:, None] >= key_max[None, :]
    if window_left is not None:
        active = active & (key_max[None, :] >= query_min[:, None] - window_left)
        full = full & (key_min[None, :] >= query_max[:, None] - window_left)

    if query_document_ids is not None:
        assert key_document_ids is not None
        query_doc_min, query_doc_max = _position_extrema(query_document_ids, block_size)
        key_doc_min, key_doc_max = _position_extrema(key_document_ids, block_size, valid_keys)
        active = active & (query_doc_min[:, None] <= key_doc_max[None, :])
        active = active & (key_doc_min[None, :] <= query_doc_max[:, None])
        full = full & (query_doc_min == query_doc_max)[:, None]
        full = full & (key_doc_min == key_doc_max)[None, :]
        full = full & (query_doc_min[:, None] == key_doc_min[None, :])

    if valid_keys is not None:
        padding = (-valid_keys.numel()) % block_size
        padded_valid_keys = torch.cat((valid_keys, torch.zeros(padding, device=valid_keys.device, dtype=torch.bool)))
        key_block_all_valid = padded_valid_keys.reshape(-1, block_size).all(dim=-1)
        full = full & key_block_all_valid[None, :]
    full = active & full
    partial = active & ~full

    num_key_blocks = key_min.numel()
    key_block_indices = torch.arange(num_key_blocks, device=key_positions.device)

    def _ordered_indices(blocks: Tensor) -> tuple[Tensor, Tensor]:
        counts = blocks.sum(dim=-1).to(torch.int32)[None, None]
        indices = torch.where(blocks, key_block_indices[None, :], num_key_blocks).sort(dim=-1).values
        return counts, indices.to(torch.int32)[None, None]

    partial_counts, partial_indices = _ordered_indices(partial)
    full_counts, full_indices = _ordered_indices(full)
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(query_positions.numel(), key_positions.numel()),
    )


class Gemma4DenseHybridCPAttention(torch.nn.Module):
    """Use TE for supported sliding layers and all-gather FlexAttention otherwise.

    Gemma 4 dense alternates head-dim-256 sliding attention with head-dim-512
    global attention. TE handles sliding CP except packed THD with all-gather;
    its fused global backends do not support head dim 512. In both fallback
    cases, Q stays CP-sharded while K/V are gathered differentiably and
    consumed by FlexAttention without a quadratic score tensor.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str = "self",
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection=None,
    ) -> None:
        super().__init__()
        from megatron.bridge.models.gemma.modeling_gemma4 import _is_gemma4_sliding_layer

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.softmax_scale = softmax_scale if softmax_scale is not None else config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1.0
        if (attention_dropout if attention_dropout is not None else config.attention_dropout) != 0.0:
            raise ValueError("Gemma 4 global FlexAttention does not support attention dropout.")

        self.cp_group = (
            pg_collection.cp
            if pg_collection is not None
            else parallel_state.get_context_parallel_group(check_initialized=False)
        )
        self._sliding_attention: TEDotProductAttention | None = None
        self._sliding_packed_flex = False
        if _is_gemma4_sliding_layer(config, layer_number):
            sliding_config = copy.copy(config)
            sliding_config.window_attn_skip_freq = None
            sliding_cp_comm_type = self._sliding_cp_comm_type(config)
            # TE's all_gather CP kernel explicitly rejects THD, while its P2P
            # kernel does not support sliding windows. Use Flex for this packed
            # combination; TE A2A and non-packed all_gather remain unchanged.
            self._sliding_packed_flex = sliding_cp_comm_type == "all_gather"
            if self._sliding_packed_flex and config.window_size[1] != 0:
                raise NotImplementedError("Packed Gemma 4 CP Flex fallback supports causal sliding windows only.")
            self._sliding_attention = TEDotProductAttention(
                config=sliding_config,
                layer_number=layer_number,
                attn_mask_type=attn_mask_type,
                attention_type=attention_type,
                attention_dropout=attention_dropout,
                softmax_scale=self.softmax_scale,
                cp_comm_type=sliding_cp_comm_type,
                pg_collection=pg_collection,
            )
        self._block_mask_cache: dict[tuple[int, int, torch.device], object] = {}

    @staticmethod
    def _sliding_cp_comm_type(config: TransformerConfig) -> str:
        """Use A2A when local Q/KV heads divide across CP, otherwise all-gather."""
        tp_size = config.tensor_model_parallel_size
        cp_size = config.context_parallel_size
        local_query_heads = config.num_attention_heads // tp_size
        local_kv_heads = config.num_query_groups // tp_size
        if local_query_heads % cp_size == 0 and local_kv_heads % cp_size == 0:
            return "a2a"
        return "all_gather"

    @staticmethod
    def _kernel_options(head_dim: int) -> dict[str, int] | None:
        """Use smaller Triton tiles when head dim 512 exceeds shared-memory limits."""
        if head_dim <= 256:
            return None
        return {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 32,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 32,
            "num_stages": 2,
        }

    @staticmethod
    def _mask_mod(
        query_positions: Tensor,
        key_positions: Tensor,
        query_document_ids: Tensor | None,
        key_document_ids: Tensor | None,
        valid_keys: Tensor | None,
        window_left: int | None = None,
    ) -> Callable:
        """Build the causal, document-isolated mask predicate for FlexAttention."""

        def mask_mod(batch_idx, head_idx, query_idx, key_idx):
            allowed = key_positions[key_idx] <= query_positions[query_idx]
            if window_left is not None:
                allowed = allowed & (key_positions[key_idx] >= query_positions[query_idx] - window_left)
            if query_document_ids is not None:
                allowed = allowed & (query_document_ids[query_idx] == key_document_ids[key_idx])
            if valid_keys is not None:
                allowed = allowed & valid_keys[key_idx]
            return allowed

        return mask_mod

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        """Apply sliding TE attention or context-parallel FlexAttention."""
        effective_mask_type = attn_mask_type or self.attn_mask_type
        if self.attention_type != "self":
            raise NotImplementedError("Gemma 4 context-parallel attention supports self-attention only.")
        if effective_mask_type not in (AttnMaskType.causal, AttnMaskType.padding_causal):
            raise NotImplementedError(
                "Gemma 4 context-parallel attention supports causal and padding_causal masks only."
            )
        if packed_seq_params is not None:
            if packed_seq_params.qkv_format != "thd":
                raise NotImplementedError("Packed Gemma 4 context parallelism requires qkv_format='thd'.")
            if packed_seq_params.local_cp_size is not None or packed_seq_params.cp_group is not None:
                raise NotImplementedError(
                    "Gemma 4 context-parallel attention does not yet support runtime dynamic or hybrid CP groups."
                )
        if self._sliding_attention is not None and (packed_seq_params is None or not self._sliding_packed_flex):
            return self._sliding_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=effective_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        if attention_bias is not None:
            raise ValueError("Gemma 4 global context-parallel attention does not support attention_bias.")
        if attention_mask is not None:
            raise ValueError(
                "Gemma 4 global context-parallel attention derives its causal/packed mask from positions; "
                "an external attention_mask is not supported."
            )
        if self.cp_group is None:
            raise RuntimeError("Gemma 4 context parallelism requires an initialized CP process group.")

        packed_thd = query.ndim == 3
        if packed_thd:
            query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        local_seq_len, batch_size, num_query_heads, head_dim = query.shape
        if key.size(0) != local_seq_len or value.size(0) != local_seq_len:
            raise ValueError("Gemma 4 context-parallel attention currently supports self-attention training only.")
        cp_size = torch.distributed.get_world_size(self.cp_group)
        cp_rank = torch.distributed.get_rank(self.cp_group)

        # Keep sequence first so the collective's output remains rank-major.
        # Gathering K/V together halves the number of latency-bearing collectives.
        key_value = torch.stack((key, value), dim=1)
        key_value = gather_from_sequence_parallel_region(key_value, group=self.cp_group)
        key, value = key_value.unbind(dim=1)

        if packed_seq_params is None:
            query_positions = _nonpacked_local_positions(local_seq_len, cp_size, cp_rank, query.device)
            key_positions = _nonpacked_rank_major_positions(local_seq_len, cp_size, query.device)
        else:
            if not packed_thd or batch_size != 1:
                raise ValueError("Packed Gemma 4 CP expects THD query/key/value tensors with physical batch size 1.")
            query_positions = _packed_local_positions(
                packed_seq_params,
                local_seq_len,
                cp_size,
                cp_rank,
                query.device,
                self.cp_group,
            )
            key_positions = _gather_rank_major_positions(query_positions, self.cp_group)

        query_document_ids = key_document_ids = valid_queries = valid_keys = None
        if packed_seq_params is not None:
            query_document_ids, key_document_ids, valid_queries, valid_keys = _packed_document_metadata(
                packed_seq_params,
                query_positions,
                key_positions,
            )

        window_left = self.config.window_size[0] if self._sliding_attention is not None else None

        query_bhsd = query.permute(1, 2, 0, 3)
        key_bhsd = key.permute(1, 2, 0, 3)
        value_bhsd = value.permute(1, 2, 0, 3)
        flex_attention = _load_flex_attention()

        cache_key = None if packed_seq_params is not None else (local_seq_len, key.size(0), query.device)
        block_mask = self._block_mask_cache.get(cache_key) if cache_key is not None else None
        if block_mask is None:
            block_mask = _sparse_block_mask(
                query_positions,
                key_positions,
                query_document_ids,
                key_document_ids,
                valid_keys,
                self._mask_mod(
                    query_positions, key_positions, query_document_ids, key_document_ids, valid_keys, window_left
                ),
                window_left=window_left,
            )
            if cache_key is not None:
                self._block_mask_cache[cache_key] = block_mask

        context = flex_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=query_bhsd.size(1) != key_bhsd.size(1),
            kernel_options=self._kernel_options(head_dim),
        )
        if valid_queries is not None:
            context = context.masked_fill(~valid_queries[None, None, :, None], 0)
        context = (
            context.permute(2, 0, 1, 3)
            .contiguous()
            .view(
                local_seq_len,
                batch_size,
                num_query_heads * head_dim,
            )
        )
        return context.squeeze(1) if packed_thd else context
