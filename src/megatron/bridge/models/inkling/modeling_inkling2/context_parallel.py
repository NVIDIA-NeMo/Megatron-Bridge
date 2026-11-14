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

"""Context parallelism for the Inkling2 decoder.

Megatron's context-parallel batch split gives CP rank ``r`` two zigzag chunks of
every sequence, or of every packed document: chunk ``r`` and chunk
``2 * CP - 1 - r``. Inkling2 reads across tokens in two places. The attention
gathers K and V from every rank and puts them back in global order, while each
rank keeps its own queries. The causal short convolutions on K, V, the attention
output, and the MoE output need only the ``W - 1`` tokens before each local
chunk; every rank shares the last ``W - 1`` rows of its chunks in a small
all-gather (the halo). Everything else in the decoder works token by token.
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from megatron.bridge.models.inkling.modeling_inkling2.ops import document_ids, document_positions


def context_parallel_token_indices(
    total_tokens: int,
    cp_size: int,
    device: torch.device,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the global index of every CP rank's local tokens, ``[cp_size, total_tokens // cp_size]``.

    Follows Megatron's split. Without ``cu_seqlens`` the sequence is cut into
    ``2 * cp_size`` equal chunks and rank ``r`` holds chunks ``r`` and
    ``2 * cp_size - 1 - r``. With packed documents, every segment between
    consecutive physical boundaries is split the same way, segment by segment, as
    Transformer Engine's ``thd_get_partitioned_indices`` does.

    Args:
        total_tokens: Sequence length before the split.
        cp_size: Context-parallel world size.
        device: Device of the returned indices.
        cu_seqlens: Physical document boundaries of a packed sequence, or ``None``.

    Returns:
        Long tensor whose row ``r`` lists rank ``r``'s tokens in local order.
    """
    chunks = 2 * cp_size
    if cu_seqlens is None:
        starts = torch.zeros(1, dtype=torch.int64, device=device)
        lengths = torch.full((1,), total_tokens, dtype=torch.int64, device=device)
    else:
        bounds = cu_seqlens.flatten().to(device=device, dtype=torch.int64)
        if int(bounds[0]) != 0 or int(bounds[-1]) != total_tokens:
            raise ValueError(f"Packed boundaries must run from 0 to {total_tokens}, got {bounds.tolist()}")
        starts, lengths = bounds[:-1], bounds[1:] - bounds[:-1]
        # Static-shape metadata repeats the last boundary; those segments are empty.
        starts, lengths = starts[lengths > 0], lengths[lengths > 0]
    if bool((lengths % chunks).any()):
        raise ValueError(f"Context parallelism needs every sequence length divisible by 2 * CP = {chunks}")
    chunk = lengths // chunks
    chunk_lengths = chunk.repeat_interleave(2)
    run_starts = torch.repeat_interleave(torch.cumsum(chunk_lengths, 0) - chunk_lengths, chunk_lengths)
    offsets = torch.arange(int(chunk_lengths.sum()), device=device) - run_starts
    ranks = []
    for rank in range(cp_size):
        chunk_starts = torch.stack((starts + rank * chunk, starts + (chunks - 1 - rank) * chunk), dim=1).flatten()
        ranks.append(torch.repeat_interleave(chunk_starts, chunk_lengths) + offsets)
    return torch.stack(ranks)


@dataclass(eq=False)
class ContextParallelLayout:
    """Where one CP rank's tokens sit in the full sequence, and how to reach the others.

    Attributes:
        group: Context-parallel process group.
        total_tokens: Sequence length before the CP split.
        local_index: Global index of each local token, ``[S_local]``.
        gather_order: Row of the rank-major all-gather that holds each global token, ``[S]``.
        local_positions: Offset of each local token inside its document, ``[S_local, B]``.
        local_documents: Document of each local token, ``[S_local]``.
        documents: Document of every global token, ``[S]``.
        halo_rows: Local rows shared with the other ranks: the last ``W - 1`` rows of every local chunk.
        halo_sources: For each lag ``< W``, the row of ``cat(local rows, gathered halos)`` that holds
            ``x[t - lag]`` for every local token ``t``, or ``-1`` before the start of its document.
        block_masks: FlexAttention block masks of this layout, by attention window.
    """

    group: torch.distributed.ProcessGroup
    total_tokens: int
    local_index: torch.Tensor
    gather_order: torch.Tensor
    local_positions: torch.Tensor
    local_documents: torch.Tensor
    documents: torch.Tensor
    halo_rows: torch.Tensor
    halo_sources: torch.Tensor
    block_masks: dict = field(default_factory=dict)

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather local rows ``[S_local, ...]`` into the full sequence ``[S, ...]`` in global order.

        The backward pass reduce-scatters, so every rank's contribution to a gathered row
        reaches the rank that owns it.
        """
        gathered = gather_from_sequence_parallel_region(
            x.contiguous(), tensor_parallel_output_grad=True, group=self.group
        )
        return gathered.index_select(0, self.gather_order)

    def extend(self, x: torch.Tensor) -> torch.Tensor:
        """Append every rank's halo rows to the local rows ``[S_local, ...]``."""
        halos = gather_from_sequence_parallel_region(
            x.index_select(0, self.halo_rows).contiguous(), tensor_parallel_output_grad=True, group=self.group
        )
        return torch.cat((x, halos), dim=0)

    def shift(self, extended: torch.Tensor, x: torch.Tensor, lag: int) -> torch.Tensor:
        """Return ``x[t - lag]`` for every local token, zero before the start of its document.

        Args:
            extended: ``extend(x)``.
            x: Local rows ``[S_local, ...]``.
            lag: Distance back in the global sequence, below the convolution width.
        """
        if lag == 0:
            return x
        sources = self.halo_sources[lag]
        shifted = extended.index_select(0, sources.clamp(min=0))
        keep = (sources >= 0).to(shifted.dtype)
        return shifted * keep.view(-1, *([1] * (shifted.ndim - 1)))


_layout_cache: "OrderedDict[tuple, ContextParallelLayout]" = OrderedDict()


def context_parallel_layout(
    group: torch.distributed.ProcessGroup,
    *,
    local_tokens: int,
    batch: int,
    device: torch.device,
    width: int,
    cu_seqlens: torch.Tensor | None = None,
) -> ContextParallelLayout | None:
    """Return this rank's token layout, or ``None`` without context parallelism.

    Args:
        group: Context-parallel process group.
        local_tokens: Local sequence length (after the CP split, before sequence parallelism).
        batch: Batch size (1 for packed sequences).
        device: Device of the layout tensors.
        width: Short-convolution width, which bounds the halo.
        cu_seqlens: Physical document boundaries of a packed batch, or ``None``.

    Returns:
        The cached layout for this sequence shape and document structure.
    """
    cp_size = group.size()
    if cp_size == 1:
        return None
    key = (
        id(group),
        local_tokens,
        batch,
        str(device),
        width,
        None if cu_seqlens is None else tuple(cu_seqlens.flatten().tolist()),
    )
    layout = _layout_cache.get(key)
    if layout is not None:
        _layout_cache.move_to_end(key)
        return layout
    if cu_seqlens is not None and batch != 1:
        raise ValueError("Packed sequences require batch size 1")

    total_tokens = local_tokens * cp_size
    rank = group.rank()
    indices = context_parallel_token_indices(total_tokens, cp_size, device, cu_seqlens)
    if indices.shape[1] != local_tokens:
        raise ValueError(f"CP rank holds {local_tokens} tokens, but the split gives {indices.shape[1]}")
    flat = indices.flatten()
    gather_order = torch.empty_like(flat)
    gather_order[flat] = torch.arange(flat.numel(), device=device)
    local_index = indices[rank]
    documents = document_ids(total_tokens, device, cu_seqlens)
    positions = document_positions(total_tokens, 1, device, cu_seqlens)[:, 0]
    local_positions = positions[local_index]

    # Every rank holds two chunks of each segment, with the same chunk lengths on every
    # rank, so the halo of each rank has the same rows and size.
    if cu_seqlens is None:
        chunk = torch.full((1,), total_tokens // (2 * cp_size), dtype=torch.int64, device=device)
    else:
        bounds = cu_seqlens.flatten().to(device=device, dtype=torch.int64)
        lengths = bounds[1:] - bounds[:-1]
        chunk = lengths[lengths > 0] // (2 * cp_size)
    chunk_lengths = chunk.repeat_interleave(2)
    chunk_ends = torch.cumsum(chunk_lengths, 0)
    tail = torch.clamp(chunk_lengths, max=width - 1)
    tail_starts = torch.repeat_interleave(chunk_ends - tail, tail)
    halo_rows = (
        tail_starts
        + torch.arange(int(tail.sum()), device=device)
        - torch.repeat_interleave(torch.cumsum(tail, 0) - tail, tail)
    )

    # Row of cat(local, halos) for every global token that this rank can reach; local rows win.
    halo_index = indices[:, halo_rows].flatten()
    row_of = torch.full((total_tokens,), -1, dtype=torch.int64, device=device)
    row_of[halo_index] = local_tokens + torch.arange(halo_index.numel(), device=device)
    row_of[local_index] = torch.arange(local_tokens, device=device)
    sources = [torch.arange(local_tokens, device=device)]
    for lag in range(1, width):
        inside = local_positions >= lag
        sources.append(torch.where(inside, row_of[(local_index - lag).clamp(min=0)], -1))
        if bool((sources[-1][inside] < 0).any()):
            raise RuntimeError(f"CP halo misses a token at lag {lag}; the split does not match the layout")

    layout = ContextParallelLayout(
        group=group,
        total_tokens=total_tokens,
        local_index=local_index,
        gather_order=gather_order,
        local_positions=local_positions.unsqueeze(1).expand(local_tokens, batch),
        local_documents=documents[local_index],
        documents=documents,
        halo_rows=halo_rows,
        halo_sources=torch.stack(sources),
    )
    _layout_cache[key] = layout
    while len(_layout_cache) > 4:
        _layout_cache.popitem(last=False)
    return layout
