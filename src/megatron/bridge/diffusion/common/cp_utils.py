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

"""Context-parallel helpers for the sbd_block_diff diffusion LM.

Megatron context parallelism shards the sequence with a "load-balanced zigzag"
layout: the sequence is cut into ``2 * cp_size`` chunks and rank ``r`` owns
chunks ``[r, 2*cp_size - 1 - r]`` (this is exactly what
``megatron.core.utils.get_batch_on_this_cp_rank`` produces and what
``TEDotProductAttention`` assumes when it slices a ``post_scale_bias``).

This module provides:
  - ``zigzag_slice``: deterministically pick this CP rank's zigzag slice of a
    full-sequence tensor (no communication). Used to shard RoPE cos/sin and the
    Llama-4 scale, which are recomputed identically on every rank.
  - ``all_gather_seq_cp``: autograd all-gather that reconstructs the full
    sequence (undoing the zigzag) so the diffusion loss can split ``[xt | x0]``.
"""

import torch


def zigzag_slice(tensor: torch.Tensor, cp_rank: int, cp_size: int, seq_dim: int) -> torch.Tensor:
    """Pick this CP rank's load-balanced zigzag slice along ``seq_dim``.

    Mirrors ``get_batch_on_this_cp_rank``: split into ``2*cp_size`` chunks and
    take chunks ``cp_rank`` and ``2*cp_size - 1 - cp_rank``.

    Args:
        tensor: Full-sequence tensor.
        cp_rank: This rank's index within the CP group.
        cp_size: CP world size.
        seq_dim: Dimension to slice.

    Returns:
        Local slice of length ``tensor.shape[seq_dim] / cp_size``.
    """
    if cp_size == 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    assert seq_len % (2 * cp_size) == 0, f"seq_len {seq_len} not divisible by 2*cp_size {2 * cp_size}"
    shard = seq_len // (2 * cp_size)
    idx1, idx2 = cp_rank, 2 * cp_size - 1 - cp_rank
    s1 = [slice(None)] * tensor.dim()
    s1[seq_dim] = slice(idx1 * shard, (idx1 + 1) * shard)
    s2 = [slice(None)] * tensor.dim()
    s2[seq_dim] = slice(idx2 * shard, (idx2 + 1) * shard)
    return torch.cat([tensor[tuple(s1)], tensor[tuple(s2)]], dim=seq_dim).contiguous()


class _AllGatherSeqCP(torch.autograd.Function):
    """All-gather a CP-zigzag-sharded tensor to the full sequence on every rank.

    Forward reconstructs the global order; the result is identical on all CP
    ranks (replicated downstream). Backward picks this rank's zigzag chunks from
    the (identical) full gradient -- no all-reduce, which would over-count by
    ``cp_size``.
    """

    @staticmethod
    def forward(ctx, tensor, cp_group, seq_dim):
        cp_size = torch.distributed.get_world_size(cp_group)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        if cp_size == 1:
            return tensor.contiguous()

        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, tensor, group=cp_group)

        # Each rank contributed two zigzag chunks; restore global chunk order.
        chunks = []
        for shard in gathered:
            chunks.extend(torch.chunk(shard, chunks=2, dim=seq_dim))
        indices = []
        for r in range(cp_size):
            indices.append(r)
            indices.append(2 * cp_size - 1 - r)
        ordered = [c for _, c in sorted(zip(indices, chunks), key=lambda t: t[0])]
        return torch.cat(ordered, dim=seq_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = ctx.cp_size
        seq_dim = ctx.seq_dim
        if cp_size == 1:
            return grad_output, None, None
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        chunk = grad_output.shape[seq_dim] // (2 * cp_size)
        lo = cp_rank * chunk
        hi = (2 * cp_size - 1 - cp_rank) * chunk
        grad_input = torch.cat(
            [grad_output.narrow(seq_dim, lo, chunk), grad_output.narrow(seq_dim, hi, chunk)],
            dim=seq_dim,
        )
        return grad_input, None, None


def all_gather_seq_cp(tensor: torch.Tensor, cp_group, seq_dim: int = 1) -> torch.Tensor:
    """Autograd all-gather of a CP-sharded sequence tensor to full length."""
    return _AllGatherSeqCP.apply(tensor, cp_group, seq_dim)


def _reorder_zigzag_chunks(gathered, cp_size, seq_dim):
    """Reassemble global sequence order from per-rank zigzag shards."""
    chunks = []
    for shard in gathered:
        chunks.extend(torch.chunk(shard, chunks=2, dim=seq_dim))
    indices = []
    for r in range(cp_size):
        indices.append(r)
        indices.append(2 * cp_size - 1 - r)
    ordered = [c for _, c in sorted(zip(indices, chunks), key=lambda t: t[0])]
    return torch.cat(ordered, dim=seq_dim).contiguous()


class _ScatterSeqCP(torch.autograd.Function):
    """Inverse of the gather: take the full sequence (identical on every CP rank)
    and keep this rank's zigzag slice. Used to scatter the attention output back
    after a redundant full-sequence flex_attention.

    Forward = zigzag_slice. Backward = all-gather + reorder (each rank contributes
    the grad for the positions it owns; assembled into the full-sequence grad).
    """

    @staticmethod
    def forward(ctx, tensor, cp_group, seq_dim):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        if cp_size == 1:
            return tensor.contiguous()
        return zigzag_slice(tensor, cp_rank, cp_size, seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = ctx.cp_size
        seq_dim = ctx.seq_dim
        if cp_size == 1:
            return grad_output, None, None
        grad_output = grad_output.contiguous()
        gathered = [torch.empty_like(grad_output) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, grad_output, group=ctx.cp_group)
        return _reorder_zigzag_chunks(gathered, cp_size, seq_dim), None, None


def scatter_seq_cp(tensor: torch.Tensor, cp_group, seq_dim: int = 1) -> torch.Tensor:
    """Autograd scatter of a full-sequence tensor to this CP rank's zigzag slice."""
    return _ScatterSeqCP.apply(tensor, cp_group, seq_dim)


def local_zigzag_mask(seq_len: int, cp_rank: int, cp_size: int, device) -> torch.Tensor:
    """Boolean ``[seq_len]`` mask of the positions this CP rank owns.

    True at the two zigzag chunks (``cp_rank`` and ``2*cp_size-1-cp_rank``). Used
    to restrict the (full, gathered) loss to this rank's share so that summing
    loss/num_tokens across the CP group recovers the global totals exactly --
    i.e. standard Megatron context-parallel loss reduction stays valid. All-True
    when ``cp_size == 1``.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if cp_size == 1:
        mask[:] = True
        return mask
    assert seq_len % (2 * cp_size) == 0, f"seq_len {seq_len} not divisible by 2*cp_size {2 * cp_size}"
    cs = seq_len // (2 * cp_size)
    for idx in (cp_rank, 2 * cp_size - 1 - cp_rank):
        mask[idx * cs : (idx + 1) * cs] = True
    return mask
