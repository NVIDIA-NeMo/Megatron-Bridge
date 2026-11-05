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

"""Pure tensor helpers for the ERNIE 5.5 D layer.

Nothing here launches a fused kernel, so every helper runs on CPU. The helpers cover
the computation Megatron-Core lacks:

* Index geometry. For query row ``t`` of a document that starts at ``d(t)``, the
  indexer may select among the *remote* keys ``[d(t), t - W]`` (``W`` is
  ``csa_window_size``) and the row always attends to its *window*
  ``[max(d(t), t - W + 1), t]``. The two sets are disjoint and their union is every
  causal key of the document. Under context parallelism ``t`` and ``d(t)`` are global
  packed positions (the helpers take the local rows' ``positions``).
* The VHA post-mix ``out'_k = sum_h (I + V U^T)[k, h] out_h`` on the head axis.
* The sigmoid output gate ``out * sigmoid(gate_logits)``.
* The analytic gradient of a softmax sink that only enters the denominator.
* The FlashMLA key list of the reference implementation without its all-invalid key blocks.

All index tensors use ``-1`` for an invalid slot and int32 storage, the convention of
the Megatron-Core sparse-attention kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.dsa_layout import get_packed_qk_cu_seqlens
from torch import Tensor


__all__ = [
    "attention_sink_grad",
    "drop_invalid_key_blocks",
    "ernie55_document_starts",
    "ernie55_kernel_selection_ends",
    "ernie55_selection_bounds",
    "ernie55_window_indices",
    "lse_with_sink",
    "mask_topk_beyond_end",
    "pack_end",
    "pad_topk",
    "record_pack_end",
    "sigmoid_output_gate",
    "vha_postmix",
]


def _row_positions(num_rows: int, positions: Tensor | None, device: torch.device | str) -> Tensor:
    """int64 key coordinate of every query row: ``positions`` when given, else ``0 .. num_rows - 1``."""
    if positions is None:
        return torch.arange(num_rows, dtype=torch.int64, device=device)
    if positions.ndim != 1 or positions.numel() != num_rows:
        raise ValueError(f"positions must be 1-D with {num_rows} entries, got shape {tuple(positions.shape)}.")
    return positions.to(device=device, dtype=torch.int64)


def ernie55_document_starts(
    packed_seq_params: PackedSeqParams | None,
    seqlen: int,
    device: torch.device | str,
    *,
    positions: Tensor | None = None,
) -> Tensor:
    """Return the first row of the document that owns each query row.

    Packed THD inputs take their document boundaries from
    :func:`~megatron.core.transformer.experimental_attention_variant.dsa_layout.get_packed_qk_cu_seqlens`,
    which prefers the padded ``cu_seqlens`` (the same boundaries the D-layer RoPE uses).
    Rows at or past ``cu_seqlens[-1]`` (tail padding) belong to the last document. The
    reference implementation assumes packs without such rows, so they are outside parity
    checks. Without THD packing (``packed_seq_params`` is ``None`` or not ``"thd"``) every
    sample is a single document, so every start is 0.

    Under context parallelism a rank holds only some rows of the pack; ``positions`` then
    gives the global packed position of each local row, and the starts are global packed
    positions too.

    Args:
        packed_seq_params: Packed-sequence metadata, or ``None`` for SBHD inputs.
        seqlen: Number of query rows ``s`` (``T`` for a THD pack, the local row count under
            context parallelism).
        device: Device of the returned tensor.
        positions: Optional int64 ``[s]`` global packed position of every row; ``None`` means
            row ``t`` is at position ``t``.

    Returns:
        int64 tensor ``[s]`` with the document start of every row.
    """
    rows = _row_positions(seqlen, positions, device)
    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        return torch.zeros_like(rows)
    cu_seqlens, _ = get_packed_qk_cu_seqlens(packed_seq_params)
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int64)
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be 1-D with at least two entries, got shape {tuple(cu_seqlens.shape)}.")
    # Document d owns rows [cu[d], cu[d + 1]); right=True skips empty documents.
    doc_ids = torch.searchsorted(cu_seqlens[1:], rows, right=True).clamp_max(cu_seqlens.numel() - 2)
    return cu_seqlens[doc_ids]


# Attribute under which a micro-batch's PackedSeqParams holds its boundary tensor and that tensor's last entry.
_PACK_END_ATTR = "_ernie55_pack_end"


def record_pack_end(packed_seq_params: PackedSeqParams, end: int) -> None:
    """Note on ``packed_seq_params`` the host value of its last boundary, so no layer has to read it from the device.

    The batch builder knows it when it builds the ``cu_seqlens``. :func:`pack_end` returns it while the metadata
    still holds the same boundary tensor.

    Args:
        packed_seq_params: Packed THD metadata.
        end: Its last boundary (``cu_seqlens[-1]`` of the padded boundaries).
    """
    cu_seqlens, _ = get_packed_qk_cu_seqlens(packed_seq_params)
    setattr(packed_seq_params, _PACK_END_ATTR, (cu_seqlens, int(end)))


def pack_end(packed_seq_params: PackedSeqParams) -> int:
    """The last boundary of a pack's (padded) ``cu_seqlens`` on the host.

    It is the value :func:`record_pack_end` noted for the current boundary tensor; otherwise it is read from the
    device (one synchronization) and noted the same way, so each micro-batch reads it at most once.

    Args:
        packed_seq_params: Packed THD metadata.

    Returns:
        ``cu_seqlens[-1]`` as a Python int.
    """
    cu_seqlens, _ = get_packed_qk_cu_seqlens(packed_seq_params)
    noted = getattr(packed_seq_params, _PACK_END_ATTR, None)
    if noted is None or noted[0] is not cu_seqlens:
        noted = (cu_seqlens, int(cu_seqlens[-1]))
        setattr(packed_seq_params, _PACK_END_ATTR, noted)
    return noted[1]


def ernie55_selection_bounds(
    doc_start: Tensor, window_size: int, *, positions: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Return the indexer's remote-candidate bounds ``[start, end)`` for every row.

    Row ``t`` may select keys ``s`` with ``doc_start[t] <= s <= t - window_size``, i.e.
    ``start = doc_start`` and ``end = max(doc_start, t - window_size + 1)``. The first
    ``window_size`` rows of every document have an empty range (``start == end``); see
    :func:`ernie55_kernel_selection_ends` before handing these bounds to a kernel.

    With ``positions`` (context parallelism), ``t`` is the row's global packed position and
    the bounds index the globally ordered keys.

    Args:
        doc_start: int64 ``[s]`` from :func:`ernie55_document_starts`.
        window_size: Sliding-window width ``W`` (``csa_window_size``), at least 1.
        positions: Optional int64 ``[s]`` global packed position of every row; ``None`` means
            row ``t`` is at position ``t``.

    Returns:
        ``(starts, ends)``, both int32 ``[s]``, end exclusive.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}.")
    rows = _row_positions(doc_start.numel(), positions, doc_start.device)
    doc_start = doc_start.to(torch.int64)
    ends = torch.maximum(doc_start, rows - window_size + 1)
    return doc_start.to(torch.int32), ends.to(torch.int32)


def ernie55_kernel_selection_ends(*, starts: Tensor, ends: Tensor) -> Tensor:
    """Widen empty candidate ranges to one key so no kernel ever sees an empty row.

    Megatron-Core's top-k and indexer-loss kernels have only been exercised with causal
    bounds, where every row contains at least itself. An ERNIE row without remote
    candidates instead gets the single placeholder candidate ``start`` (its document
    start, which the window already covers). A one-candidate row has a KL of exactly 0 and
    no indexer gradient, so the loss is unchanged; :func:`mask_topk_beyond_end` removes
    the placeholder from the attention index list afterwards.

    Args:
        starts: int32 ``[s]`` candidate starts.
        ends: int32 ``[s]`` true candidate ends (exclusive).

    Returns:
        int32 ``[s]`` kernel ends, ``max(ends, starts + 1)``.
    """
    return torch.maximum(ends, starts + 1).to(torch.int32)


def mask_topk_beyond_end(topk: Tensor, *, ends: Tensor) -> Tensor:
    """Invalidate selected indices at or past the true candidate end.

    Only the placeholder added by :func:`ernie55_kernel_selection_ends` can lie past the
    true end, so this restores the exact ERNIE candidate set ``[start, end)``.

    Args:
        topk: ``[b, s, K]`` selected key indices, ``-1`` for an empty slot.
        ends: ``[s]`` true candidate ends (exclusive), shared by every sample.

    Returns:
        int32 ``[b, s, K]`` with every index ``>= ends[t]`` replaced by ``-1``.
    """
    if topk.ndim != 3 or ends.ndim != 1 or topk.size(1) != ends.numel():
        raise ValueError(f"Expected topk [b, s, K] and ends [s], got {tuple(topk.shape)} and {tuple(ends.shape)}.")
    beyond_end = topk >= ends.view(1, -1, 1).to(device=topk.device, dtype=topk.dtype)
    return topk.masked_fill(beyond_end, -1).to(torch.int32)


def ernie55_window_indices(
    doc_start: Tensor, *, window_size: int, batch_size: int, positions: Tensor | None = None
) -> Tensor:
    """Return each row's sliding-window keys ``[max(doc_start, t - W + 1), t]``.

    Every row's window contains the row itself, so no attention row is ever empty. Slot
    ``j`` holds ``max(doc_start[t], t - W + 1) + j``, or ``-1`` when that key lies after
    ``t``; valid keys therefore come first and in ascending order.

    With ``positions`` (context parallelism), ``t`` is the row's global packed position and
    the window indexes the globally ordered keys.

    Args:
        doc_start: int64 ``[s]`` from :func:`ernie55_document_starts`.
        window_size: Sliding-window width ``W``.
        batch_size: Number of samples ``b``; every sample shares the same geometry.
        positions: Optional int64 ``[s]`` global packed position of every row; ``None`` means
            row ``t`` is at position ``t``.

    Returns:
        int32 ``[b, s, W]`` per-sample key indices.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}.")
    rows = _row_positions(doc_start.numel(), positions, doc_start.device)
    first = torch.maximum(doc_start.to(torch.int64), rows - window_size + 1)
    window = first.unsqueeze(1) + torch.arange(window_size, dtype=torch.int64, device=doc_start.device)
    window = window.masked_fill(window > rows.unsqueeze(1), -1).to(torch.int32)
    return window.unsqueeze(0).expand(batch_size, -1, -1)


def pad_topk(topk: Tensor, width: int) -> Tensor:
    """Right-pad selected indices with ``-1`` to a fixed width.

    PyTorch's ``topk`` returns ``min(K, num_keys)`` columns; the attention index list
    ``[top-k | window]`` needs exactly ``K`` columns before the window.

    Args:
        topk: ``[b, s, k]`` indices with ``k <= width``.
        width: Target width ``K``.

    Returns:
        int32 ``[b, s, width]``.
    """
    current = topk.size(-1)
    if current > width:
        raise ValueError(f"Cannot pad {current} top-k columns down to {width}.")
    if current < width:
        topk = F.pad(topk, (0, width - current), value=-1)
    return topk.to(torch.int32)


def vha_postmix(out: Tensor, *, u: Tensor, v: Tensor, bf16_mix: bool) -> Tensor:
    """Mix attention heads with the low-rank-plus-identity matrix ``M = I + V U^T``.

    ``out'[..., k, :] = sum_h M[k, h] * out[..., h, :]``, so ``V`` indexes the output head
    and ``U`` the input head. ``V = 0`` gives the identity.

    Args:
        out: ``[..., H, D]`` per-head attention output; the head axis is ``-2``.
        u: ``[H, R]`` input-head factor (an fp32 parameter in the model).
        v: ``[H, R]`` output-head factor (an fp32 parameter in the model).
        bf16_mix: When True, ``M`` is formed by one GEMM in the activation dtype
            (``addmm(I, V, U^T)``) and applied by a second GEMM in that dtype, which is the
            bf16 contract of the reference implementation. When False, ``M - I = V U^T``
            is formed and the mix is accumulated in fp32 (or ``out.dtype`` if wider), and
            the result is rounded to ``out.dtype`` once.

    Returns:
        Tensor with the shape and dtype of ``out``.
    """
    num_heads = out.size(-2)
    if u.shape != v.shape or u.ndim != 2 or u.size(0) != num_heads:
        raise ValueError(f"Expected u and v of shape [{num_heads}, R], got {tuple(u.shape)} and {tuple(v.shape)}.")
    if bf16_mix:
        identity = torch.eye(num_heads, dtype=out.dtype, device=out.device)
        mix = torch.addmm(identity, v.to(out.dtype), u.to(out.dtype).t())
        return torch.matmul(mix, out)
    compute_dtype = torch.promote_types(out.dtype, torch.float32)
    delta = v.to(compute_dtype) @ u.to(compute_dtype).t()
    out_wide = out.to(compute_dtype)
    return (out_wide + torch.einsum("kh,...hd->...kd", delta, out_wide)).to(out.dtype)


def sigmoid_output_gate(out: Tensor, *, gate_logits: Tensor, fp32: bool) -> Tensor:
    """Return ``out * sigmoid(gate_logits)`` element-wise.

    Args:
        out: Attention output, flattened to ``[..., H * D]`` with index ``h * D + d``.
        gate_logits: Gate projection of the same shape and layout.
        fp32: When True, the sigmoid and the product run in fp32 (or ``out.dtype`` if wider)
            and the result is rounded to ``out.dtype`` once; otherwise both run in the
            activation dtype.

    Returns:
        Tensor with the shape and dtype of ``out``.
    """
    if out.shape != gate_logits.shape:
        raise ValueError(f"Gate shape {tuple(gate_logits.shape)} does not match output shape {tuple(out.shape)}.")
    if fp32:
        compute_dtype = torch.promote_types(out.dtype, torch.float32)
        return (out.to(compute_dtype) * torch.sigmoid(gate_logits.to(compute_dtype))).to(out.dtype)
    return out * torch.sigmoid(gate_logits.to(out.dtype))


def lse_with_sink(lse: Tensor, *, attn_sink: Tensor) -> Tensor:
    """Add a per-head sink to a log-sum-exp that excludes it.

    Args:
        lse: ``[rows, H]`` natural-log LSE over the selected keys only (FlashMLA's ``lse``).
        attn_sink: ``[H]`` sink logits.

    Returns:
        ``[rows, H]`` ``logaddexp(lse, sink)`` in fp32 (or ``lse.dtype`` if wider).
    """
    compute_dtype = torch.promote_types(lse.dtype, torch.float32)
    return torch.logaddexp(lse.to(compute_dtype), attn_sink.to(compute_dtype).view(1, -1))


def attention_sink_grad(*, attn_sink: Tensor, lse: Tensor, out: Tensor, d_out: Tensor) -> Tensor:
    """Analytic gradient of a softmax sink that enters only the denominator.

    With ``o = sum_j e^{s_j} v_j / (sum_j e^{s_j} + e^{sigma})`` per row and head,
    ``dL/dsigma = -e^{sigma - lse_full} * <o, dL/do>``, where ``lse_full`` includes the
    sink. Summing over rows gives the per-head gradient.

    Args:
        attn_sink: ``[H]`` sink logits ``sigma``.
        lse: ``[rows, H]`` LSE over the selected keys, *excluding* the sink.
        out: ``[rows, H, d_v]`` attention output.
        d_out: ``[rows, H, d_v]`` gradient of the loss with respect to ``out``.

    Returns:
        ``[H]`` gradient with respect to ``attn_sink``, in fp32 (or ``lse.dtype`` if wider).
    """
    lse_full = lse_with_sink(lse, attn_sink=attn_sink)
    sink_prob = torch.exp(attn_sink.to(lse_full.dtype).view(1, -1) - lse_full)
    delta = (out.to(lse_full.dtype) * d_out.to(lse_full.dtype)).sum(dim=-1)
    return -(sink_prob * delta).sum(dim=0)


def drop_invalid_key_blocks(flat_idxs: Tensor, *, block: int) -> tuple[Tensor, Tensor]:
    """Remove the key blocks that hold no valid index from FlashMLA key lists, keeping every other block as is.

    FlashMLA's sparse forward reads a row's list in blocks of keys (64 per block on SM100 at 64 heads, 128 at 128
    heads) and folds each block into its online softmax as a unit; a block of ``-1`` entries is an exact no-op.
    Which keys share a block, and the order of the blocks, decide the output bits. This keeps the content and the
    order of every block that holds a valid key, moves the blocks without one to the end and returns the kept
    length, so FlashMLA (with ``topk_length``) computes exactly what it computes for the full list, while reading
    only the kept blocks. ``-1`` entries inside a kept block stay where they are.

    The reference implementation hands FlashMLA ``[top-k | window]`` rows of a fixed width, the top-k in score
    order followed by ``-1`` padding and then the window: most of its blocks are empty for short documents.
    Applied to such rows with ``block`` 64 or 128, the result reproduces its FlashMLA output bitwise.

    Args:
        flat_idxs: ``[rows, width]`` flat key indices, ``-1`` for an invalid slot.
        block: Block granularity; a multiple of FlashMLA's key block (on SM100, 64 at 64 heads and 128 at
            128 heads; 128 on SM90).

    Returns:
        ``(indices, topk_length)``: int32 ``[rows, ceil(width / block) * block]`` (padded with ``-1``) and int32
        ``[rows]`` kept lengths, a multiple of ``block`` (0 for a row without valid keys).
    """
    if flat_idxs.ndim != 2:
        raise ValueError(f"flat_idxs must be [rows, width], got shape {tuple(flat_idxs.shape)}.")
    if block < 1:
        raise ValueError(f"block must be positive, got {block}.")
    rows, width = flat_idxs.shape
    padded = F.pad(flat_idxs, (0, -width % block), value=-1).to(torch.int32)
    blocks = padded.view(rows, -1, block)
    keep = (blocks >= 0).any(dim=-1)
    # Kept blocks first, each group in its original order.
    order = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True)
    kept = torch.gather(blocks, 1, order.unsqueeze(-1).expand(-1, -1, block)).reshape(rows, -1)
    topk_length = (keep.sum(dim=-1) * block).to(torch.int32)
    return kept.contiguous(), topk_length.contiguous()
