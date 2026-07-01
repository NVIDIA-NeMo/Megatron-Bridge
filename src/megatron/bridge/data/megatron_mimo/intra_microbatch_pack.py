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

"""Sequence-packing assembly for intra-microbatch reordering.

Concatenates the real (unpadded) tokens of a cost-balanced group of examples into a single
packed sequence — the Transformer Engine ``thd`` layout (total-tokens × heads × head-dim) —
with ``cu_seqlens`` boundaries. Downstream, the ``cu_seqlens`` make attention block-diagonal
per example (the qwen3-vl attention / flash-attn path already honors this), so packing does
not leak attention across examples.

The cost-balancing happens earlier, at the intra-microbatch reordering exchange
(:func:`megatron.bridge.data.megatron_mimo.reorder_buffer.exchange_window`): each DP shard
already holds a cost-balanced subset. This module then packs the **whole shard** as one
group — there is no per-pack length budget here, so the assembled pack is the entire shard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def assemble_packed_sequence(
    group: List[int],
    tokens: Optional[torch.Tensor],
    lengths: List[int],
    *,
    pad_token_id: int = 0,
    labels: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Concatenate one group's real tokens into a single packed sequence (THD layout).

    ``tokens`` may be ``None`` on a non-first PP stage that has no ``input_ids`` but still must
    pack ``labels`` / ``loss_mask`` to the same ``[1, T]`` shape (F5, PP-consistent packing); in
    that case ``input_ids`` is ``None`` in the result and the device is taken from the first
    available tensor.

    Args:
        group: Row indices (into the source tensors) of the examples in this pack, in pack order.
        tokens: Padded ``[B, S]`` token ids, or ``None`` (last-stage label-only pack).
        lengths: Real (unpadded) length per row.
        pad_token_id: Fill value for the ``input_ids`` buffer. The pack is tight (segments
            are concatenated with no inter-segment padding and ``T`` equals their summed
            length), so every position is overwritten and this value never appears in the
            output; it is kept for parity with the unpacked path.
        labels: Optional padded ``[B, S]`` labels (pad value ``-100``).
        loss_mask: Optional padded ``[B, S]`` loss mask (pad value ``0``).
        position_ids: Optional padded position ids. Both 2-D ``[B, S]`` and Qwen-VL MRoPE
            3-D ``[3, B, S]`` are handled; each sample's real slice is concatenated, so the
            packed positions are per-sample-reset (each sample's positions already start at 0).

    Returns:
        Dict with ``input_ids`` / ``labels`` / ``loss_mask`` as packed ``[1, T]`` tensors
        (T = sum of the group's real lengths; ``input_ids`` is ``None`` when ``tokens`` is
        ``None``), ``position_ids`` as ``[1, T]`` (2-D input) or ``[3, 1, T]`` (MRoPE input),
        ``cu_seqlens`` (``int32``, length ``len(group)+1``), and ``max_seqlen`` (int).

    Raises:
        ValueError: If ``position_ids`` is neither 2-D ``[B, S]`` nor 3-D MRoPE ``[3, B, S]``;
            if none of ``tokens`` / ``labels`` / ``loss_mask`` / ``position_ids`` is a tensor
            (no device or shape to pack to); or if the group's segment lengths sum to zero
            (an empty packed shard cannot be built).
    """
    if position_ids is not None and position_ids.dim() not in (2, 3):
        raise ValueError(f"position_ids must be 2-D [B, S] or 3-D MRoPE [3, B, S] (got {position_ids.dim()}-D).")
    if position_ids is not None and position_ids.dim() == 3 and position_ids.size(0) != 3:
        raise ValueError(f"3-D position_ids must be MRoPE [3, B, S] (got leading dim {position_ids.size(0)}).")

    ref = next((t for t in (tokens, labels, loss_mask, position_ids) if isinstance(t, torch.Tensor)), None)
    if ref is None:
        raise ValueError("assemble_packed_sequence needs at least one of tokens/labels/loss_mask/position_ids.")
    device = ref.device
    seglens = [int(lengths[i]) for i in group]
    # Build the cumulative offsets in pure Python (seglens is already host-side), then move the
    # cu_seqlens tensor to device in one copy — avoids per-element GPU scatter writes and the
    # blocking ``.item()`` D2H sync this runs on the per-step pack path.
    cu = [0]
    for seg in seglens:
        cu.append(cu[-1] + seg)
    total = cu[-1]
    if total <= 0:
        raise ValueError("assemble_packed_sequence cannot build an empty packed shard; all segment lengths are zero.")
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)

    def _concat(src: Optional[torch.Tensor], pad: int) -> Optional[torch.Tensor]:
        if src is None:
            return None
        out = torch.full((1, total), pad, dtype=src.dtype, device=device)
        offset = 0
        for i, seg in zip(group, seglens):
            out[0, offset : offset + seg] = src[i, :seg]
            offset += seg
        return out

    def _concat_mrope(src: torch.Tensor) -> torch.Tensor:
        # MRoPE [3, B, S] -> [3, 1, T]: concat each sample's real [3, seg] slice along T.
        out = torch.zeros((3, 1, total), dtype=src.dtype, device=device)
        offset = 0
        for i, seg in zip(group, seglens):
            out[:, 0, offset : offset + seg] = src[:, i, :seg]
            offset += seg
        return out

    if position_ids is None:
        packed_position_ids: Optional[torch.Tensor] = None
    elif position_ids.dim() == 3:
        packed_position_ids = _concat_mrope(position_ids)
    else:
        packed_position_ids = _concat(position_ids, 0)

    return {
        "input_ids": _concat(tokens, pad_token_id),
        "labels": _concat(labels, -100),
        "loss_mask": _concat(loss_mask, 0),
        "position_ids": packed_position_ids,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(seglens) if seglens else 0,
    }


def pack_language_shard(
    data_batch: Dict[str, Any],
    *,
    pad_token_id: int = 0,
    lengths: torch.Tensor | None = None,
) -> "tuple[Dict[str, Any], Dict[str, Any]] | tuple[Dict[str, Any], None]":
    """Pack a per-DP-shard language batch ``[bs, S]`` into a single packed sequence ``[1, T]``.

    Concatenates the shard's real (unpadded) tokens — including image-placeholder tokens, so
    the downstream MIMO modality splice still fills them in order — into one packed sequence with
    ``cu_seqlens`` block-diagonal boundaries. This removes the per-sample padding compute (the
    LM no longer processes ``S``-padded sequences) and makes attention cost ∝ real tokens, the
    lever for the per-step DP straggler that intra-microbatch reordering balances.

    The returned ``packing_kwargs`` feeds ``MimoModel.forward(packing_kwargs=...)``, which
    builds a THD ``PackedSeqParams`` (``qkv_format='thd'``) and threads ``cu_seqlens`` to the
    GPT decoder for block-diagonal attention. Non-LM tensors (``modality_inputs``, etc.) are
    carried through unchanged.

    The per-sample real length comes **only** from the caller-supplied ``lengths`` (the
    PP-consistent source). ``forward_step`` derives ``lengths`` from ``input_ids`` on every
    language stage — ``input_ids`` is the only tensor that counts image-placeholder tokens —
    so all stages pack to an identical ``[1, T]`` and the decoder ``cu_seqlens`` match the
    hidden states propagated down the pipeline. (Under PP>1 the example's ``_batch_spec_for_rank``
    keeps ``input_ids`` on every language stage when packing is active, so ``lengths`` is derived
    from it before it is nulled; ``position_ids`` is then packed to the same ``[1, T]`` — neither
    ``attention_mask`` nor ``position_ids`` is used as a length source because they under-count
    image-placeholder tokens.) When ``lengths`` is ``None`` there is no supported length source, so
    the batch is returned unchanged.

    ``loss_mask`` is **not** used for length (it is a supervision mask, not a padding mask).

    Args:
        data_batch: This rank's sliced language batch with ``input_ids`` ``[bs, S]`` and/or
            ``labels`` / ``loss_mask`` ``[bs, S]``, plus optional ``position_ids`` (``[bs, S]``
            or MRoPE ``[3, bs, S]``).
        pad_token_id: Fill value for the packed ``input_ids`` buffer (overwritten by the tight
            pack; kept for parity with the unpacked path).
        lengths: Per-sample real token lengths ``[bs]`` from the caller. When ``None`` the batch
            is returned unchanged (no supported length source).

    Returns:
        ``(packed_batch, packing_kwargs)``. ``packing_kwargs`` is ``None`` (and the batch is
        returned unchanged) when ``lengths`` is ``None``, so the caller can pass it straight
        through on non-data / single-token stages.

    Raises:
        ValueError: If ``lengths`` cannot be derived to a consistent ``[bs]`` shape (a
            mis-shaped pack would otherwise be silent).
    """
    input_ids = data_batch.get("input_ids")
    labels = data_batch.get("labels")
    loss_mask = data_batch.get("loss_mask")
    position_ids = data_batch.get("position_ids")

    # The caller-provided ``lengths`` (derived from ``input_ids`` in forward_step) is the only
    # supported length source; without it there is nothing to pack.
    if lengths is None:
        return data_batch, None
    lengths_t = torch.as_tensor(lengths, dtype=torch.long)
    ref = next((t for t in (input_ids, labels, loss_mask) if isinstance(t, torch.Tensor)), None)
    if ref is not None:
        bs = ref.size(0)
    elif isinstance(position_ids, torch.Tensor) and position_ids.dim() in (2, 3):
        bs = position_ids.size(1 if position_ids.dim() == 3 else 0)
    else:
        bs = int(lengths_t.numel())

    if lengths_t.dim() != 1 or lengths_t.size(0) != bs:
        raise ValueError(
            f"pack_language_shard derived inconsistent per-sample lengths (shape {tuple(lengths_t.shape)}, "
            f"expected [{bs}]); refusing to produce a mis-shaped packed sequence."
        )
    lengths = lengths_t.tolist()
    group = list(range(bs))
    pack = assemble_packed_sequence(
        group,
        input_ids if isinstance(input_ids, torch.Tensor) else None,
        lengths,
        pad_token_id=pad_token_id,
        labels=labels,
        loss_mask=loss_mask,
        position_ids=position_ids,
    )

    packed_batch = dict(data_batch)
    if isinstance(input_ids, torch.Tensor):
        packed_batch["input_ids"] = pack["input_ids"]
    packed_batch["labels"] = pack["labels"]
    packed_batch["loss_mask"] = pack["loss_mask"]
    packed_batch["position_ids"] = pack["position_ids"]
    packed_batch["attention_mask"] = None  # block-diagonal attention comes from cu_seqlens

    cu = pack["cu_seqlens"]
    max_seqlen = int(pack["max_seqlen"])
    # The packed sequence is tight (segments concatenated with no inter-segment padding), so the
    # padded cu_seqlens coincide with the unpadded ones. They must still be populated because
    # the MIMO CP>1 THD partition path dereferences cu_seqlens_q_padded / cu_seqlens_kv_padded.
    # validation-pending (multi-node): the live CP>1 partition is verified on the 16xH100 rig.
    packing_kwargs = {
        "cu_seqlens_q": cu,
        "cu_seqlens_kv": cu,
        "cu_seqlens_q_padded": cu,
        "cu_seqlens_kv_padded": cu,
        "max_seqlen_q": max_seqlen,
        "max_seqlen_kv": max_seqlen,
    }
    return packed_batch, packing_kwargs
