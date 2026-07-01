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

"""Per-sample reorder buffer for MegatronMIMO intra-microbatch reordering (post-batch).

Each MegatronMIMO data rank reads only its **disjoint** scalable-data-parallel shard of the global
micro-batch, and the per-microbatch cost imbalance between shards is corrected by a
**communication** step (cost all-gather + ragged sample all-to-all over the module DP
group) that can be overlapped with compute — instead of every rank reading the full
global micro-batch and slicing it locally.

Contents, bottom-up:

1. Pure, communication-free building blocks (all CPU-unit-testable without a process
   group):

   - :func:`balanced_assignment` — the per-sample plan: for each global sample in a
     micro-batch, the module-local DP rank that should own it and its canonical position.
     Built on :func:`megatron.bridge.data.datasets.packing_utils.balanced_index_order`,
     with het-DP handled via canonical ``n_groups = max(module dp_sizes)`` (so the vision
     and language modules, which may have different DP sizes, derive an identical
     assignment and the ``BridgeCommunicator`` keeps vision replica *r* paired with
     language replica *r*).
   - Ragged (de)serialization of a sample's tensors to an 8-byte-aligned ``uint8`` buffer
     (:func:`serialize_sample` / :func:`deserialize_sample`), so any dtype/shape mix can
     ride a single ``all_to_all_single`` with per-rank byte splits; the receiver
     reconstructs from metadata alone.
   - :func:`prepare_sample_exchange` — from this rank's local samples plus the all-gathered
     plan and metadata, partition kept-vs-exchanged samples, pack the contiguous send buffer,
     and build the ``send_splits``/``recv_splits`` + ordered recv schedule for one
     ``all_to_all_single``.

2. The MIMO data plane: :func:`split_microbatch` / :func:`merge_samples` convert a nested
   micro-batch to and from ``B`` per-sample flat (dotted-key) dicts, and :func:`sample_cost`
   is the (collation-independent) per-sample cost — ``linear_vit·patches + linear_lm·real_tokens`` —
   that drives the balanced assignment.

3. The distributed glue: :func:`exchange_window` ties the above into one cost-balancing round
   over a window of micro-batches on a module DP group, :class:`ReorderingBuffer` wraps a data
   iterator to run that round per window (optionally on a background prefetch thread with a dedicated CUDA
   stream + side NCCL PG, so step ``t+1``'s exchange overlaps step ``t``'s compute), and
   :func:`build_module_dp_process_groups` creates the Gloo (metadata) and side NCCL
   (exchange) process groups the buffer needs.

The unit of exchange is a **single sample** (finer than a whole-microbatch swap), and the
assignment reuses the existing MIMO cost-balanced ordering rather than a separate
group-then-LPT pass.
"""

import logging
import math
import os
import queue
import threading
from typing import Any, Dict, List, Tuple

import torch

from megatron.bridge.data.datasets.packing_utils import balanced_index_order
from megatron.bridge.data.megatron_mimo.dp_utils import (
    _batch_dim_for_tensor,
    is_vision_subdict,
    real_token_lengths,
)


logger = logging.getLogger(__name__)

# All sample tensors are serialized into uint8 byte buffers padded to this alignment, so
# the receiver can view-cast any dtype back from the received bytes without alignment
# errors and compute exact offsets from metadata alone.
_ALIGN = 8


def _pad_to_align(nbytes: int) -> int:
    """Zero-padding bytes to round ``nbytes`` up to the next :data:`_ALIGN` multiple."""
    return (-nbytes) % _ALIGN


# String -> (torch.dtype, byte width), for reconstructing tensors from metadata shared via
# all_gather_object. The itemsize is precomputed once so the (de)serialize byte-size loops do not
# allocate a throwaway tensor per field just to read ``element_size()``.
_DTYPE_INFO = {
    name: (dt, torch.empty((), dtype=dt).element_size())
    for name, dt in {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.bool": torch.bool,
        "torch.uint8": torch.uint8,
    }.items()
}


def _dtype_info(dtype_name: str) -> Tuple[torch.dtype, int]:
    """Return ``(dtype, byte width)`` for a dtype encoded in sample metadata, rejecting unknown values."""
    try:
        return _DTYPE_INFO[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype in MegatronMIMO sample metadata: {dtype_name!r}.") from exc


# ---------------------------------------------------------------------------
# Per-sample balanced assignment
# ---------------------------------------------------------------------------


def balanced_assignment(costs: List[float], n_groups: int, dp_size: int) -> List[Tuple[int, int]]:
    """Per-sample exchange plan for one micro-batch.

    Runs the canonical cost-balanced ordering over ``n_groups`` equal-cardinality groups,
    then maps each global sample to the **module-local DP rank** that should own it and to
    its **canonical position** in the globally balanced order.

    The owner mapping uses canonical groups (``n_groups = max(module dp_sizes)``): a module
    of size ``dp_size`` covers ``n_groups // dp_size`` contiguous canonical groups, so
    ``owner = canonical_group // (n_groups // dp_size)``. Because the vision and language
    modules feed the same ``costs`` and ``n_groups``, they produce an identical plan and
    paired replicas end up with the same samples in the same (canonical) order — required
    by the ``BridgeCommunicator`` batch-dim fan-out.

    Args:
        costs: Per-sample cost, indexed by the sample's global position in the micro-batch
            (the patch-only cost from :func:`sample_cost`).
        n_groups: Canonical group count (``max`` module DP size). Must divide ``len(costs)``.
        dp_size: This module's DP size. Must divide ``n_groups``.

    Returns:
        For each global sample index ``g``: ``(owner_rank, canonical_pos)`` where
        ``owner_rank`` is in ``[0, dp_size)`` and ``canonical_pos`` is in ``[0, len(costs))``.
        Sorting a rank's owned samples by ``canonical_pos`` yields the order paired modules
        agree on.

    Raises:
        ValueError: If ``n_groups`` does not divide ``len(costs)`` or ``dp_size`` does not
            divide ``n_groups``.
    """
    n = len(costs)
    if n_groups <= 0 or n % n_groups != 0:
        raise ValueError(f"n_groups ({n_groups}) must be positive and divide len(costs) ({n}).")
    if dp_size <= 0 or n_groups % dp_size != 0:
        raise ValueError(f"dp_size ({dp_size}) must be positive and divide n_groups ({n_groups}).")

    perm = balanced_index_order(costs, n_groups)  # global indices laid out group-by-group
    cap = n // n_groups
    groups_per_rank = n_groups // dp_size

    assignment: List[Tuple[int, int]] = [(0, 0)] * n
    for canonical_pos, global_idx in enumerate(perm):
        canonical_group = canonical_pos // cap
        owner_rank = canonical_group // groups_per_rank
        assignment[global_idx] = (owner_rank, canonical_pos)
    return assignment


def intra_route(assignment: List[Tuple[int, int]], *, src_slot: int = 0) -> List[Tuple[int, int, int]]:
    """Lift a single-slot 2D balanced assignment to the vehicle's 3D route.

    The inter-ready vehicle routes every sample by **three** coordinates
    ``route[g] = (dst_slot, owner_rank, canonical_pos)`` (the destination micro-batch slot, the
    module-local DP rank that owns it, and its canonical intra-slot position). The **intra**
    balancer never moves a sample across micro-batch slots, so ``dst_slot == src_slot`` for every
    sample; a future inter (cross-slot) balancer fills ``dst_slot`` freely behind this same seam.

    Args:
        assignment: Per-slot output of :func:`balanced_assignment` — ``(owner_rank, canonical_pos)``
            per global sample index within the slot.
        src_slot: The slot these samples were read from (and, for intra, stay in). Defaults to ``0``
            (the single-slot ``window_size == 1`` case).

    Returns:
        For each global sample index ``g``: ``(dst_slot, owner_rank, canonical_pos)`` with
        ``dst_slot == src_slot``.
    """
    return [(src_slot, owner_rank, canonical_pos) for owner_rank, canonical_pos in assignment]


def build_window_route(costs_per_slot: List[List[float]], n_groups: int, dp_size: int) -> List[Tuple[int, int, int]]:
    """Concatenate the per-slot intra assignments of a window into one 3D route.

    The window holds ``W = len(costs_per_slot)`` micro-batch slots, each with ``B_global`` samples
    contiguous-sharded across ``dp_size`` ranks. The **intra** balancer runs **independently per
    slot** (each slot cost-balanced across ranks); slots never mix. The window global index of slot
    ``s``'s slot-local sample ``j`` is ``g = s · B_global + j`` (slots laid out back-to-back), so
    concatenating each slot's :func:`intra_route` (stamped with ``src_slot = s``) in slot order
    yields ``route[g]`` directly. ``dst_slot == src_slot == s`` for every sample (intra invariant).

    A future inter (cross-slot) balancer replaces this builder with one that fills ``dst_slot``
    across slots; the transport (:func:`exchange_by_route`) is unchanged.

    Args:
        costs_per_slot: ``costs_per_slot[s]`` = slot ``s``'s per-sample costs indexed by slot-local
            global position ``j`` (length ``B_global``, identical on paired modules).
        n_groups: Canonical group count (``max`` module DP size); must divide each ``B_global``.
        dp_size: This module's DP size.

    Returns:
        ``route`` of length ``W · B_global`` — ``(dst_slot, owner_rank, canonical_pos)`` per window
        global index, slot-major.
    """
    route: List[Tuple[int, int, int]] = []
    for slot, costs in enumerate(costs_per_slot):
        route.extend(intra_route(balanced_assignment(costs, n_groups, dp_size), src_slot=slot))
    # §10.0 safety net: the intra balancer must never move a sample across slots. Cheap, always-on.
    assert_intra_no_cross_slot(route, len(costs_per_slot[0]) if costs_per_slot else 0)
    return route


def assert_intra_no_cross_slot(route: List[Tuple[int, int, int]], b_global: int) -> None:
    """§10.0 "no cross-slot leak" guard: every sample's ``dst_slot`` equals its source slot.

    For the slot-major window layout (``g = src_slot · b_global + j``) the source slot of global
    index ``g`` is ``g // b_global``. The **intra** balancer keeps every sample in its own slot, so
    a mismatch means slot plumbing (route build / binning) corrupted the slot dimension — the §12
    "primary safety net while W>1 plumbing is new". A future inter (cross-slot) balancer would route
    across slots by design and must **not** run this guard.

    Args:
        route: The window route (``(dst_slot, owner_rank, canonical_pos)`` per global index).
        b_global: Samples per micro-batch slot (``local · dp_size``); ``0`` skips the check.

    Raises:
        RuntimeError: If any sample's ``dst_slot`` differs from its source slot.
    """
    if b_global <= 0:
        return
    for g, (dst_slot, _owner_rank, _pos) in enumerate(route):
        src_slot = g // b_global
        if dst_slot != src_slot:
            raise RuntimeError(
                f"MegatronMIMO intra route cross-slot leak at global index {g}: "
                f"dst_slot={dst_slot} != src_slot={src_slot} (the intra balancer must keep dst_slot == src_slot)."
            )


def window_cost_spread(
    costs_per_slot: List[List[float]], route: List[Tuple[int, int, int]], dp_size: int
) -> List[Dict[str, float]]:
    """Per-slot pre/post-balance per-rank cost spread + remote-sample count (the §10.5 balance probe).

    For each slot, totals the per-rank cost **before** balancing (the natural contiguous shard: rank
    ``r`` holds slot-local positions ``[r·local, (r+1)·local)``) and **after** balancing (each sample
    credited to its routed ``owner_rank``), plus how many samples change rank. Pure and cheap — used
    only to log how much the reorder tightens the per-rank load (``after`` max/min should narrow
    toward 1.0), the quantitative balance evidence the design's §10.3 validation looks for.

    Args:
        costs_per_slot: ``costs_per_slot[s]`` = slot ``s``'s per-sample costs by slot-local position.
        route: The window route (:func:`build_window_route`), indexed by ``g = s·B_global + j``.
        dp_size: This module's DP size.

    Returns:
        One dict per slot: ``{before_max, before_min, after_max, after_min, remote}``.
    """
    spreads: List[Dict[str, float]] = []
    for slot, costs in enumerate(costs_per_slot):
        n = len(costs)
        local = n // dp_size
        before = [0.0] * dp_size
        after = [0.0] * dp_size
        remote = 0
        for j, c in enumerate(costs):
            src_rank = j // local
            before[src_rank] += c
            _dst_slot, owner_rank, _pos = route[slot * n + j]
            after[owner_rank] += c
            if owner_rank != src_rank:
                remote += 1
        spreads.append(
            {
                "before_max": max(before),
                "before_min": min(before),
                "after_max": max(after),
                "after_min": min(after),
                "remote": float(remote),
            }
        )
    return spreads


# ---------------------------------------------------------------------------
# Ragged sample (de)serialization
# ---------------------------------------------------------------------------


def sample_keys(meta: Dict[str, Any]) -> List[str]:
    """Sorted non-None tensor keys for a sample, computed identically on both sides."""
    return sorted(k for k, v in meta.items() if isinstance(v, dict) and "dtype" in v)


def tensor_metadata(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Per-field metadata for a flattened sample, keyed identically on both sides.

    Each key maps to ``{shape, dtype}`` for a tensor, ``{non_tensor: value}`` for a non-None
    scalar/global field, or ``None``. Shared across ranks via ``all_gather_object`` so a
    receiver can pre-allocate buffers and reconstruct the sample without seeing the data.

    A tensor whose dtype is not in :data:`_DTYPE_INFO` is rejected **here** — at metadata creation,
    before the all-gather — via the same :func:`_dtype_info` check the deserializer uses,
    so an unsupported dtype fails loudly and uniformly on the producing rank rather than only
    surfacing later as a byte-size/deserialize ``ValueError`` on the receiver.

    Raises:
        ValueError: If a tensor field has a dtype not supported by the ragged (de)serializer.
    """
    meta: Dict[str, Any] = {}
    for key, t in flat.items():
        if isinstance(t, torch.Tensor):
            dtype_name = str(t.dtype)
            _dtype_info(dtype_name)  # reject unsupported dtypes early (same check as decode)
            meta[key] = {"shape": list(t.shape), "dtype": dtype_name}
        elif t is not None:
            meta[key] = {"non_tensor": t}
        else:
            meta[key] = None
    return meta


def serialize_sample(flat: Dict[str, Any], keys: List[str]) -> torch.Tensor:
    """Serialize one sample's tensors into a 1D uint8 buffer (8-byte aligned, sorted keys).

    Args:
        flat: Flattened tensor dict; ``None`` values are skipped.
        keys: Sorted non-None tensor keys (from :func:`sample_keys`), same on both sides.

    Returns:
        1D uint8 CPU tensor with each tensor's raw bytes concatenated in ``keys`` order,
        each padded to :data:`_ALIGN`.
    """
    parts = []
    for key in keys:
        t = flat.get(key)
        if t is None:
            continue
        raw = t.contiguous().view(torch.uint8).reshape(-1)
        pad = _pad_to_align(raw.numel())
        if pad:
            raw = torch.cat([raw, torch.zeros(pad, dtype=torch.uint8, device=raw.device)])
        parts.append(raw)
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)


def sample_byte_size(meta: Dict[str, Any], keys: List[str]) -> int:
    """Byte size of a serialized sample from metadata alone (matches :func:`serialize_sample`)."""
    total = 0
    for key in keys:
        info = meta.get(key)
        if info is None:
            continue
        _, elem = _dtype_info(info["dtype"])
        nbytes = math.prod(info["shape"]) * elem
        nbytes += _pad_to_align(nbytes)
        total += nbytes
    return total


def deserialize_sample(
    buf: torch.Tensor, offset: int, meta: Dict[str, Any], keys: List[str]
) -> Tuple[Dict[str, Any], int]:
    """Inverse of :func:`serialize_sample`: reconstruct tensors from a uint8 buffer.

    Args:
        buf: Received 1D uint8 buffer.
        offset: Starting byte offset of this sample in ``buf``.
        meta: This sample's field metadata (from :func:`tensor_metadata`:
            ``{key: {shape, dtype} | {non_tensor: value} | None}``).
        keys: Sorted non-None tensor keys (from :func:`sample_keys`).

    Returns:
        ``(flat, new_offset)`` — the reconstructed tensor dict (tensors cloned off ``buf``,
        with ``None`` and non-tensor keys restored from ``meta``) and the byte offset just
        past this sample.
    """
    flat: Dict[str, Any] = {}
    cursor = offset
    for key in keys:
        info = meta.get(key)
        if info is None:
            flat[key] = None
            continue
        dtype, elem = _dtype_info(info["dtype"])
        nbytes = math.prod(info["shape"]) * elem
        flat[key] = buf[cursor : cursor + nbytes].view(dtype).reshape(info["shape"]).clone()
        cursor += nbytes + _pad_to_align(nbytes)
    for key, info in meta.items():
        if key not in flat:
            flat[key] = info.get("non_tensor") if isinstance(info, dict) and "non_tensor" in info else None
    return flat, cursor


# ---------------------------------------------------------------------------
# All-to-all plan builder (one micro-batch, one module DP group)
# ---------------------------------------------------------------------------


def prepare_sample_exchange(
    local_flats: List[Dict[str, Any]],
    local_global_indices: List[int],
    route: List[Tuple[int, int, int]],
    all_global_indices: List[List[int]],
    all_tensor_meta: List[List[Dict[str, Any]]],
    dp_rank: int,
    dp_size: int,
    *,
    window_size: int = 1,
) -> Dict[str, Any]:
    """Prepare everything one per-sample ``all_to_all_single`` needs, in three parts.

    Pure and communication-free: all cross-rank information (which rank currently holds which
    global sample, and each sample's tensor metadata) is passed in, having been shared by a
    prior ``all_gather_object``. This single pass produces three distinct things:

    1. **Partition** — split this rank's samples into those it already owns (kept locally) and
       those it must send/receive (``local_samples`` + the send/recv schedules).
    2. **Pack** — serialize the outgoing samples into one contiguous ``send_buf`` ordered by
       destination rank.
    3. **Plan** — the per-rank byte ``send_splits``/``recv_splits`` and the ordered
       ``recv_schedule`` that drive (and decode) the ``all_to_all_single``.

    Each sample carries its destination micro-batch slot (``dst_slot``) end-to-end so the receiver
    can bin it into the right slot bucket on reassembly (:func:`reassemble_window`); this is the
    only genuinely new plumbing vs. the single-slot path (§4a of the inter-ready design). Sender
    and receiver order the moves identically by ``(dst_slot, canonical_pos, src_local_idx)`` so the
    packed bytes line up.

    Args:
        local_flats: This rank's flattened samples (tensor dicts), in local order.
        local_global_indices: Global index of each entry in ``local_flats`` (same length).
        route: Output of :func:`intra_route` (or a future inter balancer) —
            ``(dst_slot, owner_rank, canonical_pos)`` per global sample index.
        all_global_indices: ``all_global_indices[r]`` = the global indices currently held by
            rank ``r`` (the all-gathered ``local_global_indices``).
        all_tensor_meta: ``all_tensor_meta[r][i]`` = tensor metadata of rank ``r``'s ``i``-th
            local sample (the all-gathered :func:`tensor_metadata`).
        dp_rank: This rank's module-local DP rank.
        dp_size: This module's DP size.
        window_size: Number of micro-batch slots ``W`` in the window; ``dst_slot`` must be in
            ``[0, W)``. Defaults to ``1`` (the single-slot, byte-for-byte-with-intra case).

    Returns:
        Dict with:
          - (partition) ``local_samples``: ``[(dst_slot, canonical_pos, local_idx)]`` kept on this
            rank.
          - (pack) ``send_buf``: 1D uint8 tensor on the same device as the sample tensors (GPU
            when CUDA is available), outgoing samples serialized contiguously by dst.
          - (plan) ``send_splits`` / ``recv_splits``: per-rank byte counts for
            ``all_to_all_single`` (``0`` for self).
          - (plan) ``recv_schedule``: ``{src_rank: [(dst_slot, canonical_pos, src_local_idx)]}`` in
            recv order, so the receiver deserializes ``recv_buf`` in the same order it was packed.
    """
    # Map global index -> (current owner rank, current local index) from the all-gather.
    location: Dict[int, Tuple[int, int]] = {}
    for r, gidxs in enumerate(all_global_indices):
        for i, g in enumerate(gidxs):
            location[g] = (r, i)

    # Samples this rank must end up with: every global index whose owner is dp_rank.
    local_samples: List[Tuple[int, int, int]] = []
    send_schedule: Dict[int, List[Tuple[int, int, int]]] = {r: [] for r in range(dp_size)}
    recv_schedule: Dict[int, List[Tuple[int, int, int]]] = {r: [] for r in range(dp_size)}

    local_pos = {g: i for i, g in enumerate(local_global_indices)}

    for global_idx, (dst_slot, owner_rank, canonical_pos) in enumerate(route):
        if not 0 <= dst_slot < window_size:
            raise ValueError(f"dst_slot ({dst_slot}) for global sample {global_idx} out of range [0, {window_size}).")
        src_rank, src_local_idx = location[global_idx]
        if owner_rank == src_rank:
            # Stays put; only this rank records it (when it is the owner/holder).
            if owner_rank == dp_rank:
                local_samples.append((dst_slot, canonical_pos, local_pos[global_idx]))
            continue
        if src_rank == dp_rank:
            send_schedule[owner_rank].append((dst_slot, canonical_pos, src_local_idx))
        if owner_rank == dp_rank:
            recv_schedule[src_rank].append((dst_slot, canonical_pos, src_local_idx))

    # Deterministic order (by slot then canonical position) so sender and receiver agree.
    for r in range(dp_size):
        send_schedule[r].sort()
        recv_schedule[r].sort()

    # Device of the (possibly GPU-resident) sample tensors, so the send buffer and any empty
    # placeholders live on the same device for torch.cat / all_to_all_single.
    dev = torch.device("cpu")
    for f in local_flats:
        t = next((v for v in f.values() if isinstance(v, torch.Tensor)), None)
        if t is not None:
            dev = t.device
            break

    # Build the contiguous send buffer ordered by destination rank.
    send_parts: List[torch.Tensor] = []
    send_splits: List[int] = []
    for dst in range(dp_size):
        if dst == dp_rank or not send_schedule[dst]:
            send_splits.append(0)
            continue
        chunk_parts = []
        for _dst_slot, _canonical_pos, src_local_idx in send_schedule[dst]:
            keys = sample_keys(all_tensor_meta[dp_rank][src_local_idx])
            chunk_parts.append(serialize_sample(local_flats[src_local_idx], keys))
        chunk = torch.cat(chunk_parts) if chunk_parts else torch.empty(0, dtype=torch.uint8, device=dev)
        send_parts.append(chunk)
        send_splits.append(int(chunk.numel()))

    # Recv byte counts from metadata alone.
    recv_splits: List[int] = []
    for src in range(dp_size):
        if src == dp_rank or not recv_schedule[src]:
            recv_splits.append(0)
            continue
        total = 0
        for _dst_slot, _canonical_pos, src_local_idx in recv_schedule[src]:
            keys = sample_keys(all_tensor_meta[src][src_local_idx])
            total += sample_byte_size(all_tensor_meta[src][src_local_idx], keys)
        recv_splits.append(total)

    send_buf = torch.cat(send_parts) if send_parts else torch.empty(0, dtype=torch.uint8, device=dev)
    return {
        "local_samples": local_samples,
        "send_buf": send_buf,
        "send_splits": send_splits,
        "recv_splits": recv_splits,
        "recv_schedule": {r: v for r, v in recv_schedule.items() if v},
    }


# ---------------------------------------------------------------------------
# MIMO data plane: micro-batch <-> per-sample flat dicts (flatten_fn / rewrap_fn)
# ---------------------------------------------------------------------------
#
# The exchange unit is a single sample. A MegatronMIMO micro-batch is nested:
#
#   {input_ids: [B,S], position_ids: [3,B,S] (MRoPE), labels/loss_mask: [B,S] | None,
#    attention_mask: None, modality_inputs: {modality: {encoder: {hidden_states: [ΣP,d],
#    grid_thw: [n_img,3]}}}}
#
# ``split_microbatch`` turns it into B per-sample **flat** dicts (nesting flattened to
# dotted keys) so the Phase-0 serializer/all-to-all can move individual samples; after the
# exchange ``merge_samples`` rebuilds the (rebalanced) micro-batch. Sample slicing/vision
# attribution uses :func:`_apply_sample_dispatch` (LM by batch dim, vision by image via
# ``image_counts``).


def _flatten_nested(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a (possibly nested) batch dict to dotted keys; tensors and ``None`` preserved."""
    flat: Dict[str, Any] = {}
    for key, val in d.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            flat.update(_flatten_nested(val, prefix=f"{path}."))
        else:
            # tensors and None pass through; the buffer's batch has no per-sample list fields.
            flat[path] = val
    return flat


def _unflatten_nested(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`_flatten_nested`: rebuild the nested dict from dotted keys."""
    out: Dict[str, Any] = {}
    for path, val in flat.items():
        parts = path.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def _cu_img_from_counts(image_counts: "Any") -> List[int]:
    """Cumulative per-sample image-offset table from per-sample image counts.

    ``image_counts`` is a length-``B`` int tensor/sequence; returns ``[0, c0, c0+c1, ...]`` (length
    ``B + 1``) so sample ``s`` owns ``grid_thw`` rows ``[cu_img[s], cu_img[s + 1])``.
    """
    counts = image_counts.tolist() if isinstance(image_counts, torch.Tensor) else list(image_counts)
    cu = [0]
    for c in counts:
        cu.append(cu[-1] + int(c))
    return cu


def empty_like_vision(
    ref_hidden_states: torch.Tensor, ref_grid_thw: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Empty vision tensors matching a reference sample's dtype/device.

    Returns ``([0, d] hidden_states, [0, 3] grid_thw)`` — the placeholder a **text-only** sample
    (0 images) contributes so the per-sample reorder/merge keeps uniform vision keys across samples
    and ranks (the merge concat and the cross-rank byte-split metadata both stay symmetric). Used by
    :func:`merge_samples`.
    """
    empty_hidden_states = ref_hidden_states.new_empty((0, ref_hidden_states.shape[1]))
    empty_grid_thw = ref_grid_thw.new_empty((0, ref_grid_thw.shape[1]))
    return empty_hidden_states, empty_grid_thw


def patches_per_image(grid_thw: torch.Tensor) -> torch.Tensor:
    """Per-image vision patch count ``prod(t, h, w)`` for a ``[n_images, 3]`` grid.

    Centralizes the Qwen-VL "patches per image = ``prod(grid_thw[i])``" convention shared by
    the vision reorder and cost paths so a future grid layout change is a single edit.
    """
    return torch.prod(grid_thw, dim=1)


def _reorder_vision_by_images(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    image_perm: list[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permute vision ``hidden_states`` (patch-dim) and ``grid_thw`` (image-dim) by image.

    The patch block of image ``i`` (rows ``cu[i]:cu[i+1]`` where ``cu`` is the cumulative
    patch count from ``prod(grid_thw[i])``) is gathered in ``image_perm`` order so the two
    stay aligned — the patch-dim equivalent of ``grid_thw[image_perm]``. Reorders whole images
    (keeping each image's patch block intact) rather than slicing across image boundaries.

    Args:
        hidden_states: Vision patch features ``[Σ patches, dim]``.
        grid_thw: Per-image grid ``[n_images, 3]``.
        image_perm: Permutation of ``range(n_images)``.

    Returns:
        ``(hidden_states_perm, grid_thw_perm)``.
    """
    if not image_perm:
        # No images selected (e.g. a text-only sample): return empty ``[0, d]`` / ``[0, 3]`` views
        # (avoids ``torch.cat([])`` and preserves dtype/device/requires_grad).
        return hidden_states[:0], grid_thw[:0]
    patches = patches_per_image(grid_thw)  # [n_images]
    # Single host copy of the cumulative offsets (F14): materialize the cumsum once with one
    # .tolist(), then index with plain Python ints — no per-image .item() D2H sync in the loop.
    cu = torch.cat([patches.new_zeros(1), patches.cumsum(0)]).tolist()
    blocks = [hidden_states[cu[i] : cu[i + 1]] for i in image_perm]
    perm_idx = torch.as_tensor(image_perm, dtype=torch.long, device=grid_thw.device)
    return torch.cat(blocks, dim=0), grid_thw.index_select(0, perm_idx)


def _gather_vision_subdict(
    value: Dict[str, Any],
    sample_indices: list[int],
    n_samples: int,
    *,
    cu_img: "list[int] | None" = None,
) -> Dict[str, Any]:
    """Reorder/gather a vision ``{hidden_states, grid_thw}`` sub-dict by sample.

    Supports a **variable number of images per sample** (0 = text-only, 1, or N). ``cu_img`` is the
    cumulative per-sample image-offset table (length ``n_samples + 1``), so sample ``s`` owns
    ``grid_thw`` rows ``[cu_img[s], cu_img[s + 1])``. The image permutation for ``sample_indices`` is
    the concatenation of each selected sample's image range, gathered by
    :func:`_reorder_vision_by_images` (which keeps ``hidden_states`` patch-dim and ``grid_thw``
    image-dim aligned). A text-only sample contributes an empty ``[0, d]`` / ``[0, 3]`` block.

    When ``cu_img`` is ``None`` the legacy one-image-per-sample mapping (image ``i`` ↔ sample ``i``)
    is used and requires ``n_images == n_samples``.

    Args:
        value: A sub-dict carrying ``hidden_states`` and ``grid_thw``.
        sample_indices: Per-sample permutation/selection (global sample ids).
        n_samples: Global batch sample count ``B``.
        cu_img: Cumulative per-sample image offsets (length ``B + 1``), or ``None`` for the legacy
            one-image-per-sample path.

    Returns:
        A new sub-dict with permuted ``hidden_states`` / ``grid_thw`` (other keys carried).

    Raises:
        ValueError: If ``cu_img`` is ``None`` and the batch is not one-image-per-sample, or if
            ``cu_img`` does not sum to ``n_images`` (mis-sourced image counts).
    """
    hidden_states, grid_thw = value["hidden_states"], value["grid_thw"]
    n_images = grid_thw.shape[0]
    if cu_img is None:
        if n_images != n_samples:
            raise ValueError(
                f"Vision reorder without per-sample image counts expects one image per sample "
                f"(n_images={n_images}, n_samples={n_samples}); pass cu_img for variable counts."
            )
        image_perm = list(sample_indices)
    else:
        if cu_img[-1] != n_images:
            raise ValueError(
                f"Per-sample image counts sum to {cu_img[-1]} but grid_thw has {n_images} images; "
                "image_count_of is likely mis-sourced (wrong token id / off-by-one)."
            )
        image_perm = [img for s in sample_indices for img in range(cu_img[s], cu_img[s + 1])]

    hs, g = _reorder_vision_by_images(hidden_states, grid_thw, image_perm)
    new_sub = dict(value)
    new_sub["hidden_states"] = hs
    new_sub["grid_thw"] = g
    return new_sub


def _apply_sample_dispatch(
    batch: Dict[str, Any],
    sample_indices: list[int],
    *,
    n_samples: int,
    cu_img: "list[int] | None" = None,
) -> Dict[str, Any]:
    """Shared per-sample gather/permute visitor for tensors, vision sub-dicts, and lists.

    Used by :func:`split_microbatch` (per-sample local-shard selection); ``n_samples`` is the
    canonical global batch size ``B`` so list/vision per-sample detection is consistent. ``cu_img``
    (cumulative per-sample image offsets) is forwarded to :func:`_gather_vision_subdict` for the
    variable-images-per-sample reorder; ``None`` keeps the legacy one-image-per-sample path.
    """
    idx = torch.as_tensor(sample_indices, dtype=torch.long)
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_dim = _batch_dim_for_tensor(key, value)
            out[key] = value.index_select(batch_dim, idx.to(value.device))
        elif isinstance(value, dict):
            if is_vision_subdict(value):
                out[key] = _gather_vision_subdict(value, sample_indices, n_samples, cu_img=cu_img)
            else:
                out[key] = _apply_sample_dispatch(value, sample_indices, n_samples=n_samples, cu_img=cu_img)
        elif isinstance(value, list):
            # Per-sample list -> permuted gather; global-metadata / empty list -> passthrough.
            if len(value) == n_samples:
                out[key] = [value[i] for i in sample_indices]
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def split_microbatch(batch: Dict[str, Any], *, cu_img: "List[int] | None" = None) -> List[Dict[str, Any]]:
    """Split a micro-batch into ``B`` per-sample flat (dotted-key) dicts.

    Each sample is gathered with :func:`_apply_sample_dispatch` (so its image travels with it)
    and then flattened to dotted keys for serialization. Leading batch dims are kept (size 1),
    so :func:`merge_samples` is a plain concat.

    ``cu_img`` (cumulative per-sample image offsets, length ``B + 1``) lets the vision gather support
    a variable number of images per sample (0 = text-only, 1, or N). When ``None``, the legacy
    one-image-per-sample path is used.
    """
    b = _batch_size(batch)
    return [_flatten_nested(_apply_sample_dispatch(batch, [i], n_samples=b, cu_img=cu_img)) for i in range(b)]


def merge_samples(flats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rebuild a micro-batch from per-sample flat dicts (inverse of :func:`split_microbatch`).

    LM tensors concat on their batch dim (:func:`dp_utils._batch_dim_for_tensor`); vision
    ``hidden_states`` concat on the patch dim and ``grid_thw`` on the image dim; ``None`` and
    scalar/global fields are taken from the first sample.
    """
    if not flats:
        raise ValueError("merge_samples requires at least one sample.")
    nested = [_unflatten_nested(f) for f in flats]
    return _merge_nested(nested)


def _merge_nested(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Iterate the union of keys (not just ``samples[0]``) and classify a vision sub-dict by scanning
    # *all* samples: a text-only sample (0 images) carries the sub-dict as empty/None, so keying off
    # ``samples[0]`` alone would misclassify it as a plain nested dict and silently drop the other
    # samples' vision. With the empty-tensor convention every sample presents the sub-dict as a
    # tensor, so the empty substitution below is a defensive backstop.
    keys: List[str] = []
    seen = set()
    for s in samples:
        for k in s:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    out: Dict[str, Any] = {}
    for key in keys:
        vals = [s.get(key) for s in samples]
        vis_ref = next((v for v in vals if is_vision_subdict(v)), None)
        if vis_ref is not None:
            # Vision sub-dict: concat hidden_states (patch dim) + grid_thw (image dim), substituting
            # an empty [0,d]/[0,3] for any text-only sample (None/absent) so the concat stays aligned.
            ref_hidden_states, ref_grid_thw = vis_ref["hidden_states"], vis_ref["grid_thw"]
            hidden_parts: List[torch.Tensor] = []
            grid_parts: List[torch.Tensor] = []
            for v in vals:
                if is_vision_subdict(v):
                    hidden_parts.append(v["hidden_states"])
                    grid_parts.append(v["grid_thw"])
                else:
                    empty_hidden, empty_grid = empty_like_vision(ref_hidden_states, ref_grid_thw)
                    hidden_parts.append(empty_hidden)
                    grid_parts.append(empty_grid)
            sub = dict(vis_ref)
            sub["hidden_states"] = torch.cat(hidden_parts, dim=0)
            sub["grid_thw"] = torch.cat(grid_parts, dim=0)
            out[key] = sub
            continue

        rep = next((v for v in vals if v is not None), None)
        if isinstance(rep, dict):
            out[key] = _merge_nested([v for v in vals if isinstance(v, dict)])
        elif isinstance(rep, torch.Tensor):
            out[key] = torch.cat(vals, dim=_batch_dim_for_tensor(key, rep))
        else:
            out[key] = rep  # None / scalar / global metadata
    return out


def _batch_to_cuda(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively move a (possibly nested) batch dict's tensors to CUDA (non-blocking)."""
    out: Dict[str, Any] = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.cuda(non_blocking=True)
        elif isinstance(val, dict):
            out[key] = _batch_to_cuda(val)
        else:
            out[key] = val
    return out


def _batch_size(batch: Dict[str, Any]) -> int:
    """Sample count of a micro-batch from a known LM tensor (handles MRoPE position_ids)."""
    for key in ("input_ids", "labels", "loss_mask", "position_ids"):
        t = batch.get(key)
        if isinstance(t, torch.Tensor):
            return t.size(_batch_dim_for_tensor(key, t))
    raise ValueError("Cannot determine batch size: no input_ids/labels/loss_mask/position_ids tensor.")


# ---------------------------------------------------------------------------
# Per-sample joint cost (from a flat sample)
# ---------------------------------------------------------------------------


def sample_cost(
    flat: Dict[str, Any],
    *,
    linear_vit: float,
    linear_lm: float = 0.0,
    pad_token_id: int = 0,
    image_token_id: "int | None" = None,
    square_merge_size: int = 1,
) -> float:
    """Joint per-sample cost: ``linear_vit·p + linear_lm·t``.

    Both terms are **linear** in an intrinsic per-sample workload count:

    - ``p`` is the vision patch count (mirrors vision-encoder FLOPs).
    - ``t`` is the **real (non-pad) token count** of ``input_ids`` (mirrors LM FLOPs), counted
      via :func:`real_token_lengths` (``attention_mask`` when present, else ``!= pad_token_id``).

    Both terms must be **collation-independent** so vision and language derive an identical
    assignment with no cross-module communication: the patch count is intrinsic to the image,
    and the real token count is intrinsic to the sample (invariant to how each module's collator
    pads). The *padded* sequence length is collation-dependent and is deliberately **not** used —
    it would differ between the two shards and mispair the ``BridgeCommunicator`` fan-out.

    Patch-count source (must be **module-independent**):

    - When ``image_token_id`` is given, ``p`` is the image-placeholder token count in ``input_ids``
      scaled by ``square_merge_size`` (each placeholder token stands for ``spatial_merge_size²``
      patches in Qwen-VL). ``input_ids`` is present and identical on **every** module/PP stage when
      reorder is active, so vision and language derive the same cost. This is the correct source:
      the rank-aware metadata collate (#4442) nulls ``modality_inputs``/``grid_thw`` on language
      shards, so reading ``grid_thw`` there yields ``p = 0`` and mispairs the vision↔language fan-out.
    - Otherwise ``p = Σ prod(grid_thw)`` (the legacy patch-only path, used by mock/tests where no
      ``image_token_id`` is wired and ``grid_thw`` is present on the sample).

    With the default ``linear_lm=0.0`` the cost is patch-only (the historical behavior).

    Args:
        flat: One per-sample flat (dotted-key) dict.
        linear_vit: Per-patch ViT cost coefficient.
        linear_lm: Per-token LM cost coefficient. ``0.0`` disables the LM term (patch-only).
        pad_token_id: Pad id for the ``!= pad`` real-length fallback (used only when no
            ``attention_mask`` is present and ``linear_lm > 0``).
        image_token_id: Placeholder token id whose count in ``input_ids`` is the module-independent
            patch proxy. ``None`` falls back to the ``grid_thw`` patch sum.
        square_merge_size: ``spatial_merge_size²`` — patches represented by one placeholder token.
            Recovers the true patch count from the placeholder-token count (only used with
            ``image_token_id``).
    """
    patches = 0
    if image_token_id is not None:
        input_ids = flat.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.numel():
            patches = int((input_ids == image_token_id).sum().item()) * square_merge_size
    else:
        for key, val in flat.items():
            if key.endswith(".grid_thw") and isinstance(val, torch.Tensor) and val.numel():
                patches += int(torch.prod(val, dim=1).sum().item())

    tokens = 0
    if linear_lm:
        input_ids = flat.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.numel():
            attention_mask = flat.get("attention_mask")
            lengths = real_token_lengths(
                input_ids,
                pad_token_id=pad_token_id,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            )
            tokens = int(lengths.sum().item())

    return linear_vit * patches + linear_lm * tokens


# ---------------------------------------------------------------------------
# Per-slot reassembly + transport-only route exchange (W micro-batch slots)
# ---------------------------------------------------------------------------


def reassemble_window(
    local_flats: List[Dict[str, Any]],
    plan: Dict[str, Any],
    recv_buf: torch.Tensor,
    all_tensor_meta: List[List[Dict[str, Any]]],
    *,
    dp_size: int,
    window_size: int,
    local: int,
) -> List[Dict[str, Any]]:
    """Rebuild this rank's ``W`` micro-batches from kept-local + received samples (§4a).

    Generalizes the single-micro-batch reassembler to ``W`` slot-buckets: kept-local samples and
    deserialized received samples are binned by ``dst_slot``, sorted **within each slot** by
    ``canonical_pos`` (the paired-module-identical order), and merged into one micro-batch per slot
    via :func:`merge_samples`. ``deserialize_sample`` / ``merge_samples`` are reused verbatim — the
    only new step is the per-slot binning.

    The §10.0 invariant asserts are baked in (cheap, always on): they turn a silent slot mispair or
    a dropped/duplicated sample into a loud failure at the exact rank.

    Args:
        local_flats: This rank's input samples (the kept-local ones are indexed by ``local_idx``).
        plan: Output of :func:`prepare_sample_exchange` (carries ``local_samples`` / ``recv_schedule``
            with ``dst_slot``).
        recv_buf: The ``all_to_all_single`` output buffer (samples this rank received).
        all_tensor_meta: All-gathered per-rank tensor metadata (to decode ``recv_buf``).
        dp_size: This module's DP size.
        window_size: Number of micro-batch slots ``W`` to rebuild.
        local: Expected samples per (slot) bucket for this rank (``B/dp`` per slot).

    Returns:
        ``W`` rebuilt micro-batches, slot ``s`` at index ``s``.
    """
    buckets: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(window_size)]
    for dst_slot, canonical_pos, local_idx in plan["local_samples"]:
        buckets[dst_slot].append((canonical_pos, local_flats[local_idx]))

    offset = 0
    for src in range(dp_size):
        for dst_slot, canonical_pos, src_local_idx in plan["recv_schedule"].get(src, []):
            meta = all_tensor_meta[src][src_local_idx]
            flat, offset = deserialize_sample(recv_buf, offset, meta, sample_keys(meta))
            buckets[dst_slot].append((canonical_pos, flat))

    # Byte accounting: the deserialize cursor must consume exactly the received bytes.
    if offset != recv_buf.numel():
        raise RuntimeError(f"MegatronMIMO reassembly consumed {offset} bytes but received {recv_buf.numel()}.")

    out: List[Dict[str, Any]] = []
    for slot, bucket in enumerate(buckets):
        # Slot shape: each slot bucket must hold exactly `local` samples.
        if len(bucket) != local:
            raise RuntimeError(f"MegatronMIMO slot {slot} reassembled {len(bucket)} samples, expected {local}.")
        bucket.sort(key=lambda x: x[0])
        # Order: canonical positions within a slot must be unique (no dropped/duplicated sample).
        positions = [p for p, _ in bucket]
        if len(set(positions)) != len(positions):
            raise RuntimeError(f"MegatronMIMO slot {slot} has duplicate canonical positions: {positions}.")
        out.append(merge_samples([f for _pos, f in bucket]))
    return out


def exchange_by_route(
    local_flats: List[Dict[str, Any]],
    route: List[Tuple[int, int, int]],
    all_global_indices: List[List[int]],
    all_tensor_meta: List[List[Dict[str, Any]]],
    *,
    dp_rank: int,
    dp_size: int,
    window_size: int,
    local: int,
    dp_group_nccl: "Any",
    nccl_stream: "Any" = None,
) -> List[Dict[str, Any]]:
    """Transport-only per-sample exchange: route -> one ``all_to_all_single`` -> ``W`` micro-batches.

    The inter-ready seam (§3.2): given a 3D ``route`` and the all-gathered layout/metadata, build the
    plan (:func:`prepare_sample_exchange`), run **one** ragged ``all_to_all_single``, and reassemble
    ``W`` micro-batches per slot (:func:`reassemble_window`). Knows nothing about the balancer — an
    intra route (``dst_slot == src_slot``, see :func:`intra_route`) and a future inter route flow
    through unchanged.

    Args:
        local_flats: This rank's input samples (GPU-resident when CUDA is available).
        route: ``(dst_slot, owner_rank, canonical_pos)`` per global sample index.
        all_global_indices: ``all_global_indices[r]`` = global indices currently held by rank ``r``.
        all_tensor_meta: All-gathered per-rank tensor metadata.
        dp_rank, dp_size: Module-local DP coordinates.
        window_size: Number of micro-batch slots ``W`` in the window.
        local: Samples per (slot) bucket for this rank.
        dp_group_nccl: NCCL PG for the sample all-to-all.
        nccl_stream: Optional CUDA stream to run the all-to-all on (cross-step overlap).

    Returns:
        ``W`` rebuilt micro-batches for this rank, slot ``s`` at index ``s``.
    """
    import torch.distributed as dist

    plan = prepare_sample_exchange(
        local_flats,
        all_global_indices[dp_rank],
        route,
        all_global_indices,
        all_tensor_meta,
        dp_rank,
        dp_size,
        window_size=window_size,
    )

    # send_buf is already on GPU (built from GPU-resident sample tensors); no H2D copy.
    send_buf = plan["send_buf"]
    recv_buf = torch.empty(sum(plan["recv_splits"]), dtype=torch.uint8, device=send_buf.device)
    if nccl_stream is not None:
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        nccl_stream.wait_event(ev)
        with torch.cuda.stream(nccl_stream):
            dist.all_to_all_single(recv_buf, send_buf, plan["recv_splits"], plan["send_splits"], group=dp_group_nccl)
        done = torch.cuda.Event()
        done.record(nccl_stream)
        torch.cuda.current_stream().wait_event(done)
    else:
        dist.all_to_all_single(recv_buf, send_buf, plan["recv_splits"], plan["send_splits"], group=dp_group_nccl)

    return reassemble_window(
        local_flats, plan, recv_buf, all_tensor_meta, dp_size=dp_size, window_size=window_size, local=local
    )


# ---------------------------------------------------------------------------
# Distributed per-sample exchange (a window of W micro-batches, one module DP group)
# ---------------------------------------------------------------------------


def exchange_window(
    batches: List[Dict[str, Any]],
    *,
    dp_rank: int,
    dp_size: int,
    n_groups: int,
    cost_of: "Any",
    dp_group_gloo: "Any",
    dp_group_nccl: "Any",
    nccl_stream: "Any" = None,
    image_count_of: "Any" = None,
    probe: bool = False,
) -> List[Dict[str, Any]]:
    """Cost-balance a **window** of ``W`` micro-batches across a module DP group in one exchange.

    Collects this rank's ``W`` disjoint shards, all-gathers their per-sample costs + tensor metadata
    over Gloo **once**, builds the slot-major window route (:func:`build_window_route` — the intra
    balancer run independently per slot), and runs **one** ragged ``all_to_all_single`` over the
    whole window, reassembling ``W`` balanced micro-batches per slot (:func:`reassemble_window`).
    The single collective lands once per window (≈ once per optimizer step when ``W == GA``).

    The window global index of slot ``s``'s slot-local sample ``j`` is ``g = s · B_global + j``; this
    rank holds, in ``local_flats`` order, its slot-0 shard then slot-1 shard … (``local = B/dp`` per
    slot). Both modules feed identical ``n_groups`` and per-slot costs, so paired replicas end with
    the same samples in canonical order in every slot.

    Args:
        batches: This rank's ``W`` disjoint micro-batch shards (each ``local = B/dp`` samples, same
            ``local`` across slots).
        dp_rank, dp_size: Module-local DP coordinates.
        n_groups: Canonical group count (``max`` module DP size).
        cost_of: ``callable(flat) -> float`` per-sample patch cost (see :func:`sample_cost`).
        dp_group_gloo: Gloo PG for the metadata all-gather.
        dp_group_nccl: NCCL PG for the sample all-to-all.
        nccl_stream: Optional CUDA stream to run the all-to-all on (cross-step overlap).
        image_count_of: Optional ``callable(batch) -> LongTensor[local]`` giving a shard's per-sample
            image count, enabling a variable number of images per sample (0/1/N). When ``None``, the
            legacy one-image-per-sample path is used.
        probe: When ``True``, log a one-line per-slot cost-spread digest (:func:`window_cost_spread`)
            on ``dp_rank == 0`` — the §10.5 balance probe. Off by default (the caller throttles it).

    Returns:
        ``W`` rebalanced micro-batches for this rank, slot ``s`` at index ``s``.
    """
    import torch.distributed as dist

    window_size = len(batches)
    if window_size == 0:
        raise ValueError("exchange_window requires at least one micro-batch.")

    # Cost and metadata are computed on the CPU batches before the optional CUDA move, avoiding
    # per-sample .item() synchronizations on the training stream. Per-slot image-count offset tables
    # are derived once and reused for both the CPU-meta and GPU-payload splits.
    cu_imgs = [_cu_img_from_counts(image_count_of(b)) if image_count_of is not None else None for b in batches]
    cpu_flats_per_slot = [split_microbatch(b, cu_img=cu) for b, cu in zip(batches, cu_imgs)]
    local = len(cpu_flats_per_slot[0])
    if any(len(slot) != local for slot in cpu_flats_per_slot):
        raise ValueError(
            f"All micro-batches in a window must have the same local size; got {[len(s) for s in cpu_flats_per_slot]}."
        )

    # Window-local order is slot-major: this rank's slot-0 shard, then slot-1, … (matches the
    # g = s·B_global + j layout build_window_route assumes).
    cpu_local_flats = [f for slot in cpu_flats_per_slot for f in slot]
    local_meta = [tensor_metadata(f) for f in cpu_local_flats]
    local_costs_per_slot = [[cost_of(f) for f in slot] for slot in cpu_flats_per_slot]

    # Keep the exchange GPU-resident after cost calculation: move this rank's (small) shards to the
    # device once, then split / serialize / all_to_all / deserialize / merge all run on GPU and the
    # returned batches are already on GPU for the forward pass.
    if torch.cuda.is_available():
        gpu_flats_per_slot = [split_microbatch(_batch_to_cuda(b), cu_img=cu) for b, cu in zip(batches, cu_imgs)]
        local_flats = [f for slot in gpu_flats_per_slot for f in slot]
    else:
        local_flats = cpu_local_flats

    payload = {"costs_per_slot": local_costs_per_slot, "meta": local_meta}
    gathered: List[Any] = [None] * dp_size
    dist.all_gather_object(gathered, payload, group=dp_group_gloo)

    b_global = local * dp_size
    # Window global index layout: g = slot·b_global + (src_rank·local + i); rank r's local_flats are
    # ordered slot-major so its window samples are at all_global_indices[r] (same order).
    all_global_indices = [
        [slot * b_global + r * local + i for slot in range(window_size) for i in range(local)] for r in range(dp_size)
    ]
    all_tensor_meta = [g["meta"] for g in gathered]
    # Per-slot global costs indexed by slot-local position j = r·local + i (contiguous sharding).
    costs_per_slot: List[List[float]] = [[0.0] * b_global for _ in range(window_size)]
    for r, g in enumerate(gathered):
        for slot in range(window_size):
            for i, c in enumerate(g["costs_per_slot"][slot]):
                costs_per_slot[slot][r * local + i] = c

    # Balancing is driven by the patch-only cost (see sample_cost): identical across modules, so both
    # compute the same per-slot canonical order with no cross-module communication.
    route = build_window_route(costs_per_slot, n_groups, dp_size)

    if probe and dp_rank == 0:
        digest = " ".join(
            f"slot{s}:{sp['before_max']:.0f}/{sp['before_min']:.0f}->{sp['after_max']:.0f}/{sp['after_min']:.0f}"
            f"(remote={int(sp['remote'])})"
            for s, sp in enumerate(window_cost_spread(costs_per_slot, route, dp_size))
        )
        logger.info("MIMO reorder balance probe (W=%d) per-slot cost max/min before->after: %s", window_size, digest)

    return exchange_by_route(
        local_flats,
        route,
        all_global_indices,
        all_tensor_meta,
        dp_rank=dp_rank,
        dp_size=dp_size,
        window_size=window_size,
        local=local,
        dp_group_nccl=dp_group_nccl,
        nccl_stream=nccl_stream,
    )


# ---------------------------------------------------------------------------
# Reorder buffer (data-iterator wrapper)
# ---------------------------------------------------------------------------


_SENTINEL = object()


class _WorkerException:
    """Container used to re-raise overlap-worker failures on the consumer thread."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _record_stream_for_batch(value: Any, stream: torch.cuda.Stream) -> None:
    """Recursively tie CUDA tensors in ``value`` to ``stream`` before cross-thread handoff."""
    if isinstance(value, torch.Tensor) and value.is_cuda:
        value.record_stream(stream)
    elif isinstance(value, dict):
        for child in value.values():
            _record_stream_for_batch(child, stream)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _record_stream_for_batch(child, stream)


class ReorderingBuffer:
    """Wrap a MegatronMIMO data iterator so each yielded micro-batch is cost-balanced.

    Each item this rank consumes is its disjoint scalable-data-parallel shard rebalanced by
    :func:`exchange_window` (per-sample cost all-gather + ragged all-to-all over the module
    DP group).

    **Window buffering** (``window_size = W``): collect ``W`` micro-batches, cost-balance them in one
    exchange over the window (:func:`exchange_window`), and serve the ``W`` rebalanced micro-batches
    one at a time from a cursor. ``W = 1`` (default) is the historical per-micro-batch behavior,
    byte-for-byte. ``W == GA`` (the gradient-accumulation count) is the desired setting — the single
    window collective then lands once per optimizer step.

    **Cross-window prefetch overlap** (``overlap=True``): a background thread runs the *entire next
    window's* exchange — the blocking Gloo metadata all-gather *and* the NCCL all-to-all (on a
    dedicated CUDA stream) — while the main thread computes the current window's ``W`` micro-batches.
    This is what turns the scalable-data-parallel read win into a net throughput win: one big
    all-to-all is hidden behind ``W`` micro-batches of compute instead of sitting on the critical
    path (for ``W = 1`` this is the one-step-ahead per-micro-batch overlap). The exchange's GPU work
    is synchronized inside the worker before hand-off, so the main thread consumes fully materialized
    GPU batches with no further wait. A ``maxsize=1`` queue keeps every rank exactly one *window*
    ahead, so all ranks' Gloo/NCCL collectives stay in lockstep — re-synced each window by the
    optimizer/DDP all-reduce barrier between consumptions. The worker uses the (separate)
    ``dp_group_nccl`` exclusively, so its all-to-all never races the main thread's bridge/DDP
    collectives on their own PGs (requires ``CUDA_DEVICE_MAX_CONNECTIONS != 1``). Two windows stay
    resident (consuming + prefetched) ≈ ``2·W`` micro-batches — the headline memory cost.

    With ``overlap=False`` the window exchange runs synchronously in ``__next__`` (no thread).
    """

    def __init__(
        self,
        data_iterator: "Any",
        *,
        dp_rank: int,
        dp_size: int,
        n_groups: int,
        cost_of: "Any",
        dp_group_gloo: "Any",
        dp_group_nccl: "Any",
        overlap: bool = True,
        image_count_of: "Any" = None,
        window_size: int = 1,
    ):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        self._it = data_iterator
        self._dp_rank = dp_rank
        self._dp_size = dp_size
        self._n_groups = n_groups
        self._cost_of = cost_of
        self._image_count_of = image_count_of
        self._gloo = dp_group_gloo
        self._nccl = dp_group_nccl
        self._window_size = window_size
        self._cuda = torch.cuda.is_available()
        self._stream = torch.cuda.Stream() if self._cuda else None
        self._thread = None
        # Sync window state (used whenever the prefetch thread is not running): the active window's
        # rebalanced micro-batches and the cursor serving them one at a time.
        self._active: List[Dict[str, Any]] = []
        self._cursor = 0
        # Count of window exchanges run, for throttling the §10.5 balance probe (first few + every 50th).
        self._exchanges = 0
        # Cross-window prefetch: a thread exchanges window t+1 while the main thread computes
        # window t. Started for any window_size (W == 1 is the one-step-ahead per-micro-batch case).
        if overlap and self._cuda and dp_size > 1:
            # The side-stream exchange only overlaps compute when it can use a separate hardware
            # queue. With CUDA_DEVICE_MAX_CONNECTIONS=1 every stream serializes onto one queue, so
            # the prefetch a2a runs *behind* the window's compute instead of beside it — correct, but
            # silently no faster (often slower, for the extra thread + side PG). Warn loudly rather
            # than appear to overlap. Not a hard error: a user may set =1 for TP/SP comm overlap and
            # still want the (degraded) reorder to run.
            if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
                logger.warning(
                    "ReorderingBuffer overlap is enabled but CUDA_DEVICE_MAX_CONNECTIONS=1; the "
                    "exchange will serialize behind compute and give no overlap speedup. Unset it "
                    "or set it != 1 to get overlap, or pass overlap=False to skip the prefetch thread."
                )
            self._device = torch.cuda.current_device()
            self._queue: "queue.Queue" = queue.Queue(maxsize=1)
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._worker, name="mimo-reorder-prefetch", daemon=True)
            self._thread.start()

    def __iter__(self):
        return self

    def shutdown(self) -> None:
        """Stop the overlap worker and wait briefly for it to exit."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)

    def __del__(self) -> None:
        """Best-effort cleanup for runs that drop the iterator without closing it."""
        try:
            self.shutdown()
        except Exception:
            logger.debug("Ignoring exception while shutting down ReorderingBuffer.", exc_info=True)

    def _exchange_window(self, batches: List[Dict[str, Any]], stream: "Any") -> List[Dict[str, Any]]:
        # Throttle the §10.5 balance probe: log the first few windows, then every 50th.
        self._exchanges += 1
        probe = self._exchanges <= 3 or self._exchanges % 50 == 0
        return exchange_window(
            batches,
            dp_rank=self._dp_rank,
            dp_size=self._dp_size,
            n_groups=self._n_groups,
            cost_of=self._cost_of,
            dp_group_gloo=self._gloo,
            dp_group_nccl=self._nccl,
            nccl_stream=stream,
            image_count_of=self._image_count_of,
            probe=probe,
        )

    def _collect_window(self) -> List[Dict[str, Any]]:
        """Pull up to ``window_size`` micro-batches from the source iterator (fewer at the tail)."""
        batches: List[Dict[str, Any]] = []
        for _ in range(self._window_size):
            try:
                batches.append(next(self._it))
            except StopIteration:
                break
        return batches

    def _refill_window(self) -> None:
        """Collect up to ``window_size`` micro-batches, exchange the window, reset the cursor.

        A short final window (fewer than ``window_size`` micro-batches before ``StopIteration``) is
        exchanged at its actual size. ``self._active`` is left empty when the iterator is exhausted.
        """
        batches = self._collect_window()
        if not batches:
            self._active = []
        elif self._dp_size <= 1:
            self._active = batches  # nothing to balance across a single rank
        else:
            self._active = self._exchange_window(batches, stream=self._stream)
        self._cursor = 0

    def _put_worker_item(self, item: Any) -> None:
        """Put an item unless shutdown was requested while the bounded queue was full."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _worker(self) -> None:
        # Cross-window prefetch: exchange window t+1 while the main thread computes window t.
        # Each queue item is the LIST of W rebalanced micro-batches for one window. Everything
        # (Gloo metadata all-gather + NCCL all-to-all + deserialize/merge) runs here on self._stream;
        # we synchronize that stream before queuing so the consumer gets ready GPU batches.
        torch.cuda.set_device(self._device)
        try:
            while not self._stop.is_set():
                batches = self._collect_window()
                if not batches:
                    break
                with torch.cuda.stream(self._stream):
                    out = self._exchange_window(batches, stream=None)  # a2a on the current (self._stream)
                self._stream.synchronize()  # blocks the WORKER (not the main thread)
                self._put_worker_item(out)
        except BaseException as exc:
            self._put_worker_item(_WorkerException(exc))
        finally:
            self._put_worker_item(_SENTINEL)

    def __next__(self) -> Dict[str, Any]:
        if self._thread is None:
            # Synchronous path (no prefetch thread): serve the active window from the cursor,
            # refilling (collect W -> one exchange) when it is exhausted.
            if self._cursor >= len(self._active):
                self._refill_window()
            if not self._active:
                raise StopIteration
            out = self._active[self._cursor]
            self._cursor += 1
            return out
        # Cross-window prefetch path: serve from the prefetched window; pull the next one (a list of
        # W micro-batches) from the worker when the cursor is exhausted.
        if self._cursor >= len(self._active):
            item = self._queue.get()
            if item is _SENTINEL:
                raise StopIteration
            if isinstance(item, _WorkerException):
                raise item.exc
            if torch.cuda.is_available():
                _record_stream_for_batch(item, torch.cuda.current_stream())
            self._active = item
            self._cursor = 0
        out = self._active[self._cursor]
        self._cursor += 1
        return out


def build_module_dp_process_groups(dp_group_nccl_main: "Any", *, overlap: bool) -> "Tuple[int, int, Any, Any]":
    """Create the Gloo (metadata) and side NCCL (exchange) PGs for this rank's module DP group.

    ``dist.new_group`` is collective over the world PG, so every rank must create every module's
    DP groups in the same order. We all-gather each rank's DP-group global ranks, deduplicate and
    sort, then create a Gloo group (and, for overlap, a **separate** NCCL group so the exchange
    all-to-all does not serialize against the gradient all-reduce) for each, keeping the ones this
    rank belongs to.

    Args:
        dp_group_nccl_main: This rank's main DP NCCL group (``grid.get_pg(["dp"])``).
        overlap: When ``True``, allocate a separate NCCL PG for the side-stream exchange;
            otherwise reuse the main DP NCCL group (sync mode).

    Returns:
        ``(dp_rank, dp_size, dp_group_gloo, dp_group_nccl)``.
    """
    import torch.distributed as dist

    dp_rank = dist.get_rank(dp_group_nccl_main)
    dp_size = dist.get_world_size(dp_group_nccl_main)
    my_ranks = list(dist.get_process_group_ranks(dp_group_nccl_main))

    gathered: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, my_ranks)
    seen = set()
    unique: List[List[int]] = []
    for ranks in gathered:
        key = tuple(sorted(ranks))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    unique.sort()

    cur = dist.get_rank()
    dp_group_gloo = None
    for ranks in unique:
        g = dist.new_group(ranks=ranks, backend="gloo")
        if cur in ranks:
            dp_group_gloo = g

    if overlap:
        dp_group_nccl = None
        for ranks in unique:
            g = dist.new_group(ranks=ranks, backend="nccl")
            if cur in ranks:
                dp_group_nccl = g
    else:
        dp_group_nccl = dp_group_nccl_main

    return dp_rank, dp_size, dp_group_gloo, dp_group_nccl
