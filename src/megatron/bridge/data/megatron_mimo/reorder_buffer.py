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

"""Per-sample reorder buffer for MegatronMIMO intra-microbatch reordering.

Each data rank reads only its disjoint scalable-data-parallel shard of the global micro-batch.
The per-microbatch cost imbalance between shards is corrected by a cost all-gather plus one
ragged per-sample ``all_to_all_single`` over the module DP group, optionally overlapped with
compute. Paired modules derive the same assignment from the same costs and canonical grid, so
the ``BridgeCommunicator`` positional routing stays valid without cross-module communication.
"""

import logging
import math
import os
import queue
import threading
from typing import Any, Dict, List, Tuple

import torch

from megatron.bridge.data.megatron_mimo.dp_utils import (
    _batch_dim_for_tensor,
    _is_patch_packed_visual_dict,
    real_token_lengths,
)


logger = logging.getLogger(__name__)

# Serialized fields are padded to this alignment so the receiver can view-cast any dtype back
# from the byte buffer and compute exact offsets from metadata alone.
_SERIALIZED_TENSOR_BYTE_ALIGNMENT = 8


def _alignment_padding_bytes(nbytes: int) -> int:
    """Return the zero-padding bytes that round ``nbytes`` up to the serialization alignment."""
    return (-nbytes) % _SERIALIZED_TENSOR_BYTE_ALIGNMENT


# Item sizes are precomputed so the byte-size loops do not allocate a throwaway tensor per field.
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
    """Return ``(dtype, byte width)`` for a dtype name from sample metadata."""
    try:
        return _DTYPE_INFO[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype in MegatronMIMO sample metadata: {dtype_name!r}.") from exc


# ---------------------------------------------------------------------------
# Per-sample balanced assignment
# ---------------------------------------------------------------------------


def balanced_index_order(costs: List[float], n_groups: int) -> List[int]:
    """Permute indices so contiguous equal-size groups have balanced total cost.

    Equal-cardinality LPT with a stable tie-break by group index, so paired modules that
    feed the same costs produce the same permutation.

    Args:
        costs: Per-item cost.
        n_groups: Number of contiguous shards; must divide ``len(costs)``.

    Returns:
        A permutation of ``range(len(costs))`` laid out group-by-group.

    Raises:
        ValueError: If ``n_groups <= 0`` or does not divide ``len(costs)``.
    """
    n = len(costs)
    if n_groups <= 0:
        raise ValueError(f"n_groups must be positive, got {n_groups}.")
    if n % n_groups != 0:
        raise ValueError(f"len(costs) ({n}) must be divisible by n_groups ({n_groups}).")
    cap = n // n_groups
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    group_cost = [0.0] * n_groups
    for i in sorted(range(n), key=lambda j: (-costs[j], j)):
        best = min(
            (g for g in range(n_groups) if len(groups[g]) < cap),
            key=lambda g: (group_cost[g], g),
        )
        groups[best].append(i)
        group_cost[best] += costs[i]
    return [i for g in groups for i in g]


def balanced_assignment(costs: List[float], n_groups: int, dp_size: int) -> List[Tuple[int, int]]:
    """Map each sample of one micro-batch to its owner rank and canonical position.

    ``n_groups`` is the canonical grid (LCM of the module DP sizes); a module of size ``dp_size``
    owns ``n_groups // dp_size`` consecutive groups, so modules with different DP sizes agree.

    Args:
        costs: Per-sample cost indexed by global position in the micro-batch.
        n_groups: Canonical group count; must divide ``len(costs)``.
        dp_size: This module's DP size; must divide ``n_groups``.

    Returns:
        ``(owner_rank, canonical_pos)`` per global sample index.

    Raises:
        ValueError: If ``n_groups`` does not divide ``len(costs)`` or ``dp_size`` does not divide it.
    """
    n = len(costs)
    if n_groups <= 0 or n % n_groups != 0:
        raise ValueError(f"n_groups ({n_groups}) must be positive and divide len(costs) ({n}).")
    if dp_size <= 0 or n_groups % dp_size != 0:
        raise ValueError(f"dp_size ({dp_size}) must be positive and divide n_groups ({n_groups}).")

    perm = balanced_index_order(costs, n_groups)
    cap = n // n_groups
    groups_per_rank = n_groups // dp_size

    assignment: List[Tuple[int, int]] = [(0, 0)] * n
    for canonical_pos, global_idx in enumerate(perm):
        canonical_group = canonical_pos // cap
        owner_rank = canonical_group // groups_per_rank
        assignment[global_idx] = (owner_rank, canonical_pos)
    return assignment


def intra_route(assignment: List[Tuple[int, int]], *, src_slot: int = 0) -> List[Tuple[int, int, int]]:
    """Lift a single-slot assignment to the ``(dst_slot, owner_rank, canonical_pos)`` window route.

    Intra routing never moves a sample across slots, so ``dst_slot == src_slot`` for every sample.

    Args:
        assignment: Output of :func:`balanced_assignment` for one slot.
        src_slot: Slot the samples were read from.

    Returns:
        ``(dst_slot, owner_rank, canonical_pos)`` per global sample index.
    """
    return [(src_slot, owner_rank, canonical_pos) for owner_rank, canonical_pos in assignment]


def build_window_route(costs_per_slot: List[List[float]], n_groups: int, dp_size: int) -> List[Tuple[int, int, int]]:
    """Build the slot-major window route by balancing each slot independently.

    The window global index of slot ``s``, slot-local position ``j`` is ``g = s * B_global + j``.

    Args:
        costs_per_slot: Per-slot per-sample costs indexed by slot-local position.
        n_groups: Canonical group count; must divide each slot's sample count.
        dp_size: This module's DP size.

    Returns:
        ``(dst_slot, owner_rank, canonical_pos)`` per window global index.
    """
    route: List[Tuple[int, int, int]] = []
    for slot, costs in enumerate(costs_per_slot):
        route.extend(intra_route(balanced_assignment(costs, n_groups, dp_size), src_slot=slot))
    assert_intra_no_cross_slot(route, len(costs_per_slot[0]) if costs_per_slot else 0)
    return route


def assert_intra_no_cross_slot(route: List[Tuple[int, int, int]], b_global: int) -> None:
    """Raise if any sample's ``dst_slot`` differs from its source slot ``g // b_global``.

    Args:
        route: ``(dst_slot, owner_rank, canonical_pos)`` per window global index.
        b_global: Samples per micro-batch slot; ``0`` skips the check.

    Raises:
        RuntimeError: If a sample's ``dst_slot`` differs from its source slot.
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
    """Compute per-slot per-rank cost max/min before and after balancing and the remote-sample count.

    Args:
        costs_per_slot: Per-slot per-sample costs indexed by slot-local position.
        route: Window route from :func:`build_window_route`.
        dp_size: This module's DP size.

    Returns:
        One ``{before_max, before_min, after_max, after_min, remote}`` dict per slot.
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
    """Return the sorted tensor keys of a sample's metadata, identical on sender and receiver.

    Args:
        meta: Sample metadata from :func:`tensor_metadata`.

    Returns:
        Sorted keys whose metadata entry carries a ``dtype``.
    """
    return sorted(k for k, v in meta.items() if isinstance(v, dict) and "dtype" in v)


def tensor_metadata(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Build per-field ``{shape, dtype}`` / ``{non_tensor: value}`` / ``None`` metadata for one sample.

    Args:
        flat: One flattened sample dict.

    Returns:
        Metadata keyed like ``flat``, sufficient to size and decode the sample without its data.

    Raises:
        ValueError: If a tensor field has a dtype the serializer does not support.
    """
    meta: Dict[str, Any] = {}
    for key, t in flat.items():
        if isinstance(t, torch.Tensor):
            dtype_name = str(t.dtype)
            _dtype_info(dtype_name)  # fail on the producing rank, before the metadata all-gather
            meta[key] = {"shape": list(t.shape), "dtype": dtype_name}
        elif t is not None:
            meta[key] = {"non_tensor": t}
        else:
            meta[key] = None
    return meta


def serialize_sample(flat: Dict[str, Any], keys: List[str]) -> torch.Tensor:
    """Serialize one sample's tensors into a 1D aligned ``uint8`` buffer in ``keys`` order.

    Args:
        flat: Flattened sample dict; ``None`` values are skipped.
        keys: Sorted tensor keys from :func:`sample_keys`.

    Returns:
        1D ``uint8`` tensor with each field padded to the serialization alignment.
    """
    parts = []
    for key in keys:
        t = flat.get(key)
        if t is None:
            continue
        raw = t.contiguous().view(torch.uint8).reshape(-1)
        pad = _alignment_padding_bytes(raw.numel())
        if pad:
            raw = torch.cat([raw, torch.zeros(pad, dtype=torch.uint8, device=raw.device)])
        parts.append(raw)
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)


def sample_byte_size(meta: Dict[str, Any], keys: List[str]) -> int:
    """Return the serialized byte size of a sample from its metadata alone.

    Args:
        meta: Sample metadata from :func:`tensor_metadata`.
        keys: Sorted tensor keys from :func:`sample_keys`.

    Returns:
        Byte count matching :func:`serialize_sample` for the same sample.
    """
    total = 0
    for key in keys:
        info = meta.get(key)
        if info is None:
            continue
        _, elem = _dtype_info(info["dtype"])
        nbytes = math.prod(info["shape"]) * elem
        nbytes += _alignment_padding_bytes(nbytes)
        total += nbytes
    return total


def deserialize_sample(
    buf: torch.Tensor, offset: int, meta: Dict[str, Any], keys: List[str]
) -> Tuple[Dict[str, Any], int]:
    """Reconstruct one sample from a ``uint8`` buffer (inverse of :func:`serialize_sample`).

    Args:
        buf: Received 1D ``uint8`` buffer.
        offset: Byte offset of this sample in ``buf``.
        meta: This sample's metadata from :func:`tensor_metadata`.
        keys: Sorted tensor keys from :func:`sample_keys`.

    Returns:
        ``(flat, new_offset)``: the rebuilt sample and the byte offset just past it in ``buf``.
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
        cursor += nbytes + _alignment_padding_bytes(nbytes)
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
    """Build the kept-local partition, packed send buffer, and splits for one ``all_to_all_single``.

    Sender and receiver sort moves by ``(dst_slot, canonical_pos, src_local_idx)``, so the packed
    bytes and ``recv_schedule`` line up without further communication.

    Args:
        local_flats: This rank's flattened samples in local order.
        local_global_indices: Global index of each entry in ``local_flats``.
        route: ``(dst_slot, owner_rank, canonical_pos)`` per global sample index.
        all_global_indices: Global indices held by each rank, all-gathered.
        all_tensor_meta: :func:`tensor_metadata` of each rank's samples, all-gathered.
        dp_rank: This rank's module-local DP rank.
        dp_size: This module's DP size.
        window_size: Number of micro-batch slots; every ``dst_slot`` must be in ``[0, window_size)``.

    Returns:
        Dict with ``local_samples`` (``(dst_slot, canonical_pos, local_idx)`` kept here), ``send_buf``,
        per-rank byte ``send_splits`` / ``recv_splits``, and ``recv_schedule``
        (``{src_rank: [(dst_slot, canonical_pos, src_local_idx)]}`` in receive order).

    Raises:
        ValueError: If a ``dst_slot`` is outside ``[0, window_size)``.
    """
    location: Dict[int, Tuple[int, int]] = {}
    for r, gidxs in enumerate(all_global_indices):
        for i, g in enumerate(gidxs):
            location[g] = (r, i)

    local_samples: List[Tuple[int, int, int]] = []
    send_schedule: Dict[int, List[Tuple[int, int, int]]] = {r: [] for r in range(dp_size)}
    recv_schedule: Dict[int, List[Tuple[int, int, int]]] = {r: [] for r in range(dp_size)}

    local_pos = {g: i for i, g in enumerate(local_global_indices)}

    for global_idx, (dst_slot, owner_rank, canonical_pos) in enumerate(route):
        if not 0 <= dst_slot < window_size:
            raise ValueError(f"dst_slot ({dst_slot}) for global sample {global_idx} out of range [0, {window_size}).")
        src_rank, src_local_idx = location[global_idx]
        if owner_rank == src_rank:
            if owner_rank == dp_rank:
                local_samples.append((dst_slot, canonical_pos, local_pos[global_idx]))
            continue
        if src_rank == dp_rank:
            send_schedule[owner_rank].append((dst_slot, canonical_pos, src_local_idx))
        if owner_rank == dp_rank:
            recv_schedule[src_rank].append((dst_slot, canonical_pos, src_local_idx))

    # Sender and receiver must pack and decode in the same order.
    for r in range(dp_size):
        send_schedule[r].sort()
        recv_schedule[r].sort()

    # Empty placeholders must share the sample tensors' device for torch.cat / all_to_all_single.
    dev = torch.device("cpu")
    for f in local_flats:
        t = next((v for v in f.values() if isinstance(v, torch.Tensor)), None)
        if t is not None:
            dev = t.device
            break

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
# MIMO data plane: micro-batch <-> per-sample flat dicts
# ---------------------------------------------------------------------------


def _flatten_nested(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested batch dict to dotted keys, preserving tensors and ``None``."""
    flat: Dict[str, Any] = {}
    for key, val in d.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            flat.update(_flatten_nested(val, prefix=f"{path}."))
        else:
            flat[path] = val
    return flat


def _unflatten_nested(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the nested dict from dotted keys (inverse of :func:`_flatten_nested`)."""
    out: Dict[str, Any] = {}
    for path, val in flat.items():
        parts = path.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def _cu_img_from_counts(image_counts: "Any") -> List[int]:
    """Return the cumulative image-offset table ``[0, c0, c0+c1, ...]`` from per-sample image counts."""
    counts = image_counts.tolist() if isinstance(image_counts, torch.Tensor) else list(image_counts)
    cu = [0]
    for c in counts:
        cu.append(cu[-1] + int(c))
    return cu


def empty_like_vision(
    ref_hidden_states: torch.Tensor, ref_grid_thw: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return empty ``[0, d]`` hidden states and ``[0, 3]`` grid matching a reference sample's dtype and device.

    Text-only samples contribute these placeholders so every sample carries the same vision
    keys and the merge concat and byte-split metadata stay symmetric across ranks.

    Args:
        ref_hidden_states: A ``[patches, d]`` hidden-states tensor to match.
        ref_grid_thw: A ``[n_images, 3]`` grid tensor to match.

    Returns:
        ``(hidden_states, grid_thw)`` with zero rows.
    """
    empty_hidden_states = ref_hidden_states.new_empty((0, ref_hidden_states.shape[1]))
    empty_grid_thw = ref_grid_thw.new_empty((0, ref_grid_thw.shape[1]))
    return empty_hidden_states, empty_grid_thw


def patches_per_image(grid_thw: torch.Tensor) -> torch.Tensor:
    """Return the per-image patch count ``prod(t, h, w)`` for a ``[n_images, 3]`` grid.

    Args:
        grid_thw: Per-image ``(t, h, w)`` grid.

    Returns:
        ``[n_images]`` patch counts.
    """
    return torch.prod(grid_thw, dim=1)


def _reorder_vision_by_images(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    image_perm: list[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permute ``hidden_states`` (patch dim) and ``grid_thw`` (image dim) by whole image."""
    if not image_perm:
        # Avoid torch.cat([]) for a text-only sample while keeping dtype/device/requires_grad.
        return hidden_states[:0], grid_thw[:0]
    patches = patches_per_image(grid_thw)
    # One host copy of the offsets instead of a per-image .item() sync inside the loop.
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
    """Gather a vision ``{hidden_states, grid_thw}`` sub-dict by sample, supporting 0..N images per sample."""
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
    """Gather ``sample_indices`` from every tensor, vision sub-dict, and per-sample list in a batch."""
    idx = torch.as_tensor(sample_indices, dtype=torch.long)
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_dim = _batch_dim_for_tensor(key, value)
            out[key] = value.index_select(batch_dim, idx.to(value.device))
        elif isinstance(value, dict):
            if _is_patch_packed_visual_dict(value):
                out[key] = _gather_vision_subdict(value, sample_indices, n_samples, cu_img=cu_img)
            else:
                out[key] = _apply_sample_dispatch(value, sample_indices, n_samples=n_samples, cu_img=cu_img)
        elif isinstance(value, list):
            # A list of exactly n_samples entries is per-sample; any other length is global metadata.
            if len(value) == n_samples:
                out[key] = [value[i] for i in sample_indices]
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def split_microbatch(batch: Dict[str, Any], *, cu_img: "List[int] | None" = None) -> List[Dict[str, Any]]:
    """Split a micro-batch into ``B`` per-sample flat (dotted-key) dicts with size-1 batch dims.

    Args:
        batch: Nested micro-batch.
        cu_img: Cumulative per-sample image offsets (length ``B + 1``); ``None`` assumes one image per sample.

    Returns:
        ``B`` flat sample dicts in batch order.
    """
    b = _batch_size(batch)
    return [_flatten_nested(_apply_sample_dispatch(batch, [i], n_samples=b, cu_img=cu_img)) for i in range(b)]


def merge_samples(flats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rebuild a micro-batch from per-sample flat dicts (inverse of :func:`split_microbatch`).

    Args:
        flats: Per-sample flat dicts in the desired batch order.

    Returns:
        The nested micro-batch, tensors concatenated on their batch, patch, or image dim.

    Raises:
        ValueError: If ``flats`` is empty.
    """
    if not flats:
        raise ValueError("merge_samples requires at least one sample.")
    nested = [_unflatten_nested(f) for f in flats]
    return _merge_nested(nested)


def _merge_nested(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Walk the union of keys and classify a vision sub-dict from any sample: a text-only sample
    # may carry it as None, and keying off samples[0] would drop the other samples' vision.
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
        vis_ref = next((v for v in vals if _is_patch_packed_visual_dict(v)), None)
        if vis_ref is not None:
            ref_hidden_states, ref_grid_thw = vis_ref["hidden_states"], vis_ref["grid_thw"]
            hidden_parts: List[torch.Tensor] = []
            grid_parts: List[torch.Tensor] = []
            for v in vals:
                if _is_patch_packed_visual_dict(v):
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
            out[key] = rep
    return out


def _batch_to_cuda(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Move a nested batch dict's tensors to CUDA (non-blocking)."""
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
    """Return the sample count of a micro-batch from one of its LM tensors."""
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
    encoder_cost_weight: float,
    language_cost_weight: float = 0.0,
    image_token_id: "int | None" = None,
    square_merge_size: int = 1,
) -> float:
    """Return ``encoder_cost_weight * patches + language_cost_weight * real_tokens`` for one sample.

    Both terms must be collation-independent so vision and language derive the same cost without
    communicating: with ``image_token_id`` the patch count comes from ``input_ids``, which every
    module holds (``grid_thw`` is nulled on language shards), and tokens come from ``attention_mask``.

    Args:
        flat: One per-sample flat (dotted-key) dict.
        encoder_cost_weight: Weight per vision patch.
        language_cost_weight: Weight per real token; ``0.0`` disables the term.
        image_token_id: Placeholder token id counted in ``input_ids``; ``None`` sums ``grid_thw`` instead.
        square_merge_size: Patches represented by one placeholder token (``spatial_merge_size**2``).

    Returns:
        The weighted per-sample cost.
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
    if language_cost_weight:
        input_ids = flat.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.numel():
            attention_mask = flat.get("attention_mask")
            lengths = real_token_lengths(
                input_ids,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            )
            tokens = int(lengths.sum().item())

    return encoder_cost_weight * patches + language_cost_weight * tokens


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
    """Rebuild this rank's ``W`` micro-batches from kept-local and received samples.

    Args:
        local_flats: This rank's input samples; kept-local ones are indexed by ``local_idx``.
        plan: Output of :func:`prepare_sample_exchange`.
        recv_buf: The ``all_to_all_single`` output buffer.
        all_tensor_meta: All-gathered per-rank tensor metadata used to decode ``recv_buf``.
        dp_size: This module's DP size.
        window_size: Number of micro-batch slots ``W``.
        local: Expected samples per slot on this rank.

    Returns:
        ``W`` micro-batches, slot ``s`` at index ``s``, each in canonical order.

    Raises:
        RuntimeError: If the received byte count, a slot's sample count, or the canonical positions
            do not match ``plan``.
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

    if offset != recv_buf.numel():
        raise RuntimeError(f"MegatronMIMO reassembly consumed {offset} bytes but received {recv_buf.numel()}.")

    out: List[Dict[str, Any]] = []
    for slot, bucket in enumerate(buckets):
        if len(bucket) != local:
            raise RuntimeError(f"MegatronMIMO slot {slot} reassembled {len(bucket)} samples, expected {local}.")
        bucket.sort(key=lambda x: x[0])
        # Duplicate canonical positions mean a sample was dropped or duplicated in transit.
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
    """Run one ragged ``all_to_all_single`` for ``route`` and reassemble ``W`` micro-batches.

    Args:
        local_flats: This rank's input samples.
        route: ``(dst_slot, owner_rank, canonical_pos)`` per global sample index.
        all_global_indices: Global indices held by each rank, all-gathered.
        all_tensor_meta: All-gathered per-rank tensor metadata.
        dp_rank: This rank's module-local DP rank.
        dp_size: This module's DP size.
        window_size: Number of micro-batch slots ``W``.
        local: Samples per slot on this rank.
        dp_group_nccl: NCCL process group for the sample all-to-all.
        nccl_stream: Optional CUDA stream to run the all-to-all on.

    Returns:
        ``W`` micro-batches for this rank, slot ``s`` at index ``s``.
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
    """Cost-balance a window of ``W`` micro-batches across a module DP group in one exchange.

    Costs and metadata are all-gathered over Gloo once per window and the samples move in one
    ``all_to_all_single``; paired modules must pass the same ``n_groups`` and ``cost_of``.

    Args:
        batches: This rank's ``W`` micro-batch shards, each with the same local sample count.
        dp_rank: This rank's module-local DP rank.
        dp_size: This module's DP size.
        n_groups: Canonical group count (LCM of the module DP sizes).
        cost_of: ``callable(flat) -> float`` per-sample cost (see :func:`sample_cost`).
        dp_group_gloo: Gloo process group for the metadata all-gather.
        dp_group_nccl: NCCL process group for the sample all-to-all.
        nccl_stream: Optional CUDA stream to run the all-to-all on.
        image_count_of: Optional ``callable(batch) -> LongTensor`` per-sample image counts.
        probe: When ``True``, log the per-slot cost spread on ``dp_rank == 0``.

    Returns:
        ``W`` rebalanced micro-batches for this rank, slot ``s`` at index ``s``.

    Raises:
        ValueError: If ``batches`` is empty or the shards differ in local size.
    """
    import torch.distributed as dist

    window_size = len(batches)
    if window_size == 0:
        raise ValueError("exchange_window requires at least one micro-batch.")

    # Costs and metadata come from the CPU batches so the per-sample .item() calls do not
    # synchronize the training stream.
    cu_imgs = [_cu_img_from_counts(image_count_of(b)) if image_count_of is not None else None for b in batches]
    cpu_flats_per_slot = [split_microbatch(b, cu_img=cu) for b, cu in zip(batches, cu_imgs)]
    local = len(cpu_flats_per_slot[0])
    if any(len(slot) != local for slot in cpu_flats_per_slot):
        raise ValueError(
            f"All micro-batches in a window must have the same local size; got {[len(s) for s in cpu_flats_per_slot]}."
        )

    # Slot-major local order matches the g = s * B_global + j layout build_window_route assumes.
    cpu_local_flats = [f for slot in cpu_flats_per_slot for f in slot]
    local_meta = [tensor_metadata(f) for f in cpu_local_flats]
    local_costs_per_slot = [[cost_of(f) for f in slot] for slot in cpu_flats_per_slot]

    # Move the shards to the device once so serialize / all_to_all / merge stay GPU-resident and
    # the returned batches need no further copy before the forward pass.
    if torch.cuda.is_available():
        gpu_flats_per_slot = [split_microbatch(_batch_to_cuda(b), cu_img=cu) for b, cu in zip(batches, cu_imgs)]
        local_flats = [f for slot in gpu_flats_per_slot for f in slot]
    else:
        local_flats = cpu_local_flats

    payload = {"costs_per_slot": local_costs_per_slot, "meta": local_meta}
    gathered: List[Any] = [None] * dp_size
    dist.all_gather_object(gathered, payload, group=dp_group_gloo)

    b_global = local * dp_size
    all_global_indices = [
        [slot * b_global + r * local + i for slot in range(window_size) for i in range(local)] for r in range(dp_size)
    ]
    all_tensor_meta = [g["meta"] for g in gathered]
    costs_per_slot: List[List[float]] = [[0.0] * b_global for _ in range(window_size)]
    for r, g in enumerate(gathered):
        for slot in range(window_size):
            for i, c in enumerate(g["costs_per_slot"][slot]):
                costs_per_slot[slot][r * local + i] = c

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


def reorder_window_local(
    batches: List[Dict[str, Any]],
    *,
    n_groups: int,
    cost_of: "Any",
    image_count_of: "Any" = None,
) -> List[Dict[str, Any]]:
    """Permute a window into canonical order on a ``dp_size == 1`` module without any exchange.

    Args:
        batches: This rank's ``W`` whole micro-batches.
        n_groups: Canonical group count (LCM of the module DP sizes), same as the paired module.
        cost_of: ``callable(flat) -> float`` per-sample cost, same as the paired module.
        image_count_of: Optional ``callable(batch) -> LongTensor`` per-sample image counts.

    Returns:
        ``W`` micro-batches permuted into canonical order.
    """
    out: List[Dict[str, Any]] = []
    for batch in batches:
        cu_img = _cu_img_from_counts(image_count_of(batch)) if image_count_of is not None else None
        flats = split_microbatch(batch, cu_img=cu_img)
        costs = [cost_of(f) for f in flats]
        assignment = balanced_assignment(costs, n_groups, dp_size=1)
        order = sorted(range(len(flats)), key=lambda g: assignment[g][1])
        out.append(_apply_sample_dispatch(batch, order, n_samples=len(flats), cu_img=cu_img))
    return out


# ---------------------------------------------------------------------------
# Reorder buffer (data-iterator wrapper)
# ---------------------------------------------------------------------------


_SENTINEL = object()


class _WorkerException:
    """Carry an overlap-worker exception across the queue for re-raising on the consumer thread."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _record_stream_for_batch(value: Any, stream: torch.cuda.Stream) -> None:
    """Tie every CUDA tensor in ``value`` to ``stream`` before a cross-thread handoff."""
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

    ``window_size`` micro-batches are balanced in one :func:`exchange_window` call and served one
    at a time; at ``dp_size == 1`` they are permuted locally. With ``overlap=True`` a background
    thread exchanges the next window on a dedicated CUDA stream and the side ``dp_group_nccl``
    while the main thread computes the current one, so up to three windows are resident at once.
    ``CUDA_DEVICE_MAX_CONNECTIONS=1`` is rejected with ``overlap=True`` because the side
    all-to-all and the DDP all-reduce then share one hardware queue and can deadlock.

    Args:
        data_iterator: Source iterator yielding this rank's micro-batch shards.
        dp_rank: This rank's module-local DP rank.
        dp_size: This module's DP size.
        n_groups: Canonical group count (LCM of the module DP sizes).
        cost_of: ``callable(flat) -> float`` per-sample cost.
        dp_group_gloo: Gloo process group for the metadata all-gather.
        dp_group_nccl: NCCL process group for the sample all-to-all.
        overlap: Exchange the next window on a background thread.
        image_count_of: Optional ``callable(batch) -> LongTensor`` per-sample image counts.
        window_size: Micro-batches balanced per exchange.

    Raises:
        ValueError: If ``window_size < 1`` or ``overlap=True`` under ``CUDA_DEVICE_MAX_CONNECTIONS=1``.
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
        self._active: List[Dict[str, Any]] = []
        self._cursor = 0
        self._exchanges = 0
        if overlap and dp_size > 1 and os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
            # With one hardware queue the worker's all-to-all and the main thread's all-reduce run
            # in issue order, which differs per rank, so peers can each wait on a kernel that cannot
            # start. Refuse rather than degrade; overlap=False issues both from one thread.
            raise ValueError(
                "ReorderingBuffer overlap=True deadlocks under CUDA_DEVICE_MAX_CONNECTIONS=1 (single "
                "hardware queue: the side-PG all-to-all and the DDP all-reduce can wait on each other). "
                "Pass overlap=False (--no-overlap-intra-microbatch-reorder) or unset the variable."
            )
        if overlap and self._cuda and dp_size > 1:
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
        """Shut down the worker for runs that drop the iterator without closing it."""
        try:
            self.shutdown()
        except Exception:
            logger.debug("Ignoring exception while shutting down ReorderingBuffer.", exc_info=True)

    def _exchange_window(self, batches: List[Dict[str, Any]], stream: "Any") -> List[Dict[str, Any]]:
        # Probe the first few windows, then every 50th, to keep the log quiet.
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
        """Pull up to ``window_size`` micro-batches from the source iterator."""
        batches: List[Dict[str, Any]] = []
        for _ in range(self._window_size):
            try:
                batches.append(next(self._it))
            except StopIteration:
                break
        return batches

    def _refill_window(self) -> None:
        """Collect and exchange the next window synchronously; leave ``_active`` empty when exhausted."""
        batches = self._collect_window()
        if not batches:
            self._active = []
        elif self._dp_size <= 1:
            self._active = reorder_window_local(
                batches, n_groups=self._n_groups, cost_of=self._cost_of, image_count_of=self._image_count_of
            )
        else:
            self._active = self._exchange_window(batches, stream=self._stream)
        self._cursor = 0

    def _put_worker_item(self, item: Any) -> None:
        """Put an item on the bounded queue unless shutdown is requested while it is full."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _worker(self) -> None:
        # The whole exchange runs on self._stream; it is synchronized here, on the worker, so the
        # consumer receives materialized GPU batches without waiting.
        torch.cuda.set_device(self._device)
        try:
            while not self._stop.is_set():
                batches = self._collect_window()
                if not batches:
                    break
                with torch.cuda.stream(self._stream):
                    out = self._exchange_window(batches, stream=None)
                self._stream.synchronize()
                self._put_worker_item(out)
        except BaseException as exc:
            self._put_worker_item(_WorkerException(exc))
        finally:
            self._put_worker_item(_SENTINEL)

    def __next__(self) -> Dict[str, Any]:
        if self._thread is None:
            if self._cursor >= len(self._active):
                self._refill_window()
            if not self._active:
                raise StopIteration
            out = self._active[self._cursor]
            self._cursor += 1
            return out
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
    """Create the Gloo and side NCCL process groups for this rank's module DP group.

    ``dist.new_group`` is collective over the world, so every rank creates every module's groups
    in the same sorted order and keeps the ones it belongs to.

    Args:
        dp_group_nccl_main: This rank's main DP NCCL group (``grid.get_pg(["dp"])``).
        overlap: Create a separate NCCL group for the side-stream exchange; otherwise reuse the main one.

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
