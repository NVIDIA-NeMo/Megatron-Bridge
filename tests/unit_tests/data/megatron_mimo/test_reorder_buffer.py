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

"""Unit tests for the per-sample reorder-buffer building blocks (Phase 0, no process group).

The all-to-all is emulated on CPU: each rank's ``send_buf`` is split by ``send_splits`` into
per-destination chunks, and each destination's ``recv_buf`` is the concatenation of the chunks
sent to it in source-rank order — exactly the contract of ``dist.all_to_all_single``.
"""

import pytest
import torch

from megatron.bridge.data.datasets.packing_utils import balanced_index_order
from megatron.bridge.data.megatron_mimo.reorder_buffer import (
    ReorderingBuffer,
    _cu_img_from_counts,
    _reorder_vision_by_images,
    assert_intra_no_cross_slot,
    balanced_assignment,
    build_window_route,
    deserialize_sample,
    intra_route,
    merge_samples,
    prepare_sample_exchange,
    reassemble_window,
    sample_byte_size,
    sample_cost,
    sample_keys,
    serialize_sample,
    split_microbatch,
    tensor_metadata,
    window_cost_spread,
)


def _make_sample(global_idx: int, n_tokens: int, n_patches: int) -> dict:
    """A ragged synthetic MIMO-like sample, content keyed off global_idx for identity checks."""
    return {
        "input_ids": torch.arange(global_idx * 1000, global_idx * 1000 + n_tokens, dtype=torch.int64),
        "loss_mask": torch.ones(n_tokens, dtype=torch.bool),
        # vision patches: [n_patches, 4], 0 patches -> None (text-only)
        "hidden_states": (torch.full((n_patches, 4), float(global_idx), dtype=torch.float32) if n_patches else None),
    }


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_serialize_deserialize_roundtrip():
    flat = _make_sample(global_idx=7, n_tokens=5, n_patches=3)
    meta = tensor_metadata(flat)
    keys = sample_keys(meta)
    assert keys == ["hidden_states", "input_ids", "loss_mask"]  # sorted, None dropped

    buf = serialize_sample(flat, keys)
    # byte size from metadata alone must match the actual buffer
    assert buf.numel() == sample_byte_size(meta, keys)
    assert buf.numel() % 8 == 0  # 8-byte aligned

    out, offset = deserialize_sample(buf, 0, meta, keys)
    assert offset == buf.numel()
    for k in keys:
        assert torch.equal(out[k], flat[k]), k
    # None keys are restored
    assert set(out.keys()) == set(flat.keys())


def test_serialize_text_only_sample():
    flat = _make_sample(global_idx=2, n_tokens=4, n_patches=0)
    meta = tensor_metadata(flat)
    assert meta["hidden_states"] is None
    keys = sample_keys(meta)
    assert "hidden_states" not in keys
    buf = serialize_sample(flat, keys)
    out, _ = deserialize_sample(buf, 0, meta, keys)
    assert out["hidden_states"] is None
    assert torch.equal(out["input_ids"], flat["input_ids"])


def test_non_tensor_metadata_roundtrip():
    flat = {"input_ids": torch.tensor([1, 2]), "sample_id": "abc", "optional": None}
    meta = tensor_metadata(flat)
    keys = sample_keys(meta)
    assert keys == ["input_ids"]

    buf = serialize_sample(flat, keys)
    out, offset = deserialize_sample(buf, 0, meta, keys)

    assert offset == buf.numel()
    assert torch.equal(out["input_ids"], flat["input_ids"])
    assert out["sample_id"] == "abc"
    assert out["optional"] is None


def test_unknown_metadata_dtype_raises():
    meta = {"input_ids": {"shape": [2], "dtype": "torch.not_a_dtype"}}
    keys = sample_keys(meta)
    with pytest.raises(ValueError, match="Unsupported tensor dtype"):
        sample_byte_size(meta, keys)
    with pytest.raises(ValueError, match="Unsupported tensor dtype"):
        deserialize_sample(torch.empty(0, dtype=torch.uint8), 0, meta, keys)


def test_tensor_metadata_rejects_unsupported_dtype_early():
    # An out-of-map dtype (float64) must fail at metadata creation on the producing rank,
    # before the all-gather — not only later as a receiver-side byte-size/deserialize error.
    # Reuses the same _dtype_info check the decode path uses.
    flat = {"input_ids": torch.tensor([1, 2]), "weird": torch.zeros(2, dtype=torch.float64)}
    with pytest.raises(ValueError, match="Unsupported tensor dtype"):
        tensor_metadata(flat)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def test_balanced_assignment_contiguous_blocks_and_balance():
    # 8 samples, dp=2, canonical n_groups=2.
    costs = [10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0]
    n_groups, dp_size = 2, 2
    assignment = balanced_assignment(costs, n_groups, dp_size)

    n = len(costs)
    local = n // dp_size
    # each rank owns a contiguous canonical block [r*local, (r+1)*local)
    per_rank = {0: [], 1: []}
    pos_seen = set()
    for global_idx, (owner, pos) in enumerate(assignment):
        per_rank[owner].append(pos)
        pos_seen.add(pos)
    assert pos_seen == set(range(n))  # every canonical position used once
    assert sorted(per_rank[0]) == list(range(0, local))
    assert sorted(per_rank[1]) == list(range(local, n))

    # cost balance: each rank's total cost within ~max single item of the other
    rank_cost = {0: 0.0, 1: 0.0}
    for global_idx, (owner, _pos) in enumerate(assignment):
        rank_cost[owner] += costs[global_idx]
    assert abs(rank_cost[0] - rank_cost[1]) <= max(costs)


def test_balanced_assignment_matches_balanced_index_order():
    costs = [5.0, 1.0, 4.0, 2.0]
    assignment = balanced_assignment(costs, n_groups=2, dp_size=2)
    perm = balanced_index_order(costs, 2)
    # canonical_pos in the assignment must invert the perm
    pos2gidx = {pos: g for g, (_owner, pos) in enumerate(assignment)}
    assert [pos2gidx[p] for p in range(len(costs))] == perm


def test_balanced_assignment_invalid_divisibility():
    for kwargs in (
        dict(costs=[1.0, 2.0, 3.0], n_groups=2, dp_size=2),  # n_groups does not divide n
        dict(costs=[1.0, 2.0, 3.0, 4.0], n_groups=4, dp_size=3),  # dp does not divide n_groups
    ):
        try:
            balanced_assignment(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# ---------------------------------------------------------------------------
# Het-DP pairing: dp2 (vision) and dp4 (language) over the same costs/canonical groups
# ---------------------------------------------------------------------------


def test_het_dp_pairing_consistency():
    # 12 samples, canonical n_groups = max dp = 4. Vision dp2, language dp4.
    costs = [float(c) for c in [12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6]]
    n_groups = 4
    vis = balanced_assignment(costs, n_groups, dp_size=2)
    lang = balanced_assignment(costs, n_groups, dp_size=4)

    # canonical_pos must be identical across modules (same balanced order)
    for g in range(len(costs)):
        assert vis[g][1] == lang[g][1], g

    # vision rank r must cover exactly language ranks {2r, 2r+1}
    for g in range(len(costs)):
        v_owner = vis[g][0]
        l_owner = lang[g][0]
        assert l_owner // 2 == v_owner, (g, v_owner, l_owner)


# ---------------------------------------------------------------------------
# Full per-sample all-to-all (emulated collective)
# ---------------------------------------------------------------------------


def _emulate_all_to_all(plans, dp_size):
    """Emulate dist.all_to_all_single across ``dp_size`` ranks from their plans."""
    # split each source's send_buf into per-destination chunks
    chunks = {}
    for s in range(dp_size):
        buf = plans[s]["send_buf"]
        off = 0
        for d in range(dp_size):
            nbytes = plans[s]["send_splits"][d]
            chunks[(s, d)] = buf[off : off + nbytes]
            off += nbytes
        assert off == buf.numel(), f"rank {s} send_splits do not cover send_buf"

    # each destination receives chunks in source-rank order; verify split symmetry
    recv_bufs = {}
    for d in range(dp_size):
        parts = []
        for s in range(dp_size):
            assert plans[d]["recv_splits"][s] == chunks[(s, d)].numel(), (
                f"recv_splits[{d}][{s}] != bytes sent {s}->{d}"
            )
            parts.append(chunks[(s, d)])
        recv_bufs[d] = torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)
    return recv_bufs


def _run_exchange(costs, dp_size, n_groups, sample_specs):
    """End-to-end: read disjoint shards -> assign -> plan -> emulate a2a -> reconstruct.

    sample_specs[global_idx] = (n_tokens, n_patches). Initial (disjoint, contiguous) layout:
    rank r holds global indices [r*local, (r+1)*local).
    """
    n = len(costs)
    local = n // dp_size
    originals = {g: _make_sample(g, *sample_specs[g]) for g in range(n)}

    # initial disjoint layout
    rank_global_indices = [list(range(r * local, (r + 1) * local)) for r in range(dp_size)]
    rank_flats = [[originals[g] for g in rank_global_indices[r]] for r in range(dp_size)]
    all_tensor_meta = [[tensor_metadata(f) for f in rank_flats[r]] for r in range(dp_size)]

    assignment = balanced_assignment(costs, n_groups, dp_size)
    route = intra_route(assignment, src_slot=0)
    pos2gidx = {pos: g for g, (_o, pos) in enumerate(assignment)}

    plans = [
        prepare_sample_exchange(
            local_flats=rank_flats[r],
            local_global_indices=rank_global_indices[r],
            route=route,
            all_global_indices=rank_global_indices,
            all_tensor_meta=all_tensor_meta,
            dp_rank=r,
            dp_size=dp_size,
            window_size=1,
        )
        for r in range(dp_size)
    ]

    recv_bufs = _emulate_all_to_all(plans, dp_size)

    # reconstruct each rank's balanced shard
    for d in range(dp_size):
        recovered = []
        for _dst_slot, canonical_pos, local_idx in plans[d]["local_samples"]:
            recovered.append((canonical_pos, rank_flats[d][local_idx]))
        buf = recv_bufs[d]
        offset = 0
        for src in range(dp_size):
            for _dst_slot, canonical_pos, src_local_idx in plans[d]["recv_schedule"].get(src, []):
                meta = all_tensor_meta[src][src_local_idx]
                flat, offset = deserialize_sample(buf, offset, meta, sample_keys(meta))
                recovered.append((canonical_pos, flat))
        assert offset == buf.numel(), f"rank {d} did not consume entire recv_buf"

        recovered.sort(key=lambda x: x[0])
        positions = [p for p, _ in recovered]
        # rank d owns a contiguous canonical block
        assert positions == list(range(d * local, (d + 1) * local)), (d, positions)

        # tensors survived the round-trip intact, matched by global index
        for canonical_pos, flat in recovered:
            g = pos2gidx[canonical_pos]
            for k, v in originals[g].items():
                if v is None:
                    assert flat[k] is None, (g, k)
                else:
                    assert torch.equal(flat[k], v), (g, k)


def test_exchange_dp2_mixed_modality():
    # 8 samples, some text-only (0 patches), varied token/patch loads
    sample_specs = {
        0: (10, 5),
        1: (3, 0),
        2: (8, 2),
        3: (4, 0),
        4: (12, 7),
        5: (2, 1),
        6: (6, 0),
        7: (5, 3),
    }
    costs = [t + 4.0 * p for t, p in (sample_specs[g] for g in range(8))]
    _run_exchange(costs, dp_size=2, n_groups=2, sample_specs=sample_specs)


def test_exchange_dp4_canonical():
    sample_specs = {g: (3 + g, (g % 3)) for g in range(12)}
    costs = [t + 4.0 * p for t, p in (sample_specs[g] for g in range(12))]
    _run_exchange(costs, dp_size=4, n_groups=4, sample_specs=sample_specs)


# ---------------------------------------------------------------------------
# MIMO data plane: split_microbatch / merge_samples
# ---------------------------------------------------------------------------


def _make_mimo_batch(patches_per_sample):
    """Synthetic nested MIMO micro-batch: one image per sample, MRoPE position_ids."""
    b = len(patches_per_sample)
    s = 6
    # Start at 1 (not 0) so no token collides with the default pad_token_id=0: every token is real,
    # which keeps the real-token count == s for the joint-cost test.
    input_ids = torch.stack([torch.arange(1 + i * 100, 1 + i * 100 + s, dtype=torch.int64) for i in range(b)])
    labels = input_ids + 1
    loss_mask = torch.ones(b, s, dtype=torch.bool)
    position_ids = torch.stack([input_ids, input_ids, input_ids], dim=0)  # [3, B, S] MRoPE
    # one image per sample; grid [1, h, w] with prod(grid) = patches
    grids = torch.tensor([[1, 1, p] for p in patches_per_sample], dtype=torch.int64)  # [B, 3]
    hidden = torch.cat(
        [torch.full((p, 4), float(i), dtype=torch.float32) for i, p in enumerate(patches_per_sample)], dim=0
    )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": labels,
        "loss_mask": loss_mask,
        "modality_inputs": {"images": {"enc": {"hidden_states": hidden, "grid_thw": grids}}},
    }


def _assert_batch_equal(a, b):
    assert set(a.keys()) == set(b.keys())
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, dict):
            _assert_batch_equal(va, vb)
        elif va is None:
            assert vb is None, k
        else:
            assert torch.equal(va, vb), k


def test_split_merge_roundtrip_identity():
    batch = _make_mimo_batch([2, 5, 1, 3])
    samples = split_microbatch(batch)
    assert len(samples) == 4
    # per-sample flat dicts carry dotted vision keys
    assert "modality_inputs.images.enc.hidden_states" in samples[0]
    merged = merge_samples(samples)
    _assert_batch_equal(merged, batch)


# ---------------------------------------------------------------------------
# merge_samples robustness to per-sample-varying vision (PR-A)
# ---------------------------------------------------------------------------

_VKEY = "modality_inputs.images.enc"


def _flat_vision_sample(idx: int, n_tokens: int, n_patches: int, *, none_vision: bool = False) -> dict:
    """Per-sample flat dict as ``split_microbatch`` produces: ``[1, T]`` input_ids and one image's
    ``[p, 4]`` hidden_states + ``[1, 3]`` grid_thw. ``n_patches==0`` is text-only: empty ``[0, 4]`` /
    ``[0, 3]`` (or ``None`` when ``none_vision`` to exercise the legacy/None merge path)."""
    input_ids = torch.arange(idx * 100, idx * 100 + n_tokens, dtype=torch.int64).reshape(1, -1)
    if n_patches == 0:
        if none_vision:
            hidden, grid = None, None
        else:
            hidden = torch.empty((0, 4), dtype=torch.float32)
            grid = torch.empty((0, 3), dtype=torch.int64)
    else:
        hidden = torch.full((n_patches, 4), float(idx), dtype=torch.float32)
        grid = torch.tensor([[1, 1, n_patches]], dtype=torch.int64)
    return {"input_ids": input_ids, f"{_VKEY}.hidden_states": hidden, f"{_VKEY}.grid_thw": grid}


def test_merge_text_only_first():
    # Sample 0 is text-only with vision keys as None (the case the old ``ref = samples[0]``
    # classification silently dropped); samples 1-2 carry one image each.
    flats = [
        _flat_vision_sample(0, n_tokens=4, n_patches=0, none_vision=True),
        _flat_vision_sample(1, n_tokens=4, n_patches=3),
        _flat_vision_sample(2, n_tokens=4, n_patches=2),
    ]
    enc = merge_samples(flats)["modality_inputs"]["images"]["enc"]
    # 0 + 3 + 2 = 5 patches, 0 + 1 + 1 = 2 image rows — no silent loss of sample 1/2 vision.
    assert enc["hidden_states"].shape == (5, 4)
    assert enc["grid_thw"].shape == (2, 3)
    assert torch.equal(enc["hidden_states"][:3], torch.full((3, 4), 1.0))
    assert torch.equal(enc["hidden_states"][3:], torch.full((2, 4), 2.0))


def test_merge_all_text_only():
    flats = [_flat_vision_sample(i, n_tokens=4, n_patches=0) for i in range(3)]
    enc = merge_samples(flats)["modality_inputs"]["images"]["enc"]
    assert enc["hidden_states"].shape == (0, 4)
    assert enc["grid_thw"].shape == (0, 3)


def test_serialize_empty_vision_roundtrip():
    flat = _flat_vision_sample(0, n_tokens=4, n_patches=0)  # [0,4] / [0,3] empties
    meta = tensor_metadata(flat)
    keys = sample_keys(meta)
    buf = serialize_sample(flat, keys)
    assert buf.numel() == sample_byte_size(meta, keys)  # empties contribute 0 bytes
    out, offset = deserialize_sample(buf, 0, meta, keys)
    assert offset == buf.numel()
    assert out[f"{_VKEY}.hidden_states"].shape == (0, 4)
    assert out[f"{_VKEY}.grid_thw"].shape == (0, 3)
    assert torch.equal(out["input_ids"], flat["input_ids"])


# ---------------------------------------------------------------------------
# Variable images per sample: split / merge / exchange (PR-B)
# ---------------------------------------------------------------------------


def _make_mimo_batch_var(images_per_sample: list[int]) -> dict:
    """Nested MIMO micro-batch with a variable number of images per sample (0/1/N).

    ``images_per_sample[s]`` images for sample ``s``; each image gets a distinct patch count and its
    ``hidden_states`` rows are tagged with its global image index so reorder/merge can be content-checked.
    """
    b = len(images_per_sample)
    seq = 6
    input_ids = torch.stack([torch.arange(i * 100, i * 100 + seq, dtype=torch.int64) for i in range(b)])
    position_ids = torch.stack([input_ids, input_ids, input_ids], dim=0)  # [3, B, S] MRoPE
    grids: list[list[int]] = []
    hidden_blocks: list[torch.Tensor] = []
    img_gid = 0
    for n_img in images_per_sample:
        for _ in range(n_img):
            p = img_gid + 2  # distinct, identifiable patch count per image
            grids.append([1, 1, p])
            hidden_blocks.append(torch.full((p, 4), float(img_gid), dtype=torch.float32))
            img_gid += 1
    grid_thw = (
        torch.tensor(grids, dtype=torch.int64).reshape(-1, 3) if grids else torch.empty((0, 3), dtype=torch.int64)
    )
    hidden = torch.cat(hidden_blocks, dim=0) if hidden_blocks else torch.empty((0, 4), dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": input_ids + 1,
        "loss_mask": torch.ones(b, seq, dtype=torch.bool),
        "modality_inputs": {"images": {"enc": {"hidden_states": hidden, "grid_thw": grid_thw}}},
    }


def test_cu_img_from_counts():
    assert _cu_img_from_counts([2, 0, 1, 3]) == [0, 2, 2, 3, 6]
    assert _cu_img_from_counts(torch.tensor([2, 0, 1, 3])) == [0, 2, 2, 3, 6]
    assert _cu_img_from_counts([0, 0]) == [0, 0, 0]


def test_image_count_sourcing():
    vision_start = 248053
    input_ids = torch.tensor(
        [
            [vision_start, 5, vision_start, 6, 7, 8],  # 2 images
            [1, 2, 3, 4, 5, 6],  # text-only
            [vision_start, 9, 9, 9, 9, 9],  # 1 image
        ],
        dtype=torch.int64,
    )
    counts = (input_ids == vision_start).sum(dim=1)
    assert counts.tolist() == [2, 0, 1]
    batch = _make_mimo_batch_var([2, 0, 1])
    assert int(counts.sum()) == batch["modality_inputs"]["images"]["enc"]["grid_thw"].shape[0]


def test_split_merge_roundtrip_multi_image():
    images_per_sample = [2, 0, 1, 3]
    batch = _make_mimo_batch_var(images_per_sample)
    samples = split_microbatch(batch, cu_img=_cu_img_from_counts(images_per_sample))
    assert len(samples) == 4
    merged = merge_samples(samples)
    _assert_batch_equal(merged, batch)


def test_split_all_text_only():
    batch = _make_mimo_batch_var([0, 0])
    samples = split_microbatch(batch, cu_img=_cu_img_from_counts([0, 0]))
    for smp in samples:
        assert smp[f"{_VKEY}.hidden_states"].shape == (0, 4)
        assert smp[f"{_VKEY}.grid_thw"].shape == (0, 3)
    _assert_batch_equal(merge_samples(samples), batch)


def test_soft_validation_mismatch():
    batch = _make_mimo_batch_var([1, 1])  # 2 images
    with pytest.raises(ValueError, match="mis-sourced"):
        split_microbatch(batch, cu_img=[0, 1, 5])  # sums to 5 != 2 images


def test_split_legacy_no_cu_img():
    # one-image-per-sample works with cu_img=None (legacy path)
    batch = _make_mimo_batch([2, 5, 1, 3])
    _assert_batch_equal(merge_samples(split_microbatch(batch)), batch)
    # multi-image without cu_img raises a clear error
    multi = _make_mimo_batch_var([2, 1])  # 3 images, 2 samples
    with pytest.raises(ValueError, match="one image per sample"):
        split_microbatch(multi)


def _run_exchange_nested(images_per_sample, dp_size, n_groups):
    """Full split→assign→exchange→reconstruct on nested variable-image batches (emulated a2a)."""
    n = len(images_per_sample)
    local = n // dp_size
    full = _make_mimo_batch_var(images_per_sample)
    all_flats = split_microbatch(full, cu_img=_cu_img_from_counts(images_per_sample))
    costs = [sample_cost(f, linear_vit=1.0) for f in all_flats]

    rank_global_indices = [list(range(r * local, (r + 1) * local)) for r in range(dp_size)]
    rank_flats = [[all_flats[g] for g in rank_global_indices[r]] for r in range(dp_size)]
    all_tensor_meta = [[tensor_metadata(f) for f in rank_flats[r]] for r in range(dp_size)]
    assignment = balanced_assignment(costs, n_groups, dp_size)
    route = intra_route(assignment, src_slot=0)
    pos2gidx = {pos: g for g, (_o, pos) in enumerate(assignment)}

    plans = [
        prepare_sample_exchange(
            local_flats=rank_flats[r],
            local_global_indices=rank_global_indices[r],
            route=route,
            all_global_indices=rank_global_indices,
            all_tensor_meta=all_tensor_meta,
            dp_rank=r,
            dp_size=dp_size,
            window_size=1,
        )
        for r in range(dp_size)
    ]
    recv_bufs = _emulate_all_to_all(plans, dp_size)

    for d in range(dp_size):
        recovered = [(pos, rank_flats[d][li]) for _slot, pos, li in plans[d]["local_samples"]]
        buf = recv_bufs[d]
        offset = 0
        for src in range(dp_size):
            for _dst_slot, canonical_pos, src_local_idx in plans[d]["recv_schedule"].get(src, []):
                meta = all_tensor_meta[src][src_local_idx]
                flat, offset = deserialize_sample(buf, offset, meta, sample_keys(meta))
                recovered.append((canonical_pos, flat))
        assert offset == buf.numel(), (d, offset, buf.numel())
        recovered.sort(key=lambda x: x[0])
        assert [p for p, _ in recovered] == list(range(d * local, (d + 1) * local)), d
        for canonical_pos, flat in recovered:
            g = pos2gidx[canonical_pos]
            for k, v in all_flats[g].items():
                if v is None:
                    assert flat[k] is None, (g, k)
                else:
                    assert torch.equal(flat[k], v), (g, k)


def test_exchange_var_images_dp2():
    # 8 samples: text-only (0), single (1), and multi-image (2/3) mixed so a rank owns a variety.
    _run_exchange_nested([2, 0, 1, 0, 3, 1, 0, 2], dp_size=2, n_groups=2)


# ---------------------------------------------------------------------------
# 3D routing seam + per-slot reassembly (reassemble_window): Phase-1 gates §10.1.1 / §10.1.2
# ---------------------------------------------------------------------------


def _emulate_window_exchange(rank_flats, rank_global_indices, all_tensor_meta, route, *, dp_size, window_size, local):
    """Emulate the on-device exchange on CPU and reassemble each rank's window via the production
    :func:`reassemble_window`. Returns ``per_rank_batches[d][slot]``."""
    plans = [
        prepare_sample_exchange(
            local_flats=rank_flats[r],
            local_global_indices=rank_global_indices[r],
            route=route,
            all_global_indices=rank_global_indices,
            all_tensor_meta=all_tensor_meta,
            dp_rank=r,
            dp_size=dp_size,
            window_size=window_size,
        )
        for r in range(dp_size)
    ]
    recv_bufs = _emulate_all_to_all(plans, dp_size)
    return [
        reassemble_window(
            rank_flats[d],
            plans[d],
            recv_bufs[d],
            all_tensor_meta,
            dp_size=dp_size,
            window_size=window_size,
            local=local,
        )
        for d in range(dp_size)
    ]


def test_reassemble_window_w1_matches_canonical_merge():
    """§10.1.1: with the intra route and W=1, ``reassemble_window`` rebuilds exactly the canonically
    ordered balanced micro-batch — byte-identical to merging the globally-balanced samples directly."""
    images_per_sample = [2, 0, 1, 0, 3, 1, 0, 2]
    dp_size, n_groups = 2, 2
    n = len(images_per_sample)
    local = n // dp_size

    full = _make_mimo_batch_var(images_per_sample)
    all_flats = split_microbatch(full, cu_img=_cu_img_from_counts(images_per_sample))
    costs = [sample_cost(f, linear_vit=1.0) for f in all_flats]
    assignment = balanced_assignment(costs, n_groups, dp_size)
    route = intra_route(assignment, src_slot=0)
    pos2gidx = {pos: g for g, (_o, pos) in enumerate(assignment)}

    rank_global_indices = [list(range(r * local, (r + 1) * local)) for r in range(dp_size)]
    rank_flats = [[all_flats[g] for g in rank_global_indices[r]] for r in range(dp_size)]
    all_tensor_meta = [[tensor_metadata(f) for f in rank_flats[r]] for r in range(dp_size)]

    per_rank = _emulate_window_exchange(
        rank_flats, rank_global_indices, all_tensor_meta, route, dp_size=dp_size, window_size=1, local=local
    )

    for d in range(dp_size):
        assert len(per_rank[d]) == 1  # W=1 -> one micro-batch
        owned_positions = sorted(pos for g, (owner, pos) in enumerate(assignment) if owner == d)
        expected = merge_samples([all_flats[pos2gidx[p]] for p in owned_positions])
        _assert_batch_equal(per_rank[d][0], expected)


def test_reassemble_window_w2_3d_route():
    """§10.1.2: a hand-built W=2 3D route (no balancer) routes samples into specific slots; assert each
    slot reassembles its sample, a text-only sample lands as an empty-vision slot, vision is intact."""
    # 4 samples, dp=2, W=2 slots, local=1 per (slot, rank). Layout: rank0 holds globals [0,1],
    # rank1 holds [2,3]. g2 is text-only (0 patches); others carry distinct image patch counts.
    dp_size, window_size, local = 2, 2, 1
    originals = [
        _flat_vision_sample(0, n_tokens=4, n_patches=3),
        _flat_vision_sample(1, n_tokens=4, n_patches=2),
        _flat_vision_sample(2, n_tokens=4, n_patches=0),  # text-only
        _flat_vision_sample(3, n_tokens=4, n_patches=1),
    ]
    rank_global_indices = [[0, 1], [2, 3]]
    rank_flats = [[originals[g] for g in rank_global_indices[r]] for r in range(dp_size)]
    all_tensor_meta = [[tensor_metadata(f) for f in rank_flats[r]] for r in range(dp_size)]

    # route[g] = (dst_slot, owner_rank, canonical_pos). Cross-rank moves into a chosen slot:
    #   g0 -> slot0,rank0   g1 -> slot1,rank1   g2 -> slot0,rank1   g3 -> slot1,rank0
    route = [(0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)]

    per_rank = _emulate_window_exchange(
        rank_flats, rank_global_indices, all_tensor_meta, route, dp_size=dp_size, window_size=window_size, local=local
    )

    def first_token(batch):
        return int(batch["input_ids"][0, 0].item())

    # rank0: slot0 <- g0, slot1 <- g3 (received).  rank1: slot0 <- g2 (text-only), slot1 <- g1.
    assert [first_token(b) for b in per_rank[0]] == [0, 300]
    assert [first_token(b) for b in per_rank[1]] == [200, 100]

    # Vision attribution per slot: g3 (1 patch) on rank0 slot1; g2 (text-only -> [0,4]) on rank1 slot0.
    r0_slot1_enc = per_rank[0][1]["modality_inputs"]["images"]["enc"]
    assert r0_slot1_enc["hidden_states"].shape == (1, 4)
    assert torch.equal(r0_slot1_enc["hidden_states"], torch.full((1, 4), 3.0))
    r1_slot0_enc = per_rank[1][0]["modality_inputs"]["images"]["enc"]
    assert r1_slot0_enc["hidden_states"].shape == (0, 4)


def test_reassemble_window_slot_shape_guard():
    """The §10.0 slot-shape invariant fires loudly when a slot ends up with the wrong sample count."""
    # Route both rank0's samples into the same slot of rank0 -> slot0 has 2, slot1 has 0 (expected 1).
    dp_size, window_size, local = 2, 2, 1
    originals = [_flat_vision_sample(i, n_tokens=4, n_patches=1) for i in range(4)]
    rank_global_indices = [[0, 1], [2, 3]]
    rank_flats = [[originals[g] for g in rank_global_indices[r]] for r in range(dp_size)]
    all_tensor_meta = [[tensor_metadata(f) for f in rank_flats[r]] for r in range(dp_size)]
    # g0,g1 both -> slot0/rank0; g2,g3 -> slot0/rank1. Slot0 over-fills (2) -> guard fires on slot 0.
    route = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    with pytest.raises(RuntimeError, match="slot 0 reassembled 2 samples, expected 1"):
        _emulate_window_exchange(
            rank_flats,
            rank_global_indices,
            all_tensor_meta,
            route,
            dp_size=dp_size,
            window_size=window_size,
            local=local,
        )


# ---------------------------------------------------------------------------
# Window route (build_window_route) + multi-slot window exchange: Phase-2 gates
# ---------------------------------------------------------------------------


def test_build_window_route_layout():
    """The window route is slot-major: route[s*B + j] has dst_slot == s and a per-slot canonical_pos."""
    dp_size, n_groups = 2, 2
    costs_per_slot = [[4.0, 1.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]]
    b_global = len(costs_per_slot[0])
    route = build_window_route(costs_per_slot, n_groups, dp_size)
    assert len(route) == len(costs_per_slot) * b_global
    for s in range(len(costs_per_slot)):
        slot_entries = route[s * b_global : (s + 1) * b_global]
        # intra invariant: every entry in this slot's block routes to slot s.
        assert all(dst == s for dst, _owner, _pos in slot_entries)
        # canonical positions are a permutation of [0, B) within the slot.
        assert sorted(pos for _dst, _owner, pos in slot_entries) == list(range(b_global))
        # matches an independent per-slot intra_route.
        assert slot_entries == intra_route(balanced_assignment(costs_per_slot[s], n_groups, dp_size), src_slot=s)


def _run_window_exchange(images_per_slot, dp_size, n_groups):
    """Full window exchange over W micro-batches (emulated a2a) using the real window indexing
    (build_window_route + reassemble_window). Asserts each slot reassembles to exactly the
    independent per-slot intra-balanced shard — i.e. slots never mix (intra invariant)."""
    window_size = len(images_per_slot)
    # Per-slot full batches + flats + costs (slot-local global index j in [0, B)).
    slot_full = [_make_mimo_batch_var(imgs) for imgs in images_per_slot]
    slot_flats = [
        split_microbatch(slot_full[s], cu_img=_cu_img_from_counts(images_per_slot[s])) for s in range(window_size)
    ]
    b_global = len(slot_flats[0])
    local = b_global // dp_size
    costs_per_slot = [[sample_cost(f, linear_vit=1.0) for f in slot_flats[s]] for s in range(window_size)]

    # Window-local layout: rank r holds slot-major [slot0 shard, slot1 shard, ...].
    rank_global_indices = [
        [s * b_global + r * local + i for s in range(window_size) for i in range(local)] for r in range(dp_size)
    ]
    rank_flats = [
        [slot_flats[s][r * local + i] for s in range(window_size) for i in range(local)] for r in range(dp_size)
    ]
    all_tensor_meta = [[tensor_metadata(f) for f in rank_flats[r]] for r in range(dp_size)]

    route = build_window_route(costs_per_slot, n_groups, dp_size)
    per_rank = _emulate_window_exchange(
        rank_flats, rank_global_indices, all_tensor_meta, route, dp_size=dp_size, window_size=window_size, local=local
    )

    # Expected: each slot independently balanced (W=1 semantics applied per slot).
    for s in range(window_size):
        assignment_s = balanced_assignment(costs_per_slot[s], n_groups, dp_size)
        pos2gidx_s = {pos: g for g, (_o, pos) in enumerate(assignment_s)}
        for d in range(dp_size):
            owned = sorted(pos for g, (owner, pos) in enumerate(assignment_s) if owner == d)
            expected = merge_samples([slot_flats[s][pos2gidx_s[p]] for p in owned])
            _assert_batch_equal(per_rank[d][s], expected)


def test_window_exchange_two_slots_dp2():
    # W=2 window, dp=2; distinct image mixes per slot so each slot triggers a real cross-rank swap.
    _run_window_exchange([[2, 0, 1, 3], [1, 2, 0, 1]], dp_size=2, n_groups=2)


def test_window_exchange_w3_dp2_text_only_and_multi():
    # W=3 window with text-only + multi-image samples; proves §4a vision attribution per slot at W>1.
    _run_window_exchange([[0, 2, 1, 0], [3, 0, 0, 1], [1, 1, 1, 1]], dp_size=2, n_groups=2)


def test_window_exchange_het_dp_paired_slots():
    # Het-DP at the window level: dp2 and dp4 over the same per-slot costs must agree per slot.
    images_per_slot = [[2, 0, 1, 3, 1, 2, 0, 1], [1, 1, 2, 0, 3, 0, 1, 2]]
    for dp_size in (2, 4):
        _run_window_exchange(images_per_slot, dp_size=dp_size, n_groups=4)


def test_window_cost_spread_tightens_after_balance():
    """§10.5 balance probe: the per-rank cost spread must tighten (and never widen) after balancing —
    the quantitative evidence that reordering evens the per-rank load (§10.3)."""
    dp_size, n_groups = 4, 4
    # Heavily skewed costs: the natural contiguous shards are very imbalanced (rank0 holds the big
    # samples, rank3 the small), so balancing should shrink the per-rank max/min spread.
    # Both slots are skewed across the contiguous shards (rank0 heavy, rank1 light) so each must
    # move samples and strictly tighten — slot 1 differs from slot 0 to exercise per-slot balancing.
    costs_per_slot = [
        [100.0, 90.0, 80.0, 70.0, 8.0, 6.0, 4.0, 2.0, 50.0, 40.0, 30.0, 20.0, 9.0, 7.0, 5.0, 3.0],
        [80.0, 75.0, 70.0, 65.0, 10.0, 9.0, 8.0, 7.0, 45.0, 40.0, 35.0, 30.0, 12.0, 11.0, 10.0, 9.0],
    ]
    route = build_window_route(costs_per_slot, n_groups, dp_size)
    spreads = window_cost_spread(costs_per_slot, route, dp_size)
    assert len(spreads) == 2
    for sp in spreads:
        before = sp["before_max"] - sp["before_min"]
        after = sp["after_max"] - sp["after_min"]
        assert after <= before, sp  # balancing never widens the spread
        assert after < before  # and on these skewed inputs it strictly tightens
    # these skewed inputs require real movement: 0 < remote <= slot size.
    assert all(0 < sp["remote"] <= len(costs_per_slot[0]) for sp in spreads)


def test_assert_intra_no_cross_slot_passes_for_window_route():
    # §10.0 guard: build_window_route output (intra) always satisfies dst_slot == src_slot.
    dp_size, n_groups = 2, 2
    costs_per_slot = [[4.0, 1.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]]
    route = build_window_route(costs_per_slot, n_groups, dp_size)  # calls the guard internally
    assert_intra_no_cross_slot(route, b_global=4)  # explicit re-check is a no-op (no raise)


def test_assert_intra_no_cross_slot_trips_on_cross_slot_route():
    # A hand-built route that sends a slot-0 sample (g=1 -> src_slot 0) into slot 1 must trip the guard.
    b_global = 2  # 2 slots of 2 samples; g//b_global is the source slot
    route = [(0, 0, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)]  # g=1 leaks slot0 -> slot1
    with pytest.raises(RuntimeError, match="cross-slot leak at global index 1"):
        assert_intra_no_cross_slot(route, b_global=b_global)


def test_window_cost_spread_already_balanced_no_remote():
    # Uniform costs are already perfectly balanced: spread stays 0 and nothing needs to move.
    dp_size, n_groups = 2, 2
    costs_per_slot = [[5.0, 5.0, 5.0, 5.0]]
    route = build_window_route(costs_per_slot, n_groups, dp_size)
    (sp,) = window_cost_spread(costs_per_slot, route, dp_size)
    assert sp["before_max"] == sp["before_min"] == sp["after_max"] == sp["after_min"] == 10.0


# ---------------------------------------------------------------------------
# ReorderingBuffer window cursor (dp_size == 1 passthrough: collection/serve logic, no PG)
# ---------------------------------------------------------------------------


def test_reordering_buffer_window_cursor_passthrough():
    items = [{"input_ids": torch.tensor([[i]])} for i in range(7)]
    buf = ReorderingBuffer(
        iter(items),
        dp_rank=0,
        dp_size=1,  # single rank -> passthrough, no exchange / no PG needed
        n_groups=1,
        cost_of=lambda f: 0.0,
        dp_group_gloo=None,
        dp_group_nccl=None,
        overlap=False,
        window_size=3,  # windows of [3, 3, 1] over 7 items
    )
    got = [int(b["input_ids"].item()) for b in buf]
    assert got == list(range(7))  # all items served, in order, short final window handled


def test_reordering_buffer_overlap_single_rank_no_thread():
    # overlap=True but dp_size=1: no prefetch thread (nothing to exchange), pure passthrough.
    items = [{"input_ids": torch.tensor([[i]])} for i in range(5)]
    buf = ReorderingBuffer(
        iter(items),
        dp_rank=0,
        dp_size=1,
        n_groups=1,
        cost_of=lambda f: 0.0,
        dp_group_gloo=None,
        dp_group_nccl=None,
        overlap=True,
        window_size=2,
    )
    assert buf._thread is None  # dp_size <= 1 never starts the prefetch thread
    assert [int(b["input_ids"].item()) for b in buf] == list(range(5))


def test_reordering_buffer_window_size_validation():
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        ReorderingBuffer(
            iter([]),
            dp_rank=0,
            dp_size=1,
            n_groups=1,
            cost_of=lambda f: 0.0,
            dp_group_gloo=None,
            dp_group_nccl=None,
            overlap=False,
            window_size=0,
        )


# ---------------------------------------------------------------------------
# Joint per-sample cost
# ---------------------------------------------------------------------------


def test_sample_cost_patches_only():
    from megatron.bridge.data.megatron_mimo.reorder_buffer import sample_cost

    batch = _make_mimo_batch([2, 5])  # sample 0: 2 patches, sample 1: 5 patches; 6 tokens each
    samples = split_microbatch(batch)
    # default linear_lm=0.0 -> patch-only cost: linear_vit * sum(prod(grid_thw)); token length not counted
    c0 = sample_cost(samples[0], linear_vit=1.0)
    c1 = sample_cost(samples[1], linear_vit=1.0)
    assert c0 == 2
    assert c1 == 5
    # linear_vit scales the patch count
    assert sample_cost(samples[0], linear_vit=3.0) == 3 * 2


def test_sample_cost_text_only_is_zero():
    from megatron.bridge.data.megatron_mimo.reorder_buffer import sample_cost

    flat = {"input_ids": torch.tensor([[5, 5, 0, 0]], dtype=torch.int64)}  # no vision -> zero cost
    c = sample_cost(flat, linear_vit=1.0)
    assert c == 0


def test_sample_cost_joint_vit_and_lm():
    from megatron.bridge.data.megatron_mimo.reorder_buffer import sample_cost

    # _make_mimo_batch uses input_ids = arange(i*100, i*100+6); with pad_token_id=0 every token is
    # non-pad, so each sample has 6 real tokens. attention_mask is None -> != pad_token_id fallback.
    batch = _make_mimo_batch([2, 5])  # patches 2 and 5; 6 real tokens each
    samples = split_microbatch(batch)
    # cost = linear_vit*patches + linear_lm*real_tokens
    assert sample_cost(samples[0], linear_vit=1.0, linear_lm=0.5) == 2 + 0.5 * 6
    assert sample_cost(samples[1], linear_vit=1.0, linear_lm=0.5) == 5 + 0.5 * 6
    # linear_vit=0 -> token-only cost
    assert sample_cost(samples[0], linear_vit=0.0, linear_lm=2.0) == 2.0 * 6


def test_sample_cost_lm_term_ignores_padding():
    from megatron.bridge.data.megatron_mimo.reorder_buffer import sample_cost

    # Same 3 real tokens but different padded widths -> identical LM cost (collation-independent).
    short = {"input_ids": torch.tensor([[5, 6, 7, 0]], dtype=torch.int64)}
    long = {"input_ids": torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0]], dtype=torch.int64)}
    assert sample_cost(short, linear_vit=1.0, linear_lm=1.0) == 3
    assert sample_cost(long, linear_vit=1.0, linear_lm=1.0) == 3


def test_sample_cost_lm_term_prefers_attention_mask():
    from megatron.bridge.data.megatron_mimo.reorder_buffer import sample_cost

    # attention_mask is authoritative even when token ids coincide with pad_token_id.
    flat = {
        "input_ids": torch.tensor([[5, 0, 7, 0]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.int64),
    }
    assert sample_cost(flat, linear_vit=1.0, linear_lm=1.0, pad_token_id=0) == 3


def test_sample_cost_image_token_id_recovers_patches_from_input_ids():
    from megatron.bridge.data.megatron_mimo.reorder_buffer import sample_cost

    img = 99  # image-placeholder token id
    # 3 placeholder tokens * square_merge_size(4) = 12 patches.
    flat = {"input_ids": torch.tensor([[1, img, img, img, 2, 0]], dtype=torch.int64)}
    assert sample_cost(flat, linear_vit=1.0, image_token_id=img, square_merge_size=4) == 12
    # linear_vit scales the recovered patch count.
    assert sample_cost(flat, linear_vit=2.0, image_token_id=img, square_merge_size=4) == 24


def test_sample_cost_image_token_id_matches_across_modules():
    """Bug #1: vision (has grid_thw) and language (grid_thw nulled) must derive the SAME cost.

    With image_token_id wired, the patch cost comes from the module-independent input_ids image-token
    count, so a vision-style flat and a language-style flat with identical input_ids cost the same even
    though the language flat has no grid_thw (nulled by the #4442 rank-aware metadata collate).
    """
    from megatron.bridge.data.megatron_mimo.reorder_buffer import sample_cost

    img = 99
    input_ids = torch.tensor([[1, img, img, 2, 0, 0]], dtype=torch.int64)  # 2 image tokens
    # Vision-style flat: carries grid_thw (would dominate the legacy path).
    vision_flat = {
        "input_ids": input_ids,
        f"{_VKEY}.grid_thw": torch.tensor([[1, 4, 2]], dtype=torch.int64),  # prod = 8 patches
    }
    # Language-style flat: grid_thw nulled, only input_ids remains.
    language_flat = {"input_ids": input_ids}

    vc = sample_cost(vision_flat, linear_vit=1.0, image_token_id=img, square_merge_size=4)
    lc = sample_cost(language_flat, linear_vit=1.0, image_token_id=img, square_merge_size=4)
    assert vc == lc == 2 * 4  # 2 image tokens * square_merge_size, independent of grid_thw
    # Without image_token_id the two diverge (the bug): vision reads grid_thw, language gets 0.
    assert sample_cost(vision_flat, linear_vit=1.0) == 8
    assert sample_cost(language_flat, linear_vit=1.0) == 0


# ---------------------------------------------------------------------------
# F14 — single D2H on the vision reorder path (Task 4.3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReorderVisionSingleD2H:
    def test_reorder_vision_single_d2h(self, monkeypatch):
        # Count Tensor.item() calls during the reorder; the cumsum should be a single .tolist().
        calls = {"item": 0}
        orig_item = torch.Tensor.item

        def counting_item(self):
            calls["item"] += 1
            return orig_item(self)

        grid = torch.tensor([[1, 2, 2], [1, 4, 4], [1, 2, 3]])  # patches 4,16,6
        hidden = torch.arange(26, dtype=torch.float32).reshape(26, 1)
        monkeypatch.setattr(torch.Tensor, "item", counting_item)
        hs, g = _reorder_vision_by_images(hidden, grid, [2, 0, 1])
        # no per-image .item() D2H sync in the loop (cumsum uses one .tolist()).
        assert calls["item"] == 0
        # output bit-identical to the expected reorder
        expected = torch.cat([hidden[20:26], hidden[0:4], hidden[4:20]])
        torch.testing.assert_close(hs, expected)
