# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Functional test for the MegatronMIMO intra-microbatch reorder exchange on a real 2-rank
process group.

The CPU unit tests (``tests/unit_tests/data/megatron_mimo/test_reorder_buffer.py``) emulate the
``all_to_all_single``; this exercises the on-device path that those cannot — the real Gloo cost
all-gather + NCCL ragged ``all_to_all_single`` inside :func:`exchange_window`, plus the side-PG
setup in :func:`build_module_dp_process_groups`. The 2-rank world is treated as a single module's
``dp=2`` group (het-DP pairing needs two modules / 4 ranks and stays unit-emulated).

Run by ``tests/functional_tests/launch_scripts/h100/active/L0_Launch_training_megatron_mimo.sh``
(``torch.distributed.run --nproc_per_node=2 -m pytest .../test_groups/megatron_mimo``).
"""

from __future__ import annotations

import functools
from typing import Any, List, Tuple

import pytest
import torch
import torch.distributed as dist

from megatron.bridge.data.megatron_mimo.reorder_buffer import (
    balanced_assignment,
    build_module_dp_process_groups,
    exchange_window,
    sample_cost,
)
from tests.functional_tests.utils import initialize_distributed


_VISION_START = 990  # synthetic vision-start token; one per image, counted by image_count_of
_SEQ = 8
_DIM = 4


def _make_shard(samples: List[Tuple[int, List[int]]]) -> dict:
    """Build one rank's nested micro-batch from ``[(global_idx, [patches_per_image, ...]), ...]``.

    ``input_ids[i]`` is tagged with ``1 + g*100`` (so the global index is recoverable as
    ``(first_token - 1)//100``) and carries one ``_VISION_START`` token per image. Each image
    contributes ``p`` ``hidden_states`` rows all filled with ``float(g)`` so vision can be traced to
    its owning sample; a text-only sample (no images) contributes empty ``[0, d]`` / ``[0, 3]``.
    """
    b = len(samples)
    rows = []
    grids: List[List[int]] = []
    hidden_blocks: List[torch.Tensor] = []
    for g, patches in samples:
        row = [7] * _SEQ
        row[0] = 1 + g * 100
        for k in range(len(patches)):
            row[1 + k] = _VISION_START
        rows.append(row)
        for p in patches:
            grids.append([1, 1, p])
            hidden_blocks.append(torch.full((p, _DIM), float(g), dtype=torch.float32))
    input_ids = torch.tensor(rows, dtype=torch.int64)
    grid_thw = (
        torch.tensor(grids, dtype=torch.int64).reshape(-1, 3) if grids else torch.empty((0, 3), dtype=torch.int64)
    )
    hidden = torch.cat(hidden_blocks, dim=0) if hidden_blocks else torch.empty((0, _DIM), dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "position_ids": torch.stack([input_ids, input_ids, input_ids], dim=0),  # [3, B, S] MRoPE
        "attention_mask": None,
        "labels": input_ids + 1,
        "loss_mask": torch.ones(b, _SEQ, dtype=torch.bool),
        "modality_inputs": {"images": {"enc": {"hidden_states": hidden, "grid_thw": grid_thw}}},
    }


def _expected_owned(global_patches: List[List[int]], dp_rank: int, dp_size: int, n_groups: int) -> List[int]:
    """Global indices this rank should own after the exchange, in canonical-position order."""
    costs = [float(sum(p)) for p in global_patches]
    assignment = balanced_assignment(costs, n_groups, dp_size)
    owned = sorted((pos, g) for g, (owner, pos) in enumerate(assignment) if owner == dp_rank)
    return [g for _pos, g in owned]


def _run_exchange_case(global_samples: List[List[int]], image_count_of: "Any") -> None:
    """Shard ``global_samples`` (patch lists per global sample) contiguously across 2 ranks, run the
    real exchange, and assert this rank recovers exactly its cost-balanced shard with vision intact.
    """
    dp_size = dist.get_world_size()
    n_groups = dp_size
    b = len(global_samples)
    local = b // dp_size
    rank = dist.get_rank()

    my_globals = list(range(rank * local, (rank + 1) * local))
    batch = _make_shard([(g, global_samples[g]) for g in my_globals])

    main_pg = dist.new_group(ranks=list(range(dp_size)), backend="nccl")
    dp_rank, ds, gloo, nccl = build_module_dp_process_groups(main_pg, overlap=False)
    assert (ds, dp_rank) == (dp_size, rank)

    out = exchange_window(
        [batch],
        dp_rank=dp_rank,
        dp_size=ds,
        n_groups=n_groups,
        cost_of=functools.partial(sample_cost, linear_vit=1.0),
        dp_group_gloo=gloo,
        dp_group_nccl=nccl,
        image_count_of=image_count_of,
    )[0]

    expected_globals = _expected_owned(global_samples, dp_rank, dp_size, n_groups)

    got_globals = [int((int(row[0].item()) - 1) // 100) for row in out["input_ids"].cpu()]
    assert got_globals == expected_globals, (rank, got_globals, expected_globals)

    # Vision must travel with its sample through the real all-to-all: reconstruct the expected
    # concatenation (per-sample patch blocks tagged float(g)) and compare byte-for-byte.
    exp_hidden = [torch.full((sum(global_samples[g]), _DIM), float(g), dtype=torch.float32) for g in expected_globals]
    exp_grid = [
        torch.tensor([[1, 1, p] for p in global_samples[g]], dtype=torch.int64).reshape(-1, 3)
        for g in expected_globals
    ]
    exp_hidden_t = torch.cat(exp_hidden, dim=0) if exp_hidden else torch.empty((0, _DIM), dtype=torch.float32)
    exp_grid_t = torch.cat(exp_grid, dim=0) if exp_grid else torch.empty((0, 3), dtype=torch.int64)

    enc = out["modality_inputs"]["images"]["enc"]
    assert torch.equal(enc["hidden_states"].cpu().float(), exp_hidden_t), rank
    assert torch.equal(enc["grid_thw"].cpu(), exp_grid_t), rank


@pytest.mark.run_only_on("GPU")
def test_reorder_exchange_dp2_single_image():
    """One image per sample; per-sample patch counts force a real cross-rank swap."""
    initialize_distributed()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this functional test")
    if dist.get_world_size() != 2:
        pytest.skip("This functional test requires exactly 2 ranks")

    # costs [4,3,1,2]: balanced owner is g0,g2 -> rank0 and g1,g3 -> rank1, but the contiguous
    # initial shards are [g0,g1] / [g2,g3], so rank0 must SEND g1 and RECEIVE g2 (a real exchange).
    global_samples = [[4], [3], [1], [2]]
    image_count_of = lambda b: (b["input_ids"] == _VISION_START).sum(dim=1).to(torch.long)  # noqa: E731
    _run_exchange_case(global_samples, image_count_of)
    dist.barrier()


@pytest.mark.run_only_on("GPU")
def test_reorder_exchange_dp2_variable_images():
    """Mixed image counts per sample (text-only / single / multi) survive the real exchange."""
    initialize_distributed()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this functional test")
    if dist.get_world_size() != 2:
        pytest.skip("This functional test requires exactly 2 ranks")

    # counts [2,1,0,1] (g2 text-only, g0 multi-image), patches=2/image -> costs [4,2,0,2]: owner is
    # g0,g2 -> rank0 and g1,g3 -> rank1; initial shards [g0,g1] / [g2,g3], so a multi-image sample
    # stays, a single-image (g1) and a TEXT-ONLY (g2) sample swap across ranks.
    global_samples = [[2, 2], [2], [], [2]]
    image_count_of = lambda b: (b["input_ids"] == _VISION_START).sum(dim=1).to(torch.long)  # noqa: E731
    _run_exchange_case(global_samples, image_count_of)
    dist.barrier()
