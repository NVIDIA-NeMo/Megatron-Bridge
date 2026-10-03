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

"""MegatronMIMO intra-microbatch reorder exchange on a real 2-rank process group (NCCL + Gloo side PGs).

Run by ``tests/functional_tests/launch_scripts/h100/active/L0_Launch_training_megatron_mimo.sh``
(``torch.distributed.run --nproc_per_node=2 -m pytest .../test_groups/megatron_mimo``)."""

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


_VISION_START = 990  # one per image; counted by image_count_of
_SEQ = 8
_DIM = 4


def _make_shard(samples: List[Tuple[int, List[int]]]) -> dict:
    """One rank's micro-batch from ``[(g, [patches, ...]), ...]``; first token is ``1 + g*100``, vision rows ``g``."""
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
    """Shard contiguously across 2 ranks, run the real exchange, and check this rank's balanced shard."""
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
        cost_of=functools.partial(sample_cost, encoder_cost_weight=1.0),
        dp_group_gloo=gloo,
        dp_group_nccl=nccl,
        image_count_of=image_count_of,
    )[0]

    expected_globals = _expected_owned(global_samples, dp_rank, dp_size, n_groups)

    got_globals = [int((int(row[0].item()) - 1) // 100) for row in out["input_ids"].cpu()]
    assert got_globals == expected_globals, (rank, got_globals, expected_globals)

    # Vision must travel with its sample through the real all-to-all
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
    initialize_distributed()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this functional test")
    if dist.get_world_size() != 2:
        pytest.skip("This functional test requires exactly 2 ranks")

    # costs [4,3,1,2]: owners g0,g2 -> rank0, g1,g3 -> rank1; initial shards [g0,g1] / [g2,g3] force a swap
    global_samples = [[4], [3], [1], [2]]
    image_count_of = lambda b: (b["input_ids"] == _VISION_START).sum(dim=1).to(torch.long)  # noqa: E731
    _run_exchange_case(global_samples, image_count_of)
    dist.barrier()


@pytest.mark.run_only_on("GPU")
def test_reorder_exchange_dp2_variable_images():
    initialize_distributed()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this functional test")
    if dist.get_world_size() != 2:
        pytest.skip("This functional test requires exactly 2 ranks")

    # counts [2,1,0,1] at 2 patches/image -> costs [4,2,0,2]: g1 (single-image) and g2 (text-only) swap ranks
    global_samples = [[2, 2], [2], [], [2]]
    image_count_of = lambda b: (b["input_ids"] == _VISION_START).sum(dim=1).to(torch.long)  # noqa: E731
    _run_exchange_case(global_samples, image_count_of)
    dist.barrier()
