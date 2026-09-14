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

"""Two-rank CPU (gloo) tests for expert-parallel replicated adapter gradient synchronization.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_expert_parallel_grad_sync_distributed.py
"""

import os
import warnings
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.bridge.peft.utils import allreduce_expert_parallel_replicated_grads, mark_expert_parallel_replicated


_WORLD_SIZE = 2


@pytest.fixture(scope="module")
def ep_group() -> Iterator[object]:
    """Provide one two-rank gloo group standing in for the expert-model-parallel group."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="gloo")
    try:
        yield dist.group.WORLD
    finally:
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _rank_value(ep_group: object) -> float:
    """Return a per-rank scalar whose sum over both ranks is three."""
    return float(dist.get_rank(group=ep_group) + 1)


def _make_marked_param(ep_group: object, shape: tuple[int, ...] = (2, 3), dtype: torch.dtype = torch.float32):
    param = nn.Parameter(torch.ones(shape, dtype=dtype))
    mark_expert_parallel_replicated(param, ep_group=ep_group)
    return param


@pytest.mark.unit
def test_fallback_hook_sums_each_microbatch_until_finalize_sync_is_enabled(ep_group: object) -> None:
    """Without the finalize hook the per-layer fallback sums grads; afterwards finalize owns the reduction."""
    param = _make_marked_param(ep_group)
    value = _rank_value(ep_group)

    (param * value).sum().backward()
    torch.testing.assert_close(param.grad, torch.full_like(param, 3.0))

    # Finalize must not double count what the hook already summed; it removes the hook instead.
    allreduce_expert_parallel_replicated_grads([nn.ParameterList([param])], ep_group=ep_group)
    torch.testing.assert_close(param.grad, torch.full_like(param, 3.0))
    assert not hasattr(param, "_expert_parallel_grad_hook")

    # From now on finalize performs the sum.
    param.grad = None
    (param * value).sum().backward()
    torch.testing.assert_close(param.grad, torch.full_like(param, value))
    allreduce_expert_parallel_replicated_grads([nn.ParameterList([param])], ep_group=ep_group)
    torch.testing.assert_close(param.grad, torch.full_like(param, 3.0))


@pytest.mark.unit
def test_finalize_sums_fused_accumulation_gradients_the_hook_cannot_reach(ep_group: object) -> None:
    """The hook must not all-reduce MCore's dummy gradient; finalize reduces main_grad, coalesced per dtype."""
    first = _make_marked_param(ep_group)
    second = _make_marked_param(ep_group, shape=(4,), dtype=torch.float64)
    value = _rank_value(ep_group)
    for param in (first, second):
        param.main_grad = torch.full_like(param, value)
        # MCore's fused wgrad path sets this before returning its dummy weight gradient.
        param.grad_added_to_main_grad = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (first * value).sum().backward()

    torch.testing.assert_close(first.grad, torch.full_like(first, value))
    if dist.get_rank(group=ep_group) == 0:
        assert any("main_grad" in str(warning.message) for warning in caught)

    allreduce_expert_parallel_replicated_grads([nn.ParameterList([first, second])], ep_group=ep_group)
    for param in (first, second):
        torch.testing.assert_close(param.main_grad, torch.full_like(param, 3.0))
        assert not hasattr(param, "_expert_parallel_grad_hook")
