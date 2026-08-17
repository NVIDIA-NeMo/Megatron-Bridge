# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.bridge.data.samplers import MegatronPretrainingSampler


pytestmark = pytest.mark.unit


def test_single_sampler_rejects_unsafe_partial_distributed_batch() -> None:
    """A partial sequential tail must not desynchronize data-parallel ranks."""
    sampler = MegatronPretrainingSampler(
        total_samples=5,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=2,
        drop_last=False,
    )

    with pytest.raises(ValueError, match="drop_last=False"):
        list(sampler)


def test_single_sampler_keeps_partial_batch_without_data_parallelism() -> None:
    """Single-rank users can safely retain a smaller final batch."""
    sampler = MegatronPretrainingSampler(
        total_samples=5,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=1,
        drop_last=False,
    )

    assert list(sampler) == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize(
    ("data_parallel_rank", "expected_batches"),
    [(0, [[0, 1], [4, 5]]), (1, [[2, 3], [6, 7]])],
)
def test_single_sampler_distributes_complete_batches_across_ranks(
    data_parallel_rank: int, expected_batches: list[list[int]]
) -> None:
    """Complete distributed batches retain their rank-local ownership."""
    sampler = MegatronPretrainingSampler(
        total_samples=8,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=2,
        drop_last=False,
    )

    assert list(sampler) == expected_batches


def test_single_sampler_drops_partial_distributed_batch() -> None:
    """The existing drop-last path remains valid for distributed tails."""
    sampler = MegatronPretrainingSampler(
        total_samples=5,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=2,
        drop_last=True,
    )

    assert list(sampler) == [[2, 3]]
