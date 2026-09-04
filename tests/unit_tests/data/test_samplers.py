# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from megatron.bridge.data.samplers import (
    MegatronPretrainingSampler,
    RandomSeedDataset,
    build_pretraining_data_loader,
)


pytestmark = pytest.mark.unit


class _RandomValueDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return idx, torch.randint(0, 1_000_000, (1,)).item()


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


@pytest.mark.parametrize(("num_workers", "persistent_workers"), [(0, False), (1, False), (1, True)])
def test_cyclic_sampler_resume_seeds_worker_dataset_for_current_epoch(
    num_workers: int, persistent_workers: bool
) -> None:
    """A resumed cyclic loader must seed stochastic samples with its resumed epoch."""
    base_seed = 100
    dataset = RandomSeedDataset(_RandomValueDataset(), seed=base_seed)
    dataloader = build_pretraining_data_loader(
        dataset=dataset,
        consumed_samples=len(dataset),
        dataloader_type="cyclic",
        micro_batch_size=1,
        num_workers=num_workers,
        data_sharding=False,
        persistent_workers=persistent_workers,
    )

    sample_idx, actual_value = next(iter(dataloader))
    sample_idx = sample_idx.item()
    generator = torch.Generator().manual_seed(base_seed + 1 + sample_idx)
    expected_value = torch.randint(0, 1_000_000, (1,), generator=generator).item()

    assert actual_value.item() == expected_value


def test_random_seed_dataset_preserves_main_process_rng_state() -> None:
    """Synchronous dataset seeding must not overwrite the training RNG streams."""
    dataset = RandomSeedDataset(_RandomValueDataset(), seed=100)
    dataloader = build_pretraining_data_loader(
        dataset=dataset,
        consumed_samples=0,
        dataloader_type="cyclic",
        micro_batch_size=1,
        num_workers=0,
        data_sharding=False,
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    data_iterator = iter(dataloader)

    python_generator = random.Random()
    python_generator.setstate(random.getstate())
    expected_python = python_generator.random()
    numpy_generator = np.random.RandomState()
    numpy_generator.set_state(np.random.get_state())
    expected_numpy = numpy_generator.random_sample()
    torch_generator = torch.Generator()
    torch_generator.set_state(torch.random.get_rng_state())
    expected_torch = torch.rand(1, generator=torch_generator)

    next(data_iterator)

    assert random.random() == expected_python
    assert np.random.random_sample() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
