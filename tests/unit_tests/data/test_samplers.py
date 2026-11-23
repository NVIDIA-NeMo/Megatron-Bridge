# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import multiprocessing

import pytest
import torch
from torch.utils.data import Dataset

import megatron.bridge.data.samplers as samplers
from megatron.bridge.data.samplers import (
    MegatronPretrainingRandomSampler,
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


def test_dataloader_worker_releases_gpu_fds_before_caller_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker setup must release inherited GPU descriptors on every loader path."""
    cleanup_calls = multiprocessing.Value("i", 0)
    caller_calls = multiprocessing.Value("i", 0)

    def record_cleanup() -> None:
        cleanup_calls.value += 1

    def record_caller_init(_worker_id: int) -> None:
        assert cleanup_calls.value == 1
        caller_calls.value += 1

    monkeypatch.setattr(samplers, "_close_nvidia_device_fds", record_cleanup)
    dataloader = build_pretraining_data_loader(
        dataset=_RandomValueDataset(),
        consumed_samples=0,
        dataloader_type="single",
        micro_batch_size=1,
        num_workers=1,
        data_sharding=False,
        persistent_workers=False,
        worker_init_fn=record_caller_init,
    )

    next(iter(dataloader))

    assert cleanup_calls.value == 1
    assert caller_calls.value == 1


@pytest.mark.parametrize(
    ("num_workers", "prefetch_factor", "expected"),
    [(0, None, None), (0, 4, None), (2, None, 2), (2, 8, 8)],
)
def test_prefetch_factor_applies_only_to_worker_loaders(
    num_workers: int, prefetch_factor: int | None, expected: int | None
) -> None:
    """prefetch_factor reaches worker-backed loaders; PyTorch rejects it without workers."""
    dataloader = build_pretraining_data_loader(
        dataset=_RandomValueDataset(),
        consumed_samples=0,
        dataloader_type="single",
        micro_batch_size=1,
        num_workers=num_workers,
        data_sharding=False,
        prefetch_factor=prefetch_factor,
    )

    assert dataloader.prefetch_factor == expected


def test_cyclic_sampler_non_sharded_tail_keeps_data_parallel_epochs_aligned() -> None:
    """Every DP rank must cross a truncated cyclic epoch on the same step."""

    def make_sampler(rank: int, consumed_samples: int = 0) -> MegatronPretrainingRandomSampler:
        return MegatronPretrainingRandomSampler(
            list(range(10)),
            total_samples=10,
            consumed_samples=consumed_samples,
            micro_batch_size=2,
            data_parallel_rank=rank,
            data_parallel_size=3,
            data_sharding=False,
        )

    def cyclic_batches(sampler: MegatronPretrainingRandomSampler):
        while True:
            yield from sampler

    uninterrupted = [cyclic_batches(make_sampler(rank)) for rank in range(3)]
    _ = [next(rank_batches) for rank_batches in uninterrupted]
    second_step = [next(rank_batches) for rank_batches in uninterrupted]
    resumed_step = [next(iter(make_sampler(rank, consumed_samples=6))) for rank in range(3)]

    assert len({index for batch in second_step for index in batch}) == 6
    assert second_step == resumed_step


def _sharded_indices(
    rank: int, *, total_samples: int, data_parallel_size: int, consumed_samples: int = 0
) -> list[int]:
    sampler = MegatronPretrainingRandomSampler(
        list(range(total_samples)),
        total_samples=total_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=1,
        data_parallel_rank=rank,
        data_parallel_size=data_parallel_size,
        data_sharding=True,
    )
    return [index for batch in sampler for index in batch]


def test_cyclic_sampler_sharded_ranks_do_not_replay_a_repeating_dataset() -> None:
    """Shards that hold the same examples must not be read in the same order.

    DirectSFTDataset repeats its base examples (index % base size). With two ranks and twice the base size, each
    rank's shard holds every example once; one shared shuffle fed both ranks the same example at every step.
    """
    num_examples, data_parallel_size = 64, 2
    examples = [
        [
            index % num_examples
            for index in _sharded_indices(
                rank, total_samples=num_examples * data_parallel_size, data_parallel_size=data_parallel_size
            )
        ]
        for rank in range(data_parallel_size)
    ]

    assert all(sorted(rank_examples) == list(range(num_examples)) for rank_examples in examples)
    assert sum(first == second for first, second in zip(*examples)) < num_examples // 8


def test_cyclic_sampler_sharded_rank_zero_keeps_the_epoch_shuffle() -> None:
    """Rank 0 still shuffles its shard with the epoch seed, so its order does not change."""
    expected = torch.randperm(6, generator=torch.Generator().manual_seed(0)).tolist()

    assert _sharded_indices(0, total_samples=12, data_parallel_size=2) == expected


def test_cyclic_sampler_sharded_resume_continues_each_rank_shuffle() -> None:
    """A resumed sampler continues every rank's own shuffle where the uninterrupted one would be."""
    for rank in range(2):
        uninterrupted = _sharded_indices(rank, total_samples=12, data_parallel_size=2)
        resumed = _sharded_indices(rank, total_samples=12, data_parallel_size=2, consumed_samples=4)

        assert sorted(uninterrupted) == list(range(rank * 6, rank * 6 + 6))
        assert resumed == uninterrupted[2:]


def _small_loader(dataloader_type: str):
    return build_pretraining_data_loader(
        dataset=list(range(8)),
        consumed_samples=0,
        dataloader_type=dataloader_type,
        micro_batch_size=2,
        num_workers=0,
        data_sharding=True,
        global_batch_size=2,
    )


@pytest.mark.parametrize("dataloader_type", ["single", "cyclic", "batch"])
def test_loader_iterators_leave_global_rng_untouched(dataloader_type: str) -> None:
    """Creating and exhausting loader iterators must not draw from the global CPU RNG.

    On resume, the checkpoint has already restored that RNG before the iterators are created,
    so a draw here would leave the resumed run one draw ahead of an uninterrupted run. The
    second pass is what cyclic_iter does at an epoch boundary.
    """
    torch.manual_seed(1234)
    rng_state = torch.get_rng_state()
    dataloader = _small_loader(dataloader_type)

    for _ in range(2):
        for _batch in dataloader:
            pass

    assert torch.equal(torch.get_rng_state(), rng_state)


def test_loader_base_seed_does_not_depend_on_restored_rng_state() -> None:
    """A loader built on resume gives its workers the same base seed as one built at step 0."""
    torch.manual_seed(1234)
    fresh_base_seed = iter(_small_loader("single"))._base_seed
    # Training advances the global RNG before the checkpoint that a resume restores.
    torch.rand(16)

    assert iter(_small_loader("single"))._base_seed == fresh_base_seed
