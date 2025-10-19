# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""

import random
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def build_pretraining_data_loader(
    dataset: Dataset,
    consumed_samples: int,
    dataloader_type: str,
    micro_batch_size: int,
    num_workers: int,
    data_sharding: bool,
    worker_init_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    drop_last: Optional[bool] = True,
    global_batch_size: Optional[int] = None,
) -> Optional[DataLoader]:
    """Build a dataloader for pretraining.

    Selects the appropriate sampler (MegatronPretrainingSampler,
    MegatronPretrainingRandomSampler, or MegatronPretrainingBatchSampler)
    based on `dataloader_type` and constructs a PyTorch DataLoader.

    Args:
        dataset: The dataset to load data from.
        consumed_samples: The number of samples already consumed (for resuming).
        dataloader_type: Type of dataloader: 'single', 'cyclic', 'batch', or 'external'.
                         'external' passes the dataset through directly.
        micro_batch_size: The batch size per GPU.
        num_workers: Number of workers for the DataLoader.
        data_sharding: Whether data sharding is enabled (used for random sampler).
        worker_init_fn: Optional function to initialize workers.
        collate_fn: Optional custom collate function.
        pin_memory: Whether to pin memory for the DataLoader.
        persistent_workers: Whether to use persistent workers.
        data_parallel_rank: Rank of the current GPU in the data parallel group.
        data_parallel_size: Total number of GPUs in the data parallel group.
        drop_last: Whether to drop last incomplete batch.
        global_batch_size: Total batch size across all data parallel ranks.
                          Required for 'batch' dataloader_type.

    Returns:
        A PyTorch DataLoader instance, or the dataset itself if dataloader_type is
        'external', or None if the input dataset is None.

    Raises:
        Exception: If an unsupported dataloader_type is provided.
        RuntimeError: If 'batch' dataloader_type is used without global_batch_size.
    """

    if dataset is None:
        return None

    # Megatron sampler
    if dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
        )
    elif dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding=data_sharding,
        )
    elif dataloader_type == "batch":
        if global_batch_size is None:
            raise RuntimeError(
                "global_batch_size must be provided when using dataloader_type='batch'. "
                "This is required for proper batch sampling in fine-tuning scenarios."
            )
        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            pad_samples_to_global_batch_size=not drop_last,
        )
    elif dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception("{} dataloader type is not supported.".format(dataloader_type))

    # Torch dataloader.
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
    )


class MegatronPretrainingSampler:
    """Batch sampler for Megatron pretraining (sequential, non-random).

    Provides indices for microbatches for a specific data parallel rank,
    ensuring that data is processed sequentially across ranks and iterations.

    Args:
        total_samples: Total number of samples in the dataset.
        consumed_samples: Number of samples already consumed (for resuming).
        micro_batch_size: Batch size per GPU.
        data_parallel_rank: Rank of the current GPU in the data parallel group.
        data_parallel_size: Total number of GPUs in the data parallel group.
        drop_last (bool, optional): If True, drops the last incomplete batch.
                                  Defaults to True.
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
    ) -> None:
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.consumed_samples < self.total_samples, "no samples left to consume: {}, {}".format(
            self.consumed_samples, self.total_samples
        )
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, (
            "data_parallel_rank should be smaller than data size: {}, {}".format(
                self.data_parallel_rank, data_parallel_size
            )
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.total_samples

    def get_start_end_idx(self) -> tuple[int, int]:
        """Calculate the start and end index for the current rank's microbatch."""
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self) -> Iterator[list[int]]:
        """Yields lists of indices for each microbatch assigned to this rank."""
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingBatchSampler:
    """Batch sampler for Megatron fine-tuning (global batch sampling).

    Samples entire global batches before distributing indices to data parallel ranks
    in an interleaved fashion. This ensures sequences within a single global batch
    are padded to the same length, which is critical for fine-tuning with variable
    sequence lengths.

    Args:
        total_samples: Total number of samples in the dataset.
        consumed_samples: Number of samples already consumed (for resuming).
        micro_batch_size: Batch size per GPU.
        global_batch_size: Total batch size across all data parallel ranks.
        data_parallel_rank: Rank of the current GPU in the data parallel group.
        data_parallel_size: Total number of GPUs in the data parallel group.
        drop_last: If True, drops the last incomplete batch.
        pad_samples_to_global_batch_size: If True, pads incomplete batches with -1 indices.
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        pad_samples_to_global_batch_size: bool = False,
    ) -> None:
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.drop_last = drop_last
        self.pad_samples_to_global_batch_size = pad_samples_to_global_batch_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size

        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.micro_batch_size > 0, f"micro_batch_size must be greater than 0, but {self.micro_batch_size}"
        assert data_parallel_size > 0, f"data parallel size must be greater than 0, but {data_parallel_size}"
        assert self.data_parallel_rank < data_parallel_size, (
            "data_parallel_rank should be smaller than data size: {}, {}".format(
                self.data_parallel_rank, data_parallel_size
            )
        )

        # Set and validate global batch size
        self._global_batch_size = global_batch_size
        if self._global_batch_size % self.micro_batch_times_data_parallel_size != 0:
            raise RuntimeError(
                f"`global_batch_size` ({self._global_batch_size}) is not divisible by "
                f"`micro_batch_size ({self.micro_batch_size}) x data_parallel_size "
                f"({self.data_parallel_size})`"
            )
        self._num_micro_batches = self._global_batch_size // self.micro_batch_times_data_parallel_size
        self._global_batch_size_on_this_data_parallel_rank = self._num_micro_batches * self.micro_batch_size

    def __len__(self) -> int:
        """Return the number of microbatches this sampler will yield.

        Since we yield one microbatch per global batch × num_micro_batches,
        multiply by num_micro_batches to get the total number of yields.
        """
        num_available_samples = self.total_samples - self.consumed_samples % self.total_samples
        if self.drop_last:
            num_global_batches = num_available_samples // self._global_batch_size
        else:
            num_global_batches = (num_available_samples + self._global_batch_size - 1) // self._global_batch_size

        # Each global batch yields num_micro_batches microbatches
        return num_global_batches * self._num_micro_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yields lists of indices for each microbatch assigned to this rank.

        Accumulates a full global batch, then distributes indices in interleaved fashion
        to data parallel ranks, yielding one microbatch at a time for megatron-core compatibility.

        This ensures all samples in a global batch can be padded to the same max length
        (important for variable-length finetuning) while being compatible with megatron-core's
        microbatch loop that calls next() multiple times per training step.
        """
        batch = []
        # Last batch will be dropped if drop_last is True
        for idx in range(self.consumed_samples % self.total_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self._global_batch_size:
                # Distribute indices in interleaved fashion across ranks
                all_indices = [
                    batch[i]
                    for i in range(
                        self.data_parallel_rank,
                        self._global_batch_size,
                        self.data_parallel_size,
                    )
                ]
                assert len(all_indices) == self._global_batch_size_on_this_data_parallel_rank

                # Yield one microbatch at a time
                for microbatch_idx in range(self._num_micro_batches):
                    start = microbatch_idx * self.micro_batch_size
                    end = start + self.micro_batch_size
                    yield all_indices[start:end]

                batch = []

        # Check the last partial batch and see if drop_last is set
        if len(batch) > 0 and not self.drop_last:
            # Distribute partial batch in interleaved fashion
            all_indices = [batch[i] for i in range(self.data_parallel_rank, len(batch), self.data_parallel_size)]
            if self.pad_samples_to_global_batch_size:
                num_pad = self._global_batch_size // self.data_parallel_size - len(all_indices)
                all_indices = all_indices + [-1] * num_pad

            # Yield one microbatch at a time
            for microbatch_idx in range(self._num_micro_batches):
                start = microbatch_idx * self.micro_batch_size
                end = start + self.micro_batch_size
                if start < len(all_indices):
                    yield all_indices[start:end]


class RandomSeedDataset(Dataset):
    """A dataset wrapper that sets the random seed based on epoch and index.

    Ensures reproducibility for random operations within the dataset's __getitem__
    when using multiple workers.

    Args:
        dataset: The base dataset to wrap.
        seed: The base random seed.
    """

    def __init__(self, dataset: Dataset, seed: int) -> None:
        """Initialize RandomSeedDataset."""
        self.base_seed = seed
        self.curr_seed = seed
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the length of the base dataset."""
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch number to adjust the random seed."""
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx: int) -> Any:
        """Get an item from the dataset, setting the random seed first."""
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:
    """Batch sampler for Megatron pretraining (randomized).

    Provides indices for microbatches for a specific data parallel rank,
    randomizing the order of samples within each epoch while supporting resumption.
    Handles data sharding across ranks if enabled.

    Args:
        dataset: The dataset (potentially wrapped with RandomSeedDataset).
        total_samples: Total number of samples in the dataset.
        consumed_samples: Number of samples already consumed (for resuming).
        micro_batch_size: Batch size per GPU.
        data_parallel_rank: Rank of the current GPU in the data parallel group.
        data_parallel_size: Total number of GPUs in the data parallel group.
        data_sharding: Whether data sharding is enabled.
    """

    def __init__(
        self,
        dataset: Dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        data_sharding: bool,
    ) -> None:
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, (
            "data_parallel_rank should be smaller than data size: {}, {}".format(
                self.data_parallel_rank, data_parallel_size
            )
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.total_samples

    def __iter__(self) -> Iterator[list[int]]:
        """Yields lists of indices for each microbatch assigned to this rank.

        Handles randomization within an epoch and data sharding.
        """
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
