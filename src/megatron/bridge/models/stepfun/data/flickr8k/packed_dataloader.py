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

"""Synchronous ``MixedPackedDataloader``.

Instead of being a stateful ``__next__`` iterator, this exposes ``__len__``
+ ``__getitem__(idx)`` so it plugs into mbridge's
``MegatronPretrainingSampler`` + standard PyTorch ``DataLoader`` flow.

The internal schedule (sample order + non-truncation packing) is computed
once at ``__init__`` from fixed seeds, so the contents of pack ``idx`` are
deterministic. Per-step ordering across the train loop may still differ
because mbridge's sampler shuffles pack indices independently — but each
individual pack is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Optional, Union

import torch
from tqdm import tqdm

from megatron.bridge.models.stepfun.data.flickr8k.packing import PackingResult, pack
from megatron.bridge.models.stepfun.data.flickr8k.samplers import (
    LoopedSequentialSampler,
    LoopedShuffleSampler,
    WeightedRandomSampler,
)


class MixedPackedDataloader(torch.utils.data.Dataset):
    """Map-style packed dataset.

    Returns a fully assembled packed sample (already passed through
    ``transform``) for each index. Used by
    :class:`Step37Flickr8kSFTDataProvider` to feed mbridge's standard
    ``MegatronPretrainingSampler`` + DataLoader.
    """

    def __init__(
        self,
        datasets: list,
        epochs: list[float],
        max_length: int,
        oversize_policy: Literal["drop", "extend"] = "extend",
        transform: Optional[Callable] = None,
        dataset_sampling: Union[Literal["sequential", "random"], list[Literal["sequential", "random"]]] = "random",
    ):
        if len(datasets) == 0:
            raise ValueError("datasets cannot be empty.")
        if len(datasets) != len(epochs):
            raise ValueError("datasets and epochs must have the same length.")

        self.datasets = datasets
        self.ds_epochs = epochs
        self.transform = transform
        self.dataset_sampling = self._normalize_dataset_sampling(dataset_sampling, len(datasets))

        self.piece_order, self.packing_result = self._schedule_all(
            max_length=max_length, oversize_policy=oversize_policy
        )
        self._pack_ranges = self.packing_result.packed_sample_ranges
        if len(self._pack_ranges) == 0:
            raise ValueError("packed dataset is empty.")

    @staticmethod
    def _normalize_dataset_sampling(
        dataset_sampling: Union[Literal["sequential", "random"], list[Literal["sequential", "random"]]],
        num_datasets: int,
    ) -> list[Literal["sequential", "random"]]:
        if isinstance(dataset_sampling, str):
            normalized = [dataset_sampling] * num_datasets
        else:
            normalized = list(dataset_sampling)
            if len(normalized) != num_datasets:
                raise ValueError("dataset_sampling list must have the same length as datasets.")

        invalid = [strategy for strategy in normalized if strategy not in {"sequential", "random"}]
        if invalid:
            raise ValueError(
                f"dataset_sampling contains unsupported strategy: {invalid[0]!r}. "
                "Supported strategies are 'sequential' and 'random'."
            )

        return normalized

    @staticmethod
    def _build_in_domain_sampler(
        sampling_strategy: Literal["sequential", "random"],
        size: int,
        idx: int,
    ) -> Union[LoopedShuffleSampler, LoopedSequentialSampler]:
        if sampling_strategy == "random":
            return LoopedShuffleSampler(size=size, base_seed=1234 + idx)
        return LoopedSequentialSampler(size=size)

    def _schedule_all(
        self,
        max_length: int,
        oversize_policy: str = "drop",
    ) -> tuple[list[tuple[int, int]], PackingResult]:
        in_domain_samplers: list[Union[LoopedShuffleSampler, LoopedSequentialSampler]] = []
        _weights: list[float] = []
        sample_sizes: list[list[int]] = []

        for idx, (dataset, epoch, sampling_strategy) in enumerate(
            zip(self.datasets, self.ds_epochs, self.dataset_sampling, strict=True)
        ):
            size = len(dataset)
            if size <= 0:
                raise ValueError("Dataset is empty.")
            if float(epoch) <= 0:
                raise ValueError("Epoch must be positive.")

            # Probe each sample for its packed-NTP length (= tokens.numel()-1).
            # There is no precomputed dataset with cached lengths here, so we
            # always probe.
            sizes_i = [len(x) for x in tqdm(dataset, desc=f"Probing dataset {idx} sizes")]
            sample_sizes.append(sizes_i)
            in_domain_samplers.append(self._build_in_domain_sampler(sampling_strategy, size=size, idx=idx))

            weight = float(epoch) * float(size)  # weight = epoch × N_samples
            if weight <= 0:
                raise ValueError("Sampling weight must be positive.")
            _weights.append(weight)

        inter_domain_sampler = WeightedRandomSampler(
            size=len(self.datasets),
            base_seed=1234,
            weights=_weights,
        )

        scheduled_piece_order: list[tuple[int, int]] = []
        scheduled_piece_sizes: list[int] = []
        for _i in tqdm(range(int(sum(_weights))), desc="Sampling"):
            domain_idx = next(inter_domain_sampler)
            in_domain_idx = next(in_domain_samplers[domain_idx])
            scheduled_piece_order.append((domain_idx, in_domain_idx))
            scheduled_piece_sizes.append(sample_sizes[domain_idx][in_domain_idx])

        packing_result = pack(scheduled_piece_sizes, max_len=max_length, oversize_policy=oversize_policy)
        return scheduled_piece_order, packing_result

    def __len__(self) -> int:
        return len(self._pack_ranges)

    def __getitem__(self, idx: int) -> Any:
        """Assemble the pack at index ``idx`` without using a mutable
        internal cursor.

        Returns the same result for a given ``idx`` on every call: the
        precomputed in-domain order selects the same samples, which are then
        run through ``transform``.
        """
        start, count = self._pack_ranges[idx % len(self._pack_ranges)]
        packed_items: list[Any] = []
        for domain_idx, in_domain_idx in self.piece_order[start : start + count]:
            packed_items.append(self.datasets[domain_idx][in_domain_idx])
        if self.transform:
            packed_items = self.transform(packed_items)
        return packed_items
