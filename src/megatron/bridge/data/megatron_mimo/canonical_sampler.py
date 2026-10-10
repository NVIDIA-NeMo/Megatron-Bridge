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

"""Canonical-grid batch sampling for MegatronMIMO scalable data-parallel reads.

With ``megatron_mimo_scalable_dp`` every module's loaders must materialize the same
ordered global micro-batch, because the ``BridgeCommunicator`` routes modality
embeddings to language ranks by contiguous position along the batch dim. A sampler's
shard assignment depends on its ``(data_parallel_rank, data_parallel_size,
micro_batch_size)``, so modules with different DP sizes cannot shard by their own DP:
a shuffling sampler would hand them different sample sets. Instead, all loaders shard
on one shared **canonical grid** — the least common multiple of the module DP sizes.
A rank whose module has DP size ``d`` covers ``grid // d`` consecutive canonical
groups and concatenates their windows, so concatenating any module's rank batches
reproduces the identical ordered global micro-batch under any deterministic sampler
(``single`` and ``cyclic`` included; the shuffle seed derives from the epoch alone).
"""

from __future__ import annotations

import math
from typing import Callable, Iterator

from torch.utils.data import DataLoader, Dataset

from megatron.bridge.data.samplers import MegatronPretrainingRandomSampler, MegatronPretrainingSampler


def canonical_grid_size(module_dp_sizes: list[int]) -> int:
    """Return the shared sampler grid size: the LCM of every module's DP size."""
    if not module_dp_sizes or any(dp is None or dp < 1 for dp in module_dp_sizes):
        raise ValueError(f"module DP sizes must be positive integers (got {module_dp_sizes}).")
    return math.lcm(*module_dp_sizes)


def covered_canonical_groups(dp_rank: int, dp_size: int, grid_size: int) -> list[int]:
    """Return the canonical groups this rank reads.

    The groups are the ``grid_size // dp_size`` consecutive slots matching the
    contiguous batch-dim chunk the ``BridgeCommunicator`` routes to this DP rank.
    """
    if grid_size % dp_size != 0:
        raise ValueError(
            f"canonical grid size ({grid_size}) is not divisible by the module DP size ({dp_size}); "
            "module DP sizes must divide their least common multiple."
        )
    span = grid_size // dp_size
    return list(range(dp_rank * span, (dp_rank + 1) * span))


class CanonicalGroupBatchSampler:
    """Concatenate per-canonical-group Megatron samplers into one batch sampler.

    Holds one flat sampler per covered canonical group, all built with the same
    ``(micro_batch // grid, grid)`` geometry and the same global ``consumed_samples``.
    Each yield emits the groups' current windows concatenated in group order. Group
    streams are functions of ``(group, grid, micro_batch, consumed, epoch seed)`` only,
    so every rank covering group ``g`` — in any module — sees the identical stream.
    """

    def __init__(self, samplers: list) -> None:
        if not samplers:
            raise ValueError("CanonicalGroupBatchSampler needs at least one group sampler.")
        self.samplers = samplers

    def __len__(self) -> int:
        """Match the flat Megatron samplers' convention of reporting total samples."""
        return min(len(sampler) for sampler in self.samplers)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one concatenated index window per micro-batch."""
        iterators = [iter(sampler) for sampler in self.samplers]
        while True:
            window: list[int] = []
            for iterator in iterators:
                group_batch = next(iterator, None)
                if group_batch is None:
                    # Groups share one truncated geometry, so they exhaust on the same window.
                    return
                window.extend(group_batch)
            yield window


def build_canonical_group_batch_sampler(
    *,
    dataloader_type: str,
    dataset: Dataset,
    consumed_samples: int,
    micro_batch_size: int,
    grid_size: int,
    groups: list[int],
    data_sharding: bool,
    drop_last: bool = True,
) -> CanonicalGroupBatchSampler:
    """Build this rank's canonical-group batch sampler for scalable MIMO reads.

    Args:
        dataloader_type: ``"single"`` or ``"cyclic"``; other types have no shard
            assignment that is consistent across modules and are rejected.
        dataset: The dataset the loader reads.
        consumed_samples: Global consumed-sample count, exactly as the flat samplers
            expect (used for resume / epoch derivation).
        micro_batch_size: The global micro-batch size (not the per-rank share).
        grid_size: Canonical grid size from :func:`canonical_grid_size`.
        groups: This rank's groups from :func:`covered_canonical_groups`.
        data_sharding: Passed through to the cyclic sampler.
        drop_last: Must stay ``True``: a partial final window gives the groups
            unequal shares and breaks the positional routing.

    Returns:
        The merged batch sampler for ``torch.utils.data.DataLoader(batch_sampler=...)``.
    """
    if not drop_last:
        raise ValueError("megatron_mimo_scalable_dp requires drop_last=True (partial windows misalign modules).")
    if micro_batch_size % grid_size != 0:
        raise ValueError(
            f"micro_batch_size ({micro_batch_size}) must be divisible by the canonical grid size ({grid_size})."
        )
    group_micro_batch_size = micro_batch_size // grid_size
    # Truncate to whole global micro-batches: the flat samplers round their active range at
    # per-group granularity, which diverges per group for a non-multiple dataset size (the
    # cyclic data_sharding=False stride would give groups unequal window counts).
    total_samples = (len(dataset) // micro_batch_size) * micro_batch_size
    if total_samples <= 0:
        raise ValueError(f"dataset ({len(dataset)} samples) is smaller than one micro-batch ({micro_batch_size}).")

    samplers = []
    for group in groups:
        if dataloader_type == "single":
            samplers.append(
                MegatronPretrainingSampler(
                    total_samples=total_samples,
                    consumed_samples=consumed_samples,
                    micro_batch_size=group_micro_batch_size,
                    data_parallel_rank=group,
                    data_parallel_size=grid_size,
                    drop_last=True,
                )
            )
        elif dataloader_type == "cyclic":
            samplers.append(
                MegatronPretrainingRandomSampler(
                    dataset,
                    total_samples=total_samples,
                    consumed_samples=consumed_samples,
                    micro_batch_size=group_micro_batch_size,
                    data_parallel_rank=group,
                    data_parallel_size=grid_size,
                    data_sharding=data_sharding,
                )
            )
        else:
            raise ValueError(
                f"megatron_mimo_scalable_dp supports dataloader_type 'single' or 'cyclic' (got {dataloader_type!r})."
            )
    return CanonicalGroupBatchSampler(samplers)


def build_canonical_mimo_data_loader(
    dataset: Dataset | None,
    *,
    consumed_samples: int,
    dataloader_type: str,
    micro_batch_size: int,
    module_dp_sizes: list[int],
    dp_rank: int,
    dp_size: int,
    data_sharding: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    collate_fn: Callable | None,
    persistent_workers: bool,
) -> DataLoader | None:
    """Build this rank's read-sharded DataLoader for scalable MegatronMIMO reads.

    Computes the canonical grid from ``module_dp_sizes``, derives this rank's covered
    groups from its module-local ``(dp_rank, dp_size)``, and wraps the merged batch
    sampler in a ``DataLoader``. Returns ``None`` when ``dataset`` is ``None``.
    """
    if dataset is None:
        return None
    grid = canonical_grid_size(module_dp_sizes)
    groups = covered_canonical_groups(dp_rank, dp_size, grid)
    batch_sampler = build_canonical_group_batch_sampler(
        dataloader_type=dataloader_type,
        dataset=dataset,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        grid_size=grid,
        groups=groups,
        data_sharding=data_sharding,
        drop_last=drop_last,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
    )
