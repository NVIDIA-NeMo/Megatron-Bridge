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

import json
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, Self, TypeVar, cast

from megatron.energon import WorkerConfig


T = TypeVar("T")


class _RestorableDataset(Protocol):
    def restore_sample(self, key: tuple[str | int, ...]) -> object: ...


def shuffle_and_shard(
    items: Sequence[T],
    *,
    seed: int,
    rank: int,
    world_size: int,
    worker_id: int = 0,
    num_workers: int = 0,
    sort_key: Callable[[T], object] | None = None,
) -> list[T]:
    """Apply BAGEL's sorted epoch shuffle and rank/worker floor sharding."""
    ordered = sorted(items, key=sort_key)
    random.Random(seed).shuffle(ordered)
    per_rank = len(ordered) // world_size
    rank_items = ordered[rank * per_rank : (rank + 1) * per_rank]
    if num_workers == 0:
        return rank_items
    per_worker = len(rank_items) // num_workers
    worker_items = rank_items[worker_id * per_worker : (worker_id + 1) * per_worker]
    return worker_items[::-1]


def plan_t2i_sources(
    parquet_paths: Sequence[str],
    row_counts: Mapping[str, Sequence[int]],
    **shard_args: int,
) -> list[dict[str, object]]:
    """Plan stable BAGEL T2I source IDs in file, row-group, and row order."""
    paths = shuffle_and_shard(parquet_paths, **shard_args)
    return [
        {
            "dataset_name": "t2i_pretrain",
            "source": {"parquet": Path(path).name, "row_group": row_group, "row": row},
        }
        for path in paths
        for row_group, count in enumerate(row_counts[path])
        for row in range(count)
    ]


def plan_editing_sources(
    row_groups: Sequence[tuple[str, int]],
    row_counts: Mapping[tuple[str, int], int],
    **shard_args: int,
) -> list[dict[str, object]]:
    """Plan stable BAGEL Editing source IDs in shuffled row-group order."""
    planned = shuffle_and_shard(row_groups, **shard_args)
    return [
        {
            "dataset_name": "unified_edit",
            "source": {"parquet": Path(path).name, "row_group": row_group, "row": row},
        }
        for path, row_group in planned
        for row in range(row_counts[(path, row_group)])
    ]


def plan_vlm_sources(
    lines: Sequence[str],
    *,
    jsonl_name: str,
    num_used_data: int,
    shuffle_seed: int,
    **shard_args: int,
) -> list[dict[str, object]]:
    """Plan stable BAGEL VLM source IDs through line and epoch shuffles."""
    indexed_lines = list(enumerate(lines))
    random.Random(shuffle_seed).shuffle(indexed_lines)
    selected = indexed_lines[:num_used_data]
    planned = shuffle_and_shard(selected, sort_key=lambda item: item[1], **shard_args)
    return [{"dataset_name": "vlm_sft", "source": {"jsonl": jsonl_name, "row": row}} for row, _ in planned]


def choose_group(weights: Sequence[float], rng: random.Random) -> int:
    """Draw one BAGEL dataset group with the caller's shared Python RNG."""
    total = sum(weights)
    if total <= 0:
        raise ValueError("BAGEL group weights must have a positive sum")
    value = rng.random()
    for index in range(len(weights)):
        if value < sum(weights[: index + 1]) / total:
            return index
    raise RuntimeError("BAGEL group draw fell outside cumulative weights")


def plan_manifest_indices(
    manifest_path: Path,
    *,
    seed: int,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    num_used_data: int,
    shuffle_seed: int = 0,
) -> list[int]:
    """Map BAGEL's independently planned source stream to physical WDS sample indices."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = cast(list[dict[str, object]], manifest["samples"])
    planning = cast(dict[str, object], manifest["planning"])
    shard_args = {
        "seed": seed,
        "rank": rank,
        "world_size": world_size,
        "worker_id": worker_id,
        "num_workers": num_workers,
    }
    dataset_group = manifest["dataset_group"]
    if dataset_group == "t2i_pretrain":
        parquet_paths = cast(list[str], planning["parquet_paths"])[:num_used_data]
        row_counts = cast(dict[str, list[int]], planning["row_counts"])
        planned = plan_t2i_sources(parquet_paths, row_counts, **shard_args)
    elif dataset_group == "unified_edit":
        rows = cast(list[dict[str, object]], planning["row_groups"])
        selected = set(sorted({cast(str, row["parquet"]) for row in rows})[:num_used_data])
        row_groups = [
            (cast(str, row["parquet"]), cast(int, row["row_group"])) for row in rows if row["parquet"] in selected
        ]
        row_counts = {
            (cast(str, row["parquet"]), cast(int, row["row_group"])): cast(int, row["rows"])
            for row in rows
            if row["parquet"] in selected
        }
        planned = plan_editing_sources(row_groups, row_counts, **shard_args)
    elif dataset_group == "vlm_sft":
        planned = plan_vlm_sources(
            cast(list[str], planning["lines"]),
            jsonl_name=cast(str, planning["jsonl"]),
            num_used_data=num_used_data,
            shuffle_seed=shuffle_seed,
            **shard_args,
        )
    else:
        raise ValueError(f"unknown BAGEL dataset group: {dataset_group}")

    source_indexes = {
        json.dumps(sample["source"], sort_keys=True, separators=(",", ":")): cast(int, sample["index"])
        for sample in samples
    }
    return [source_indexes[json.dumps(item["source"], sort_keys=True, separators=(",", ":"))] for item in planned]


class BagelPlannedLoader(Iterator[object]):
    """Read an independently planned repeating source stream by Energon restore key."""

    def __init__(
        self,
        dataset: _RestorableDataset,
        sample_indices: Sequence[int],
        worker_config: WorkerConfig,
    ) -> None:
        """Store the random-access Energon dataset and canonical source plan."""
        if not sample_indices:
            raise ValueError("BAGEL source plan is empty")
        self.dataset = dataset
        self.sample_indices = list(sample_indices)
        self.worker_config = worker_config
        self.position = 0

    def __iter__(self) -> Self:
        """Return this stateful source reader."""
        return self

    def __next__(self) -> object:
        """Cook the next planned WDS sample through Energon's restore API."""
        sample_index = self.sample_indices[self.position % len(self.sample_indices)]
        restore_key = (
            "MapDataset",
            self.position,
            "MapDataset",
            self.position,
            "Webdataset",
            sample_index,
        )
        self.worker_config.worker_activate(self.position)
        try:
            sample = self.dataset.restore_sample(restore_key)
        finally:
            self.worker_config.worker_deactivate()
        self.position += 1
        return sample

    def save_state_rank(self) -> dict[str, int]:
        """Save this worker's source-stream position."""
        return {"position": self.position}

    def restore_state_rank(self, state: Mapping[str, object]) -> None:
        """Restore this worker's source-stream position."""
        position = state["position"]
        if not isinstance(position, int):
            raise TypeError("BAGEL planned-loader position must be an integer")
        self.position = position
