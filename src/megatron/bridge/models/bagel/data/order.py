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

import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar


T = TypeVar("T")


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
