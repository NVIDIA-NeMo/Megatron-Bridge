import random

import pytest

from megatron.bridge.models.bagel.data.order import (
    choose_group,
    plan_editing_sources,
    plan_t2i_sources,
    plan_vlm_sources,
    shuffle_and_shard,
)


pytestmark = pytest.mark.unit


def test_shuffle_and_shard_matches_bagel_rank_worker_floor_slices() -> None:
    assert shuffle_and_shard(
        list(range(10)),
        seed=42,
        rank=0,
        world_size=2,
        worker_id=0,
        num_workers=2,
    ) == [3, 7]
    assert shuffle_and_shard(
        list(range(10)),
        seed=42,
        rank=0,
        world_size=2,
        worker_id=1,
        num_workers=2,
    ) == [8, 2]


def test_source_planners_keep_stable_raw_coordinates() -> None:
    t2i = plan_t2i_sources(
        ["/data/chunk_0.parquet"],
        {"/data/chunk_0.parquet": [2, 1]},
        seed=42,
        rank=0,
        world_size=1,
    )
    assert [item["source"] for item in t2i] == [
        {"parquet": "chunk_0.parquet", "row_group": 0, "row": 0},
        {"parquet": "chunk_0.parquet", "row_group": 0, "row": 1},
        {"parquet": "chunk_0.parquet", "row_group": 1, "row": 0},
    ]

    editing = plan_editing_sources(
        [("/data/chunk_0.parquet", 0), ("/data/chunk_0.parquet", 1)],
        {("/data/chunk_0.parquet", 0): 1, ("/data/chunk_0.parquet", 1): 1},
        seed=42,
        rank=0,
        world_size=1,
    )
    assert [item["source"]["row_group"] for item in editing] == [1, 0]

    vlm = plan_vlm_sources(
        ["z", "a", "m", "b", "q"],
        jsonl_name="samples.jsonl",
        num_used_data=5,
        shuffle_seed=0,
        seed=42,
        rank=0,
        world_size=1,
    )
    assert [item["source"]["row"] for item in vlm] == [4, 3, 2, 0, 1]


def test_choose_group_uses_bagel_cumulative_weights_and_shared_rng() -> None:
    rng = random.Random(42)
    assert [choose_group([1, 1, 1], rng) for _ in range(5)] == [1, 0, 0, 0, 2]
