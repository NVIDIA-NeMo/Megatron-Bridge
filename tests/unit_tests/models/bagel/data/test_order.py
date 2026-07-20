from pathlib import Path

import pytest
from megatron.energon import WorkerConfig

from megatron.bridge.models.bagel.data.order import (
    BagelPlannedLoader,
    plan_editing_sources,
    plan_manifest_indices,
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


def test_manifest_plan_maps_official_sources_to_canonical_wds_indices(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
          "dataset_group": "t2i_pretrain",
          "samples": [
            {"index": 0, "source": {"parquet": "a.parquet", "row_group": 0, "row": 0}},
            {"index": 1, "source": {"parquet": "b.parquet", "row_group": 0, "row": 0}}
          ],
          "planning": {
            "parquet_paths": ["a.parquet", "b.parquet"],
            "row_counts": {"a.parquet": [1], "b.parquet": [1]}
          }
        }""",
        encoding="utf-8",
    )
    assert plan_manifest_indices(
        manifest,
        seed=1,
        rank=0,
        world_size=1,
        worker_id=0,
        num_workers=0,
        num_used_data=2,
    ) == [1, 0]


def test_planned_loader_restores_physical_indices_and_position() -> None:
    class Dataset:
        def restore_sample(self, key: tuple[str | int, ...]) -> object:
            return key

    loader = BagelPlannedLoader(Dataset(), [2, 0], WorkerConfig(rank=0, world_size=1, num_workers=0))
    assert next(loader) == ("MapDataset", 0, "MapDataset", 0, "Webdataset", 2)
    state = loader.save_state_rank()
    assert next(loader) == ("MapDataset", 1, "MapDataset", 1, "Webdataset", 0)
    loader.restore_state_rank(state)
    assert next(loader) == ("MapDataset", 1, "MapDataset", 1, "Webdataset", 0)
