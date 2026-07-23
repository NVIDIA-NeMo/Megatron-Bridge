# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path
from unittest.mock import patch

import pytest

from megatron.bridge.data.energon.prepare import prepare_webdataset


pytestmark = pytest.mark.unit


def test_prepare_webdataset_indexes_relative_shards_with_explicit_splits(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "train-shard-000000.tar").touch()
    (tmp_path / "nested" / "val-shard-000000.tgz").touch()

    with patch(
        "megatron.energon.flavors.webdataset.base_webdataset.BaseWebdatasetFactory.prepare_dataset"
    ) as prepare_dataset:
        prepare_webdataset(
            tmp_path,
            {"train": "train-shard-.*", "val": "nested/val-shard-.*"},
            num_workers=3,
        )

    prepare_dataset.assert_called_once_with(
        tmp_path,
        ["nested/val-shard-000000.tgz", "train-shard-000000.tar"],
        split_parts_patterns=[("train", "train-shard-.*"), ("val", "nested/val-shard-.*")],
        shuffle_seed=None,
        workers=3,
    )


@pytest.mark.parametrize(
    ("split_patterns", "match"),
    [
        ({}, "at least one named split"),
        ({"train": "missing-.*"}, "did not match any tar shards"),
        ({"train": "["}, "Invalid regex"),
    ],
)
def test_prepare_webdataset_rejects_invalid_split_patterns(tmp_path: Path, split_patterns: dict[str, str], match: str):
    (tmp_path / "train-shard-000000.tar").touch()

    with pytest.raises(ValueError, match=match):
        prepare_webdataset(tmp_path, split_patterns)


def test_prepare_webdataset_requires_shards_and_positive_workers(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No .tar or .tgz shards"):
        prepare_webdataset(tmp_path, {"train": ".*"})

    with pytest.raises(ValueError, match="num_workers"):
        prepare_webdataset(tmp_path, {"train": ".*"}, num_workers=0)
