# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import runpy
import tarfile
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_vl" / "prepare_mantis_energon.py"


def _load_module() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_sample_split_is_stable_and_validates_fraction():
    module = _load_module()

    assert module["_sample_split"]("sample", 0.0) == "train"
    assert module["_sample_split"]("sample", 0.5) == module["_sample_split"]("sample", 0.5)
    with pytest.raises(ValueError, match="validation_fraction"):
        module["convert"]("missing", "output", 1, validation_fraction=1.0)


def test_convert_writes_mantis_images_and_conversation(tmp_path: Path):
    module = _load_module()
    subset = tmp_path / "source" / "subset-a"
    subset.mkdir(parents=True)
    (subset / "sample.jpg").write_bytes(b"test-jpeg-bytes")
    pd.DataFrame(
        [
            {
                "images": [{"path": "sample.jpg"}],
                "conversation": [
                    {"role": "user", "content": "<image>Describe this image."},
                    {"role": "assistant", "content": "A test image."},
                ],
            }
        ]
    ).to_parquet(subset / "train-00000-of-00001.parquet")

    output = tmp_path / "output"
    counts = module["convert"](str(tmp_path / "source"), str(output), 10, validation_fraction=0.0)

    assert counts == {"train": 1, "val": 0}
    assert not (output / "val-shard-000000.tar").exists()
    with tarfile.open(output / "train-shard-000000.tar") as archive:
        assert archive.getnames() == [
            "subset-a__train-00000-of-00001__000000.jpgs",
            "subset-a__train-00000-of-00001__000000.json",
        ]


def test_prepare_energon_dataset_indexes_nonempty_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    calls = []
    prepare_energon_dataset = module["prepare_energon_dataset"]
    monkeypatch.setitem(
        prepare_energon_dataset.__globals__,
        "prepare_webdataset",
        lambda path, patterns, *, num_workers: calls.append((path, patterns, num_workers)),
    )

    prepare_energon_dataset(
        str(tmp_path),
        counts={"train": 10, "val": 0},
        num_workers=4,
    )

    assert calls == [(str(tmp_path), {"train": "train-shard-.*"}, 4)]
    dataset_yaml = (tmp_path / ".nv-meta" / "dataset.yaml").read_text(encoding="utf-8")
    assert "megatron.bridge.data.energon.task_encoder_utils" in dataset_yaml
    assert "imgs: jpgs" in dataset_yaml
