# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
import json
import runpy
import sys
import tarfile
import types
from pathlib import Path

import pytest
from PIL import Image


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_vl" / "prepare_datacomp_energon.py"


def _load_module() -> dict:
    return runpy.run_path(str(SCRIPT))


def _jpeg(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="JPEG")
    return output.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_source_shard(path: Path) -> None:
    with tarfile.open(path, "w") as archive:
        for index, caption in enumerate(("a red square", "a blue square")):
            key = f"{index:012d}"
            metadata = {
                "uid": f"{index + 1:032x}",
                "url": f"https://example.test/{index}.jpg",
                "key": key,
                "caption": caption,
                "face_bboxes": [] if index == 0 else [[0, 0, 1, 1]],
                "sha256": f"{index + 2:064x}",
                "width": 8,
                "height": 6,
                "original_width": 8,
                "original_height": 6,
                "status": "success",
            }
            _add_bytes(archive, f"{key}.jpg", _jpeg((255 if index == 0 else 0, 0, 255 if index else 0)))
            _add_bytes(archive, f"{key}.json", json.dumps(metadata).encode())
            _add_bytes(archive, f"{key}.txt", caption.encode())


def test_stable_split_validates_fraction_and_is_deterministic():
    module = _load_module()
    stable_split = module["_stable_split"]

    assert stable_split("sample", 0.0) == "train"
    assert stable_split("sample", 0.5) == stable_split("sample", 0.5)
    with pytest.raises(ValueError, match="validation_fraction"):
        stable_split("sample", 1.0)


def test_convert_writes_deterministic_qwen_chatml_shards(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_source_shard(source / "00000000.tar")

    output = tmp_path / "output"
    manifest = module["convert"](
        source,
        output,
        max_count=1,
        validation_fraction=0.0,
        minimum_train_samples=2,
    )

    assert manifest["accepted"] is True
    assert manifest["counts"] == {
        "input_samples": 2,
        "train": 2,
        "val": 0,
        "skipped": 0,
        "skip_reasons": {},
    }
    assert manifest["source"]["raw_shards_opened"] == ["00000000.tar"]
    assert [shard["samples"] for shard in manifest["output_shards"]] == [1, 1]

    first_shard = output / "train-shard-000000.tar"
    with tarfile.open(first_shard) as archive:
        assert archive.getnames() == [
            "00000000000000000000000000000001.image.jpg",
            "00000000000000000000000000000001.conversation.json",
        ]
        conversation_file = archive.extractfile(archive.getmember(archive.getnames()[1]))
        assert conversation_file is not None
        conversation = json.load(conversation_file)
        assert conversation["conversation"][0]["content"] == [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ]
        assert conversation["conversation"][1]["content"] == [{"type": "text", "text": "a red square"}]
        assert conversation["source"]["revision"] == module["DATACOMP_REVISION"]
        assert conversation["source"]["download_status"] == "success"
        assert conversation["source"]["original_download_sha256"] == f"{2:064x}"
        assert conversation["source"]["face_bbox_count"] == 0

    first_hash = manifest["output_shards"][0]["sha256"]
    second_output = tmp_path / "output-second"
    second_manifest = module["convert"](
        source,
        second_output,
        max_count=1,
        validation_fraction=0.0,
        minimum_train_samples=2,
    )
    assert second_manifest["output_shards"][0]["sha256"] == first_hash


def test_convert_rejects_nonempty_output_and_enforces_train_minimum(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_source_shard(source / "00000000.tar")

    output = tmp_path / "output"
    output.mkdir()
    (output / "existing").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        module["convert"](
            source,
            output,
            max_count=10,
            validation_fraction=0.0,
            minimum_train_samples=0,
        )

    with pytest.raises(RuntimeError, match="require at least 3"):
        module["convert"](
            source,
            tmp_path / "too-small",
            max_count=10,
            validation_fraction=0.0,
            minimum_train_samples=3,
        )

    limited_manifest = module["convert"](
        source,
        tmp_path / "limited",
        max_count=10,
        validation_fraction=0.0,
        minimum_train_samples=1,
        maximum_samples=1,
    )
    assert limited_manifest["counts"]["input_samples"] == 1
    assert limited_manifest["counts"]["train"] == 1
    assert limited_manifest["maximum_samples"] == 1


def test_prepare_energon_dataset_indexes_nonempty_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    calls = []
    prepare = module["prepare_energon_dataset"]
    prepare_module = types.ModuleType("megatron.bridge.data.energon")
    prepare_module.prepare_webdataset = lambda path, patterns, *, num_workers: calls.append(
        (path, patterns, num_workers)
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge.data.energon", prepare_module)

    prepare(tmp_path, counts={"train": 10, "val": 0}, num_workers=4)
    prepare(tmp_path, counts={"train": 0, "val": 10}, num_workers=2)

    assert calls == [
        (tmp_path, {"train": "train-shard-.*"}, 4),
        (tmp_path, {"val": "val-shard-.*"}, 2),
    ]
    dataset_yaml = (tmp_path / ".nv-meta" / "dataset.yaml").read_text(encoding="utf-8")
    assert "ChatMLWebdataset" in dataset_yaml
    assert "imgs: image.jpg" in dataset_yaml

    with pytest.raises(ValueError, match="At least one converted split"):
        prepare(tmp_path, counts={"train": 0, "val": 0}, num_workers=1)
