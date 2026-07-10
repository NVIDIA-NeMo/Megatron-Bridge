# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
import runpy
import struct
import tarfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).parents[3]
DIRECT_TUTORIAL = REPO_ROOT / "tutorials" / "data" / "multimodal-direct"
ENERGON_TUTORIAL = REPO_ROOT / "tutorials" / "data" / "energon"
QWEN_README = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_vl" / "README.md"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_multimodal_direct_preparation_writes_resolvable_qwen_rows(tmp_path: Path):
    module = runpy.run_path(str(DIRECT_TUTORIAL / "prepare_example_data.py"))

    module["prepare_example_data"](tmp_path)

    assert [path.name for path in sorted(tmp_path.glob("*.jsonl"))] == [
        "test.jsonl",
        "training.jsonl",
        "validation.jsonl",
    ]
    assert [len(_load_jsonl(tmp_path / f"{split}.jsonl")) for split in ("training", "validation", "test")] == [
        4,
        1,
        1,
    ]

    for row in _load_jsonl(tmp_path / "training.jsonl"):
        user_content = row["messages"][0]["content"]
        image_part, text_part = user_content
        image_path = Path(image_part["image"])
        assert image_part["type"] == "image"
        assert text_part["type"] == "text"
        assert image_path.is_absolute() and image_path.is_file()
        png = image_path.read_bytes()
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert struct.unpack(">II", png[16:24]) == (448, 448)


def test_energon_preparation_writes_matching_tar_members_and_loader(tmp_path: Path):
    module = runpy.run_path(str(ENERGON_TUTORIAL / "prepare_example_data.py"))

    module["prepare_example_data"](tmp_path, run_prepare=False)

    train_tar = tmp_path / "train-shard-000000.tar"
    val_tar = tmp_path / "val-shard-000000.tar"
    assert train_tar.is_file() and val_tar.is_file()
    with tarfile.open(train_tar) as archive:
        names = archive.getnames()
        assert len(names) == 8
        assert names[0::2] == [f"train-{index:06d}.image.png" for index in range(4)]
        assert names[1::2] == [f"train-{index:06d}.conversation.json" for index in range(4)]
        conversation = json.load(archive.extractfile("train-000000.conversation.json"))
    assert conversation[0]["content"][0] == {"type": "image"}
    assert conversation[-1]["role"] == "assistant"

    dataset_yaml = (tmp_path / ".nv-meta" / "dataset.yaml").read_text(encoding="utf-8")
    assert "megatron.bridge.data.energon.task_encoder_utils" in dataset_yaml
    assert "imgs: image.png" in dataset_yaml
    assert "conversation: conversation.json" in dataset_yaml


def test_energon_preparation_uses_noninteractive_explicit_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = runpy.run_path(str(ENERGON_TUTORIAL / "prepare_example_data.py"))
    calls = []
    monkeypatch.setattr(module["subprocess"], "run", lambda command, check: calls.append((command, check)))

    module["run_energon_prepare"](tmp_path, num_workers=3)

    assert calls == [
        (
            [
                "energon",
                "prepare",
                str(tmp_path),
                "--non-interactive",
                "--num-workers",
                "3",
                "--split-parts",
                "train:train-shard-.*",
                "--split-parts",
                "val:val-shard-.*",
                "--skip-dataset-yaml",
                "--force-overwrite",
            ],
            True,
        )
    ]


def test_multimodal_tutorials_document_runnable_qwen_paths():
    direct = (DIRECT_TUTORIAL / "README.md").read_text(encoding="utf-8")
    energon = (ENERGON_TUTORIAL / "README.md").read_text(encoding="utf-8")
    qwen = QWEN_README.read_text(encoding="utf-8")

    assert "qwen3_vl_8b_peft_config" in direct
    assert "--nproc_per_node=1" in direct
    assert "dataset.source.path_or_dataset=json" in direct
    assert "dataset.defer_in_batch_packing_to_step=True" in direct
    assert "qwen3_vl_8b_peft_energon_config" in energon
    assert "--dataset vlm-energon" in energon
    assert "train:train-shard-.*" in energon
    assert "min_pixels` and `max_pixels` are not visual keys" in energon
    assert "multimodal Direct-HF tutorial" in qwen
    assert "multimodal Energon tutorial" in qwen
    assert "qwen3_vl_8b_finetune_config" not in qwen
    assert "megatron.bridge.models.qwen_vl.data.energon" not in qwen
