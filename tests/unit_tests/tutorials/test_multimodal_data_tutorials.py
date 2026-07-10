# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
import runpy
import struct
import tarfile
from pathlib import Path

import pytest
from PIL import Image


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).parents[3]
HF_MULTIMODAL_TUTORIAL = REPO_ROOT / "tutorials" / "data" / "hf-multimodal"
ENERGON_TUTORIAL = REPO_ROOT / "tutorials" / "data" / "energon"
QWEN_README = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_vl" / "README.md"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hf_multimodal_preparation_writes_resolvable_qwen_rows(tmp_path: Path):
    from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, HFDatasetSourceConfig
    from megatron.bridge.data.builders.direct_hf_sft import load_direct_hf_sft_examples

    module = runpy.run_path(str(HF_MULTIMODAL_TUTORIAL / "prepare_example_data.py"))

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

    source = HFDatasetSourceConfig(
        path_or_dataset="json",
        split="train",
        load_kwargs={"data_files": {"train": str(tmp_path / "training.jsonl")}},
    )
    loaded = load_direct_hf_sft_examples(source, ChatSFTPreprocessingConfig())
    assert len(loaded) == 4
    assert loaded[0]["conversation"][0]["content"][0]["type"] == "image"
    assert Path(loaded[0]["conversation"][0]["content"][0]["image"]).is_file()


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


def test_energon_preparation_uses_api_with_explicit_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = runpy.run_path(str(ENERGON_TUTORIAL / "prepare_example_data.py"))
    calls = []
    run_energon_prepare = module["run_energon_prepare"]
    monkeypatch.setitem(
        run_energon_prepare.__globals__,
        "prepare_webdataset",
        lambda path, patterns, *, num_workers: calls.append((path, patterns, num_workers)),
    )

    run_energon_prepare(tmp_path, num_workers=3)

    assert calls == [
        (
            tmp_path,
            {"train": "train-shard-.*", "val": "val-shard-.*"},
            3,
        )
    ]


def test_medpix_energon_preparation_writes_real_schema_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = runpy.run_path(str(ENERGON_TUTORIAL / "prepare_medpix_data.py"))
    calls = []

    def fake_load_dataset(dataset_id: str, *, split: str):
        calls.append((dataset_id, split))
        count = 2 if split.startswith("train") else 1
        return [
            {
                "image_id": Image.new("RGB", (16 + index, 12), (20 * index, 40, 80)),
                "question": f"Question {index}?",
                "answer": f"Answer {index}.",
            }
            for index in range(count)
        ]

    prepare_medpix_data = module["prepare_medpix_data"]
    monkeypatch.setitem(prepare_medpix_data.__globals__, "load_dataset", fake_load_dataset)
    prepare_medpix_data(
        tmp_path,
        train_rows=2,
        validation_rows=1,
        run_prepare=False,
    )

    assert calls == [
        ("mmoukouba/MedPix-VQA", "train[:2]"),
        ("mmoukouba/MedPix-VQA", "validation[:1]"),
    ]
    with tarfile.open(tmp_path / "train-shard-000000.tar") as archive:
        assert archive.getnames() == [
            "train-000000.image.png",
            "train-000000.conversation.json",
            "train-000001.image.png",
            "train-000001.conversation.json",
        ]
        conversation = json.load(archive.extractfile("train-000000.conversation.json"))
    assert conversation[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "Question 0?"},
    ]
    assert conversation[1]["content"] == [{"type": "text", "text": "Answer 0."}]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "dataset": "mmoukouba/MedPix-VQA",
        "train_split": "train[:2]",
        "validation_split": "validation[:1]",
        "train_rows": 2,
        "validation_rows": 1,
    }
    dataset_yaml = (tmp_path / ".nv-meta" / "dataset.yaml").read_text(encoding="utf-8")
    assert "imgs: image.png" in dataset_yaml
    assert "conversation: conversation.json" in dataset_yaml


def test_medpix_training_slice_override_preserves_hydra_brackets():
    from omegaconf import OmegaConf

    from megatron.bridge.training.utils.omegaconf_utils import parse_hydra_overrides

    config = OmegaConf.create({"dataset": {"source": {"dataset_name": "medpix", "split": None}}})
    result = parse_hydra_overrides(config, ['dataset.source.split="train[:16]"'])

    assert result.dataset.source.split == "train[:16]"


def test_multimodal_tutorials_document_runnable_qwen_paths():
    hf_multimodal = (HF_MULTIMODAL_TUTORIAL / "README.md").read_text(encoding="utf-8")
    energon = (ENERGON_TUTORIAL / "README.md").read_text(encoding="utf-8")
    qwen = QWEN_README.read_text(encoding="utf-8")

    assert "qwen3_vl_8b_peft_config" in hf_multimodal
    assert "## Start with a hosted chat dataset" in hf_multimodal
    assert 'HFDatasetSourceConfig(dataset_name="medpix")' in hf_multimodal
    assert "--nproc_per_node=1" in hf_multimodal
    assert "--dataset vlm-hf" in hf_multimodal
    assert "dataset.source.path_or_dataset=json" in hf_multimodal
    assert "dataset.source.load_kwargs={data_files:{train:" in hf_multimodal
    assert "dataset.source.dataset_name=medpix" in hf_multimodal
    assert 'dataset.source.split="train[:16]"' in hf_multimodal
    assert "logger.wandb_project=bridge-qwen3-vl-medpix" in hf_multimodal
    assert "--step_func vlm_step" in hf_multimodal
    assert "dataset.defer_in_batch_packing_to_step=False" in hf_multimodal
    assert "dataset.defer_in_batch_packing_to_step=True" not in hf_multimodal
    assert "qwen3_vl_8b_peft_energon_config" in energon
    assert "--dataset vlm-energon" in energon
    assert "--step_func vlm_step" in energon
    assert "dataset.num_workers=2" in energon
    assert "dataset.num_val_workers=2" in energon
    assert "dataset.defer_in_batch_packing_to_step=False" in energon
    assert "dataset.defer_in_batch_packing_to_step=True" not in energon
    assert '"train": "train-shard-.*"' in energon
    assert "prepare_medpix_data.py" in energon
    assert "dataset.enable_in_batch_packing=True" in energon
    assert "min_pixels` and `max_pixels` are not visual keys" in energon
    assert "Hugging Face multimodal tutorial" in qwen
    assert "multimodal Energon tutorial" in qwen
    assert "qwen3_vl_8b_finetune_config" not in qwen
    assert "megatron.bridge.models.qwen_vl.data.energon" not in qwen
