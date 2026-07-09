# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
import runpy
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[3]
TEXT_ONLY_SFT_TUTORIAL = REPO_ROOT / "tutorials/data/text-only-sft"
DIRECT_HF_SFT_TUTORIAL = REPO_ROOT / "tutorials/data/direct-hf-sft"


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_text_only_sft_prepare_example_data(tmp_path):
    module = runpy.run_path(str(TEXT_ONLY_SFT_TUTORIAL / "prepare_example_data.py"))
    module["prepare_example_data"](tmp_path)

    assert [path.name for path in sorted(tmp_path.glob("*.jsonl"))] == [
        "test.jsonl",
        "training.jsonl",
        "validation.jsonl",
    ]
    assert len(_load_rows(tmp_path / "training.jsonl")) == 4
    assert all(set(row) == {"input", "output"} for row in _load_rows(tmp_path / "training.jsonl"))


def test_direct_hf_sft_prepare_example_data(tmp_path):
    module = runpy.run_path(str(DIRECT_HF_SFT_TUTORIAL / "prepare_example_data.py"))
    module["prepare_example_data"](tmp_path)

    rows = _load_rows(tmp_path / "training.jsonl")
    assert len(rows) == 4
    assert all([message["role"] for message in row["messages"]] == ["user", "assistant"] for row in rows)


@pytest.mark.parametrize(
    ("tutorial", "config_name", "builder_name"),
    [
        (TEXT_ONLY_SFT_TUTORIAL, "GPTSFTDatasetConfig", "GPTSFTDatasetBuilder"),
        (DIRECT_HF_SFT_TUTORIAL, "DirectHFSFTDatasetConfig", "DirectHFSFTDatasetBuilder"),
    ],
)
def test_text_dataset_tutorial_uses_config_builder_primary_path(tutorial, config_name, builder_name):
    readme = (tutorial / "README.md").read_text(encoding="utf-8")

    assert config_name in readme
    assert builder_name in readme
    assert "prepare_example_data.py" in readme
    assert "training.jsonl" in readme
    assert "validation.jsonl" in readme
    assert "HFConversationDatasetProvider(" not in readme


def test_text_only_sft_tutorial_uses_standard_distributed_launcher():
    readme = (TEXT_ONLY_SFT_TUTORIAL / "README.md").read_text(encoding="utf-8")
    assert "uv run python -m torch.distributed.run" in readme


def test_direct_hf_sft_tutorial_documents_hosted_native_messages_source():
    readme = (DIRECT_HF_SFT_TUTORIAL / "README.md").read_text(encoding="utf-8")
    assert "HuggingFaceH4/ultrachat_200k" in readme
    assert 'split="train_sft"' in readme
    assert "schema_adapter" in readme


def test_direct_hf_sft_tutorial_routes_json_through_hf_loader():
    readme = (DIRECT_HF_SFT_TUTORIAL / "README.md").read_text(encoding="utf-8")

    assert 'path_or_dataset="json"' in readme
    assert 'load_kwargs={"data_files"' in readme
    assert "LocalConversationDatasetSourceConfig" not in readme
    assert "vlm-local" not in readme
    assert "vlm-preloaded" in readme  # Migration guidance names the removed selector.
