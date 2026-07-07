import json
import runpy
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[3]
GPT_SFT_TUTORIAL = REPO_ROOT / "tutorials/data/gpt-sft"
HF_CONVERSATION_TUTORIAL = REPO_ROOT / "tutorials/data/hf-conversation"


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_gpt_sft_prepare_example_data(tmp_path):
    module = runpy.run_path(str(GPT_SFT_TUTORIAL / "prepare_example_data.py"))
    module["prepare_example_data"](tmp_path)

    assert [path.name for path in sorted(tmp_path.glob("*.jsonl"))] == [
        "test.jsonl",
        "training.jsonl",
        "validation.jsonl",
    ]
    assert len(_load_rows(tmp_path / "training.jsonl")) == 4
    assert all(set(row) == {"input", "output"} for row in _load_rows(tmp_path / "training.jsonl"))


def test_hf_conversation_prepare_example_data(tmp_path):
    module = runpy.run_path(str(HF_CONVERSATION_TUTORIAL / "prepare_example_data.py"))
    module["prepare_example_data"](tmp_path)

    rows = _load_rows(tmp_path / "training.jsonl")
    assert len(rows) == 4
    assert all([message["role"] for message in row["messages"]] == ["user", "assistant"] for row in rows)


@pytest.mark.parametrize(
    ("tutorial", "config_name", "builder_name"),
    [
        (GPT_SFT_TUTORIAL, "GPTSFTDatasetConfig", "GPTSFTDatasetBuilder"),
        (HF_CONVERSATION_TUTORIAL, "HFConversationDatasetConfig", "HFConversationDatasetBuilder"),
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


def test_gpt_sft_tutorial_uses_standard_distributed_launcher():
    readme = (GPT_SFT_TUTORIAL / "README.md").read_text(encoding="utf-8")
    assert "uv run python -m torch.distributed.run" in readme
