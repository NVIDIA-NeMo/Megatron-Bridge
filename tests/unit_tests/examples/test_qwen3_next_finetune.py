# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

import megatron.bridge.data.builders.gpt_sft as gpt_sft_builder
from megatron.bridge.data.builders import (
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)


pytestmark = pytest.mark.unit

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_next" / "finetune_qwen3_next_80b_a3b.py"


def _load_example_module(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_data_path_replaces_named_source_with_custom_json_source():
    name = "qwen3_next_finetune_custom_data"
    try:
        module = _load_example_module(name)
        config = SimpleNamespace(
            dataset=GPTSFTDatasetConfig(
                seq_length=2048,
                hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
                hf_validation_proportion=0.1,
                do_test=False,
            )
        )

        module._replace_with_custom_data_path(config, "/data/custom.jsonl")
        config.dataset.validate()

        assert config.dataset.hf_dataset.dataset_name is None
        assert config.dataset.hf_dataset.path_or_dataset == "json"
        assert config.dataset.hf_dataset.load_kwargs == {"data_files": "/data/custom.jsonl"}
        assert config.dataset.dataset_kwargs is None
        assert config.dataset.hf_validation_proportion == 0.1
    finally:
        sys.modules.pop(name, None)


def test_data_path_materializes_advertised_messages_schema(monkeypatch, tmp_path):
    name = "qwen3_next_finetune_messages_data"
    try:
        module = _load_example_module(name)
        config = SimpleNamespace(
            dataset=GPTSFTDatasetConfig(
                seq_length=2048,
                hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
                hf_validation_proportion=0.1,
                do_test=False,
                preprocessing=PromptCompletionSFTPreprocessingConfig(
                    prompt_column="input",
                    completion_column="output",
                    separator=" ",
                ),
            )
        )
        messages_rows = [
            {
                "id": index,
                "messages": [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ],
            }
            for index in range(10)
        ]
        monkeypatch.setattr(gpt_sft_builder, "load_and_adapt_hf_dataset", lambda _: messages_rows)

        module._replace_with_custom_data_path(config, "/data/custom.jsonl")
        gpt_sft_builder.materialize_hf_dataset(config.dataset, tmp_path)

        materialized_rows = []
        for split in ("training", "validation"):
            with (tmp_path / f"{split}.jsonl").open(encoding="utf-8") as input_file:
                materialized_rows.extend(json.loads(line) for line in input_file)
        assert len(materialized_rows) == len(messages_rows)
        assert all("conversation" in row and "messages" not in row for row in materialized_rows)
    finally:
        sys.modules.pop(name, None)
