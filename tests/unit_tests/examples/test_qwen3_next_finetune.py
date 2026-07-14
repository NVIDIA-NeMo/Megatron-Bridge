# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

from megatron.bridge.data.builders import GPTSFTDatasetConfig, HFDatasetSourceConfig


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
