# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``scripts/training/pack_sft_data.py``."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "training" / "pack_sft_data.py"


@dataclass
class _DataloaderConfig:
    micro_batch_size: int = 1


@dataclass
class _PackedSequenceSpecs:
    packed_sequence_size: int = 2048
    pad_seq_to_mult: int = 8
    num_tokenizer_workers: int = -1


@dataclass
class _DatasetConfig:
    seq_length: int = 2048
    seed: int = 123
    dataset_kwargs: dict[str, object] = field(default_factory=lambda: {"chat": "template"})
    enable_offline_packing: bool = True
    offline_packing_specs: _PackedSequenceSpecs | None = field(default_factory=_PackedSequenceSpecs)


@dataclass
class _RecipeConfig:
    dataset: _DatasetConfig | None = field(default_factory=_DatasetConfig)
    tokenizer: object = "tokenizer-config"


class _FinetuningDatasetBuilder:
    pack_metadata = Path("default-pack-metadata.json")

    def __init__(self, *, tokenizer: object, **kwargs: object) -> None:
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def prepare_packed_data(self) -> None:
        raise AssertionError("explicit pack-path tests should not call prepare_packed_data")


def _load_module():
    spec = importlib.util.spec_from_file_location("pack_sft_data_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _install_pack_sft_stubs(monkeypatch: pytest.MonkeyPatch, recipe_fn) -> Mock:
    megatron = sys.modules.get("megatron", types.ModuleType("megatron"))
    bridge = types.ModuleType("megatron.bridge")
    data = types.ModuleType("megatron.bridge.data")
    builders = types.ModuleType("megatron.bridge.data.builders")
    datasets = types.ModuleType("megatron.bridge.data.datasets")
    training = types.ModuleType("megatron.bridge.training")
    tokenizers = types.ModuleType("megatron.bridge.training.tokenizers")
    recipes = types.ModuleType("megatron.bridge.recipes")
    recipes.unit_recipe = recipe_fn

    finetuning_dataset = types.ModuleType("megatron.bridge.data.builders.finetuning_dataset")
    finetuning_dataset.FinetuningDatasetBuilder = _FinetuningDatasetBuilder

    packed_sequence = types.ModuleType("megatron.bridge.data.datasets.packed_sequence")
    prepare_packed_sequence_data = Mock()
    packed_sequence.prepare_packed_sequence_data = prepare_packed_sequence_data

    training_config = types.ModuleType("megatron.bridge.training.config")
    training_config.DataloaderConfig = _DataloaderConfig

    tokenizer_module = types.ModuleType("megatron.bridge.training.tokenizers.tokenizer")
    tokenizer_module.build_tokenizer = Mock(return_value="tokenizer")

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    monkeypatch.setitem(sys.modules, "megatron.bridge.data", data)
    monkeypatch.setitem(sys.modules, "megatron.bridge.data.builders", builders)
    monkeypatch.setitem(sys.modules, "megatron.bridge.data.datasets", datasets)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training", training)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.tokenizers", tokenizers)
    monkeypatch.setitem(sys.modules, "megatron.bridge.recipes", recipes)
    monkeypatch.setitem(sys.modules, "megatron.bridge.data.builders.finetuning_dataset", finetuning_dataset)
    monkeypatch.setitem(sys.modules, "megatron.bridge.data.datasets.packed_sequence", packed_sequence)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.config", training_config)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.tokenizers.tokenizer", tokenizer_module)
    monkeypatch.setattr(megatron, "bridge", bridge, raising=False)
    monkeypatch.setattr(bridge, "recipes", recipes, raising=False)
    monkeypatch.setattr(bridge, "data", data, raising=False)
    monkeypatch.setattr(data, "builders", builders, raising=False)
    monkeypatch.setattr(data, "datasets", datasets, raising=False)
    monkeypatch.setattr(builders, "finetuning_dataset", finetuning_dataset, raising=False)
    monkeypatch.setattr(datasets, "packed_sequence", packed_sequence, raising=False)
    monkeypatch.setattr(bridge, "training", training, raising=False)
    monkeypatch.setattr(training, "config", training_config, raising=False)
    monkeypatch.setattr(training, "tokenizers", tokenizers, raising=False)
    monkeypatch.setattr(tokenizers, "tokenizer", tokenizer_module, raising=False)
    return prepare_packed_sequence_data


def test_pack_sft_data_rejects_unsupported_seq_length(monkeypatch):
    module = _load_module()

    def unit_recipe():
        raise AssertionError("recipe should not run when explicit override cannot be forwarded")

    _install_pack_sft_stubs(monkeypatch, unit_recipe)
    monkeypatch.setattr(sys, "argv", ["pack_sft_data.py", "--recipe", "unit_recipe", "--seq-length", "4096"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert str(exc_info.value) == "Error: recipe 'unit_recipe' does not accept a 'seq_length' parameter."


def test_pack_sft_data_rejects_unsupported_hf_path(monkeypatch):
    module = _load_module()

    def unit_recipe():
        raise AssertionError("recipe should not run when explicit override cannot be forwarded")

    _install_pack_sft_stubs(monkeypatch, unit_recipe)
    monkeypatch.setattr(sys, "argv", ["pack_sft_data.py", "--recipe", "unit_recipe", "--hf-path", "nvidia/unit"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert str(exc_info.value) == "Error: recipe 'unit_recipe' does not accept an 'hf_path' parameter."


def test_pack_sft_data_forwards_supported_overrides_and_explicit_paths(monkeypatch, tmp_path):
    module = _load_module()
    recipe_calls = []

    def unit_recipe(seq_length: int, hf_path: str):
        recipe_calls.append({"seq_length": seq_length, "hf_path": hf_path})
        return _RecipeConfig(dataset=_DatasetConfig(seq_length=seq_length))

    prepare_packed_sequence_data = _install_pack_sft_stubs(monkeypatch, unit_recipe)

    train_input = tmp_path / "train.jsonl"
    train_output = tmp_path / "packed-train.parquet"
    metadata_output = tmp_path / "metadata.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pack_sft_data.py",
            "--recipe",
            "unit_recipe",
            "--seq-length",
            "4096",
            "--hf-path",
            "nvidia/unit",
            "--train-input-path",
            str(train_input),
            "--packed-train-data-path",
            str(train_output),
            "--packed-metadata-path",
            str(metadata_output),
        ],
    )

    module.main()

    assert recipe_calls == [{"seq_length": 4096, "hf_path": "nvidia/unit"}]
    prepare_packed_sequence_data.assert_called_once()
    _, kwargs = prepare_packed_sequence_data.call_args
    assert kwargs["input_path"] == train_input
    assert kwargs["output_path"] == train_output
    assert kwargs["output_metadata_path"] == metadata_output
    assert kwargs["packed_sequence_size"] == 2048
    assert kwargs["max_seq_length"] == 4096
    assert kwargs["num_tokenizer_workers"] == 1
