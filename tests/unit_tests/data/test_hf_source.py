# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data import hf_source as source_module
from megatron.bridge.data.hf_source import HFDatasetSourceConfig, load_and_adapt_hf_dataset, load_hf_dataset_source


pytestmark = pytest.mark.unit


def test_source_keeps_loading_and_schema_adaptation_separate(monkeypatch):
    calls = []

    def _load_dataset(path, subset, *, split, **kwargs):
        calls.append((path, subset, split, kwargs))
        return [{"messages": [{"role": "assistant", "content": "ok"}]}]

    monkeypatch.setattr(source_module, "load_dataset", _load_dataset)
    source = HFDatasetSourceConfig(
        path_or_dataset="org/chat",
        subset="default",
        split="train_sft",
        load_kwargs={"revision": "main"},
        adapter_kwargs={"messages_column": "messages"},
    )

    dataset = load_hf_dataset_source(source)

    assert calls == [("org/chat", "default", "train_sft", {"revision": "main"})]
    assert dataset[0]["messages"][0]["content"] == "ok"


def test_source_rejects_runtime_load_objects():
    source = HFDatasetSourceConfig(path_or_dataset="org/chat", load_kwargs={"transform": lambda row: row})

    with pytest.raises(TypeError, match="declarative values"):
        source.validate()


def test_source_rejects_unknown_adapter_during_validation():
    source = HFDatasetSourceConfig(path_or_dataset="org/chat", schema_adapter="missing")

    with pytest.raises(ValueError, match="Unknown Hugging Face schema adapter"):
        source.validate()


def test_load_and_adapt_composes_source_loader_and_adapter(monkeypatch):
    rows = [{"context": "ctx", "question": "q", "answers": {"text": ["a"]}}]
    monkeypatch.setattr(source_module, "load_hf_dataset_source", lambda source: rows)
    source = HFDatasetSourceConfig(path_or_dataset="org/squad", schema_adapter="squad")

    adapted = load_and_adapt_hf_dataset(source)

    assert adapted[0]["messages"][-1] == {"role": "assistant", "content": "a"}


def test_source_loads_and_concatenates_multiple_subsets(monkeypatch):
    calls = []

    def _load_dataset(path, subset, *, split, **kwargs):
        calls.append((path, subset, split, kwargs))
        return [{"subset": subset}]

    monkeypatch.setattr(source_module, "load_dataset", _load_dataset)
    monkeypatch.setattr(source_module, "concatenate_datasets", lambda datasets: [row for ds in datasets for row in ds])
    source = HFDatasetSourceConfig(
        path_or_dataset="org/multi",
        subset=["first", "second"],
        split="train",
    )

    dataset = load_hf_dataset_source(source)

    assert calls == [
        ("org/multi", "first", "train", {}),
        ("org/multi", "second", "train", {}),
    ]
    assert dataset == [{"subset": "first"}, {"subset": "second"}]
