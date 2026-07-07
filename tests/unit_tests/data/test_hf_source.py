# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data import hf_source as source_module
from megatron.bridge.data.hf_datasets import adapters as adapter_module
from megatron.bridge.data.hf_datasets.adapters import adapt_hf_dataset, prepare_hf_dataset_for_adapter
from megatron.bridge.data.hf_source import HFDatasetSourceConfig, load_hf_dataset_source


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


def test_native_messages_source_needs_no_adapter():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "id": 1,
        }
    ]

    assert adapt_hf_dataset(rows, adapter_name=None) == rows


def test_squad_adapter_normalizes_loaded_rows():
    rows = [{"context": "ctx", "question": "q", "answers": {"text": ["a"]}}]

    adapted = adapt_hf_dataset(rows, adapter_name="squad")

    assert adapted[0]["messages"][-1] == {"role": "assistant", "content": "a"}
    assert adapted[0]["original_answers"] == ["a"]


@pytest.mark.parametrize(
    "ground_truth",
    [
        '{"gt_parses": [{"name": "receipt"}]}',
        '{"gt_parse": {"name": "receipt"}}',
    ],
)
def test_cord_adapter_supports_plural_and_singular_ground_truth(ground_truth):
    rows = [{"ground_truth": ground_truth, "image": object()}]

    adapted = adapt_hf_dataset(rows, adapter_name="cord_v2")

    assert adapted[0]["conversation"][0]["content"][0]["type"] == "image"
    assert adapted[0]["conversation"][1]["role"] == "assistant"


def test_source_rejects_runtime_load_objects():
    source = HFDatasetSourceConfig(path_or_dataset="org/chat", load_kwargs={"transform": lambda row: row})

    with pytest.raises(TypeError, match="declarative values"):
        source.validate()


def test_audio_adapter_preparation_disables_automatic_decoding():
    class _Dataset:
        cast = None

        def cast_column(self, column, feature):
            self.cast = (column, feature.decode)
            return self

    dataset = _Dataset()

    assert prepare_hf_dataset_for_adapter(dataset, adapter_name="cv17") is dataset
    assert dataset.cast == ("audio", False)


def test_cv17_and_default_audio_preserve_legacy_text_defaults(monkeypatch):
    monkeypatch.setattr(adapter_module, "_decode_audio", lambda audio: ([0.0], 16_000))

    cv17 = adapt_hf_dataset(
        [{"audio": {}, "transcription": "iki kelime"}],
        adapter_name="cv17",
    )
    generic = adapt_hf_dataset(
        [{"audio": {}, "text": "two words"}],
        adapter_name="default_audio",
    )

    assert cv17[0]["conversation"][-1]["content"][0]["text"] == "iki kelime"
    assert generic[0]["conversation"][-1]["content"][0]["text"] == "twowords"
