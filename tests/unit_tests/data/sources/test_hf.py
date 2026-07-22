# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.sources import hf as source_module
from megatron.bridge.data.sources.hf import (
    HFDatasetSourceConfig,
    hf_dataset_supports_split,
    load_and_adapt_hf_dataset,
    load_hf_dataset_source,
    prepare_hf_dataset_sources,
    resolve_hf_dataset_source,
)


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


@pytest.mark.parametrize("field_name", ["load_kwargs", "adapter_kwargs"])
def test_source_rejects_non_mapping_kwargs(field_name):
    source = HFDatasetSourceConfig(path_or_dataset="org/chat", **{field_name: ["not", "a", "mapping"]})

    with pytest.raises(TypeError, match=f"{field_name} must be a mapping"):
        source.validate()


def test_source_rejects_unknown_adapter_during_validation():
    source = HFDatasetSourceConfig(path_or_dataset="org/chat", schema_adapter="missing")

    with pytest.raises(ValueError, match="Unknown Hugging Face schema adapter"):
        source.validate()


@pytest.mark.parametrize(
    "source",
    [
        HFDatasetSourceConfig(),
        HFDatasetSourceConfig(dataset_name="squad", path_or_dataset="rajpurkar/squad"),
    ],
)
def test_source_requires_exactly_one_source_mode(source):
    with pytest.raises(ValueError, match="Exactly one Hugging Face source"):
        source.validate()


def test_source_rejects_unknown_dataset_preset():
    source = HFDatasetSourceConfig(dataset_name="missing")

    with pytest.raises(ValueError, match="Unknown Hugging Face dataset preset"):
        source.validate()


def test_named_source_rejects_non_string_split():
    source = HFDatasetSourceConfig(dataset_name="squad", split=1)

    with pytest.raises(ValueError, match="split must be non-empty"):
        source.validate()


def test_named_source_owns_subset_and_adapter():
    source = HFDatasetSourceConfig(dataset_name="raven", subset="other")

    with pytest.raises(ValueError, match="presets own subset and schema_adapter"):
        source.validate()


@pytest.mark.parametrize("video_root_path", [None, "", "   "])
def test_named_source_validates_required_adapter_kwargs(video_root_path):
    adapter_kwargs = None if video_root_path is None else {"video_root_path": video_root_path}
    source = HFDatasetSourceConfig(dataset_name="llava_video_178k", adapter_kwargs=adapter_kwargs)

    with pytest.raises(ValueError, match="requires adapter_kwargs: video_root_path"):
        source.validate()


def test_llava_video_preset_maps_train_to_physical_split():
    source = HFDatasetSourceConfig(
        dataset_name="llava_video_178k",
        split="train",
        adapter_kwargs={"video_root_path": "/videos"},
    )

    assert resolve_hf_dataset_source(source).split == "open_ended"


def test_named_source_validates_each_compound_split_component():
    source = HFDatasetSourceConfig(dataset_name="cord_v2", split="train[:10%]+test")

    assert resolve_hf_dataset_source(source).split == "train[:10%]+test"

    with pytest.raises(ValueError, match="only supports split"):
        HFDatasetSourceConfig(dataset_name="raven", split="train[:10%]+test").validate()


@pytest.mark.parametrize("dataset_name", ["raven", "llava_video_178k"])
def test_train_only_presets_reject_validation_split(dataset_name):
    adapter_kwargs = {"video_root_path": "/videos"} if dataset_name == "llava_video_178k" else None
    source = HFDatasetSourceConfig(
        dataset_name=dataset_name,
        split="validation",
        adapter_kwargs=adapter_kwargs,
    )

    with pytest.raises(ValueError, match="only supports split"):
        source.validate()


@pytest.mark.parametrize(
    ("dataset_name", "split", "expected"),
    [
        ("cord_v2", "test", True),
        ("medpix", "validation", True),
        ("medpix", "test", False),
        ("rdr", "validation", False),
    ],
)
def test_named_source_declares_supported_splits(dataset_name, split, expected):
    source = HFDatasetSourceConfig(dataset_name=dataset_name)

    assert hf_dataset_supports_split(source, split) is expected


def test_source_rejects_load_kwargs_that_duplicate_source_fields():
    source = HFDatasetSourceConfig(path_or_dataset="org/chat", load_kwargs={"split": "test"})

    with pytest.raises(ValueError, match="must not override source fields: split"):
        source.validate()


@pytest.mark.parametrize(
    "data_files",
    [None, "", [], {}, {"train": None}, {"train": ""}],
)
def test_source_rejects_unset_data_files(data_files):
    source = HFDatasetSourceConfig(path_or_dataset="json", load_kwargs={"data_files": data_files})

    with pytest.raises(ValueError, match="data_files must contain non-empty paths"):
        source.validate()


def test_named_source_resolves_physical_source_and_declarative_overrides():
    source = HFDatasetSourceConfig(
        dataset_name="raven",
        split="train[:10%]",
        load_kwargs={"revision": "main"},
        adapter_kwargs={"prompt": "Solve the puzzle."},
    )

    resolved = resolve_hf_dataset_source(source)

    assert resolved.dataset_name is None
    assert resolved.path_or_dataset == "HuggingFaceM4/the_cauldron"
    assert resolved.subset == "raven"
    assert resolved.split == "train[:10%]"
    assert resolved.schema_adapter == "raven"
    assert resolved.load_kwargs == {"revision": "main"}
    assert resolved.adapter_kwargs == {"prompt": "Solve the puzzle."}


def test_named_source_uses_preset_split_and_adapter(monkeypatch):
    calls = []

    def _load_dataset(path, subset, *, split, **kwargs):
        calls.append((path, subset, split, kwargs))
        return [{"question": "1+1", "answer": "work #### 2"}]

    monkeypatch.setattr(source_module, "load_dataset", _load_dataset)

    adapted = load_and_adapt_hf_dataset(HFDatasetSourceConfig(dataset_name="gsm8k"))

    assert calls == [("openai/gsm8k", "main", "train", {})]
    assert adapted[0]["original_answers"] == ["2"]


def test_tulu3_preset_uses_current_native_chat_dataset(monkeypatch):
    rows = [
        {
            "id": "example-1",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "source": "example-source",
        }
    ]
    calls = []

    def _load_dataset(path, *, split, **kwargs):
        calls.append((path, split, kwargs))
        return rows

    monkeypatch.setattr(source_module, "load_dataset", _load_dataset)
    source = HFDatasetSourceConfig(dataset_name="tulu3")

    resolved = resolve_hf_dataset_source(source)
    adapted = load_and_adapt_hf_dataset(source)

    assert resolved.path_or_dataset == "allenai/tulu-3-sft-mixture"
    assert resolved.split == "train"
    assert resolved.schema_adapter is None
    assert calls == [("allenai/tulu-3-sft-mixture", "train", {})]
    assert adapted == rows


def test_load_and_adapt_composes_source_loader_and_adapter(monkeypatch):
    rows = [{"context": "ctx", "question": "q", "answers": {"text": ["a"]}}]
    monkeypatch.setattr(source_module, "load_hf_dataset_source", lambda source: rows)
    source = HFDatasetSourceConfig(path_or_dataset="org/squad", schema_adapter="squad")

    adapted = load_and_adapt_hf_dataset(source)

    assert adapted[0]["prompt"] == "Context: ctx Question: q Answer:"
    assert adapted[0]["completion"] == "a"


def test_json_data_files_use_hugging_face_loader(monkeypatch):
    calls = []

    def _load_dataset(path, *, split, **kwargs):
        calls.append((path, split, kwargs))
        return [{"messages": [{"role": "user", "content": "hello"}]}]

    monkeypatch.setattr(source_module, "load_dataset", _load_dataset)
    source = HFDatasetSourceConfig(
        path_or_dataset="json",
        split="train",
        load_kwargs={"data_files": {"train": "training.jsonl"}},
    )

    rows = load_and_adapt_hf_dataset(source)

    assert calls == [("json", "train", {"data_files": {"train": "training.jsonl"}})]
    assert rows[0]["messages"][0]["content"] == "hello"


def test_distributed_source_preparation_materializes_on_rank_zero(monkeypatch):
    source = HFDatasetSourceConfig(path_or_dataset="org/chat")
    calls = []
    monkeypatch.setattr(source_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(source_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(source_module.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(source_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(source_module.torch.distributed, "broadcast_object_list", lambda status, src: None)
    monkeypatch.setattr(source_module, "load_hf_dataset_source", lambda requested: calls.append(requested))

    prepare_hf_dataset_sources([source])

    assert calls == [source]


def test_distributed_source_preparation_waits_without_loading_on_nonzero_rank(monkeypatch):
    source = HFDatasetSourceConfig(path_or_dataset="org/chat")
    calls = []
    monkeypatch.setattr(source_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(source_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(source_module.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(source_module.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(source_module.torch.distributed, "broadcast_object_list", lambda status, src: None)
    monkeypatch.setattr(source_module, "load_hf_dataset_source", lambda requested: calls.append(requested))

    prepare_hf_dataset_sources([source])

    assert calls == []


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
