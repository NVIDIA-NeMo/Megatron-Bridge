# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
from types import SimpleNamespace

import pytest
from megatron.training.config.instantiate_utils import instantiate

from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    FinetuningDatasetConfig,
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)
from megatron.bridge.data.builders import gpt_sft as builder_mod
from megatron.bridge.data.builders.gpt_sft import (
    FinetuningDatasetBuilder,
    GPTSFTDatasetBuilder,
    build_gpt_sft_split,
    materialize_hf_dataset,
    normalize_gpt_sft_dataset_kwargs,
    resolve_gpt_sft_dataset_root,
)
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.data.sources.hf import resolve_hf_dataset_source
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit


def _hf_config(tmp_path, **source_overrides):
    source_kwargs = {
        "path_or_dataset": "mock/squad",
        "schema_adapter": "squad",
        **source_overrides,
    }
    return GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(**source_kwargs),
        hf_output_root=tmp_path,
        preprocessing=ChatSFTPreprocessingConfig(),
        seed=5678,
        do_test=False,
    )


def test_config_round_trip_is_declarative_and_serializable(tmp_path):
    specs = PackedSequenceSpecs(packed_sequence_size=128, pad_seq_to_mult=8)
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        preprocessing=PromptCompletionSFTPreprocessingConfig(),
        hf_output_root=str(tmp_path),
        hf_validation_proportion=0.1,
        seed=5678,
        enable_offline_packing=True,
        offline_packing_specs=specs,
        do_test=False,
    )

    serialized = ConfigContainer._convert_value_to_dict(config)
    restored = instantiate(serialized)

    assert isinstance(restored, GPTSFTDatasetConfig)
    assert isinstance(restored.hf_dataset, HFDatasetSourceConfig)
    assert restored.hf_dataset.dataset_name == "squad"
    assert restored.offline_packing_specs.packed_sequence_size == 128
    assert isinstance(restored.preprocessing, PromptCompletionSFTPreprocessingConfig)
    assert "tokenizer" not in serialized


@pytest.mark.parametrize(
    ("dataset_root", "hf_dataset"),
    [
        (None, None),
        ("/tmp/local", HFDatasetSourceConfig(path_or_dataset="mock/squad", schema_adapter="squad")),
    ],
)
def test_config_requires_exactly_one_source(dataset_root, hf_dataset):
    config = GPTSFTDatasetConfig(seq_length=128, dataset_root=dataset_root, hf_dataset=hf_dataset)

    with pytest.raises(ValueError, match="Exactly one text-only SFT source"):
        config.validate()


def test_config_rejects_max_num_samples_in_dataset_kwargs(tmp_path):
    config = GPTSFTDatasetConfig(
        seq_length=128,
        dataset_root=tmp_path,
        max_train_samples=10,
        dataset_kwargs={"max_num_samples": 20},
    )

    with pytest.raises(ValueError, match="Set max_train_samples directly"):
        config.validate()


def test_config_rejects_non_mapping_dataset_kwargs(tmp_path):
    config = GPTSFTDatasetConfig(seq_length=128, dataset_root=tmp_path, dataset_kwargs=["invalid"])

    with pytest.raises(TypeError, match="dataset_kwargs must be a mapping"):
        config.validate()


def test_config_rejects_implicit_unsupported_preset_split():
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="raven"),
        do_validation=True,
        do_test=False,
    )

    with pytest.raises(ValueError, match="has no validation split"):
        config.validate()


def test_hf_rewrite_rejects_explicit_packed_paths(tmp_path):
    specs = PackedSequenceSpecs(packed_sequence_size=128, packed_metadata_path=tmp_path / "metadata.jsonl")
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        hf_output_root=tmp_path,
        hf_rewrite=True,
        enable_offline_packing=True,
        offline_packing_specs=specs,
        do_test=False,
    )

    with pytest.raises(ValueError, match="cannot safely replace explicit packed data paths"):
        config.validate()


@pytest.mark.parametrize(
    ("filename", "pad_cu_seqlens", "expects_metadata"),
    [
        ("training.idx.parquet", False, False),
        ("training.idx.parquet", True, True),
        ("training.npy", False, True),
    ],
)
def test_packed_metadata_forwarding_depends_on_format_and_padding(
    monkeypatch, tmp_path, filename, pad_cu_seqlens, expects_metadata
):
    """Test Parquet and legacy packed formats receive metadata when required."""
    packed_path = tmp_path / filename
    packed_path.touch()
    metadata_path = tmp_path / "metadata.jsonl"
    captured = {}

    def _build(**kwargs):
        captured.update(kwargs)
        return kwargs["file_path"]

    if filename.endswith(".npy"):
        from megatron.bridge.data.packing import gpt_sft as packed_module
    else:
        from megatron.bridge.data.packing import parquet as packed_module

    dataset_class_name = "GPTSFTPackedDataset" if filename.endswith(".npy") else "GPTSFTPackedParquetDataset"
    monkeypatch.setattr(packed_module, dataset_class_name, _build)

    build_gpt_sft_split(
        packed_path,
        tokenizer=object(),
        seq_length=128,
        memmap_workers=1,
        seed=1234,
        packed_sequence_size=128,
        pack_metadata_path=metadata_path,
        pad_cu_seqlens=pad_cu_seqlens,
        pad_seq_to_mult=4,
        dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
    )

    expected = metadata_path if expects_metadata else None
    assert captured["pack_metadata_file_path"] == expected
    assert captured["pad_seq_to_mult"] == 4


def test_build_gpt_sft_split_routes_chat_options(monkeypatch, tmp_path):
    dataset_path = tmp_path / "training.jsonl"
    dataset_path.touch()
    captured = {}

    def _build_chat(**kwargs):
        captured.update(kwargs)
        return kwargs["file_path"]

    monkeypatch.setattr(builder_mod, "GPTSFTChatDataset", _build_chat)

    result = build_gpt_sft_split(
        dataset_path,
        tokenizer=object(),
        seq_length=128,
        memmap_workers=1,
        seed=1234,
        packed_sequence_size=-1,
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "tool_schemas": {"type": "function"},
        },
    )

    assert result == str(dataset_path)
    assert captured["use_hf_tokenizer_chat_template"] is True
    assert captured["tool_schemas"] == {"type": "function"}


def test_local_source_rejects_hf_only_settings(tmp_path):
    config = GPTSFTDatasetConfig(
        seq_length=128,
        dataset_root=tmp_path,
        hf_output_root=tmp_path / "materialized",
    )

    with pytest.raises(ValueError, match="require hf_dataset"):
        config.validate()


def test_hf_source_rejects_competing_validation_modes(tmp_path):
    config = _hf_config(tmp_path)
    config.hf_validation_dataset = HFDatasetSourceConfig(path_or_dataset="mock/validation")
    config.hf_validation_proportion = 0.1

    with pytest.raises(ValueError, match="either hf_validation_dataset or hf_validation_proportion"):
        config.validate()


def test_local_jsonl_source_uses_explicit_prompt_completion_preprocessing(monkeypatch, tmp_path):
    config = GPTSFTDatasetConfig(
        seq_length=256,
        dataset_root=tmp_path,
        preprocessing=PromptCompletionSFTPreprocessingConfig(),
    )
    monkeypatch.setattr(
        builder_mod,
        "load_and_adapt_hf_dataset",
        lambda _source: pytest.fail("local JSONL mode must not load a Hugging Face source"),
    )

    assert resolve_gpt_sft_dataset_root(config) == tmp_path
    dataset_kwargs = normalize_gpt_sft_dataset_kwargs(config)
    assert dataset_kwargs["chat"] is False
    assert isinstance(dataset_kwargs["prompt_completion_config"], PromptCompletionSFTPreprocessingConfig)


def test_local_jsonl_source_preserves_implicit_legacy_preprocessing(tmp_path):
    config = GPTSFTDatasetConfig(seq_length=256, dataset_root=tmp_path)

    assert normalize_gpt_sft_dataset_kwargs(config) == {}


def test_config_rejects_combining_explicit_and_legacy_preprocessing(tmp_path):
    config = GPTSFTDatasetConfig(
        seq_length=256,
        dataset_root=tmp_path,
        preprocessing=ChatSFTPreprocessingConfig(),
        dataset_kwargs={"chat": True},
    )

    with pytest.raises(ValueError, match="Do not combine explicit preprocessing"):
        config.validate()


def test_config_warns_and_preserves_legacy_preprocessing_kwargs(tmp_path):
    config = GPTSFTDatasetConfig(
        seq_length=256,
        dataset_root=tmp_path,
        dataset_kwargs={"prompt_template": "{input} {output}", "label_key": "output"},
    )

    with pytest.warns(DeprecationWarning, match="preprocessing through dataset_kwargs"):
        dataset_kwargs = normalize_gpt_sft_dataset_kwargs(config)

    assert dataset_kwargs == config.dataset_kwargs


def test_hf_source_gets_chat_normalization_defaults(tmp_path):
    config = _hf_config(tmp_path)
    config.dataset_kwargs = {"pad_to_max_length": True}

    assert normalize_gpt_sft_dataset_kwargs(config) == {
        "chat": True,
        "use_hf_tokenizer_chat_template": True,
        "chat_loss_mode": "assistant",
        "pad_to_max_length": True,
    }


def test_hf_source_preserves_implicit_chat_compatibility(tmp_path):
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        do_validation=False,
        do_test=False,
    )

    assert normalize_gpt_sft_dataset_kwargs(config)["chat"] is True


def test_default_hf_materialization_root_fingerprints_full_source(monkeypatch):
    monkeypatch.setattr(builder_mod, "get_dataset_root", lambda name: f"cache/{name}")
    base = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(
            path_or_dataset="json",
            load_kwargs={"data_files": {"train": "math.jsonl"}},
            adapter_kwargs={"messages_column": "messages"},
        ),
        do_validation=False,
        do_test=False,
    )
    different_file = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(
            path_or_dataset="json",
            load_kwargs={"data_files": {"train": "science.jsonl"}},
            adapter_kwargs={"messages_column": "messages"},
        ),
        do_validation=False,
        do_test=False,
    )
    different_adapter_kwargs = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(
            path_or_dataset="json",
            load_kwargs={"data_files": {"train": "math.jsonl"}},
            adapter_kwargs={"messages_column": "conversation"},
        ),
        do_validation=False,
        do_test=False,
    )

    roots = {
        resolve_gpt_sft_dataset_root(base),
        resolve_gpt_sft_dataset_root(different_file),
        resolve_gpt_sft_dataset_root(different_adapter_kwargs),
    }

    assert len(roots) == 3
    assert all(root.startswith("cache/hf-sft-native-") for root in roots)


def test_named_and_equivalent_custom_sources_share_materialization_identity(monkeypatch):
    monkeypatch.setattr(builder_mod, "get_dataset_root", lambda name: f"cache/{name}")
    named = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        do_validation=False,
        do_test=False,
    )
    custom = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(path_or_dataset="rajpurkar/squad", schema_adapter="squad"),
        do_validation=False,
        do_test=False,
    )

    assert resolve_gpt_sft_dataset_root(named) == resolve_gpt_sft_dataset_root(custom)


def test_materialization_identity_includes_preprocessing(monkeypatch):
    monkeypatch.setattr(builder_mod, "get_dataset_root", lambda name: f"cache/{name}")
    source = HFDatasetSourceConfig(path_or_dataset="json")
    chat = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=source,
        preprocessing=ChatSFTPreprocessingConfig(),
        do_validation=False,
        do_test=False,
    )
    paired = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=source,
        preprocessing=PromptCompletionSFTPreprocessingConfig(),
        do_validation=False,
        do_test=False,
    )

    assert resolve_gpt_sft_dataset_root(chat) != resolve_gpt_sft_dataset_root(paired)


def test_hf_source_materializes_requested_jsonl_splits(monkeypatch, tmp_path):
    calls = []

    def _fake_load(source):
        source = resolve_hf_dataset_source(source)
        calls.append(source)
        return [
            {
                "messages": [
                    {"role": "user", "content": source.split},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]

    monkeypatch.setattr(builder_mod, "load_and_adapt_hf_dataset", _fake_load)
    config = _hf_config(tmp_path)
    config.hf_validation_dataset = HFDatasetSourceConfig(
        path_or_dataset="mock/squad",
        split="train[90%:]",
        schema_adapter="squad",
    )

    materialize_hf_dataset(config, tmp_path)

    assert [call.split for call in calls] == ["train", "train[90%:]"]
    training_row = json.loads((tmp_path / "training.jsonl").read_text().splitlines()[0])
    validation_row = json.loads((tmp_path / "validation.jsonl").read_text().splitlines()[0])
    assert training_row["conversation"][0]["content"] == "train"
    assert validation_row["conversation"][0]["content"] == "train[90%:]"
    assert not (tmp_path / "test.jsonl").exists()


def test_hf_source_can_split_validation_from_training(monkeypatch, tmp_path):
    def _fake_load(source):
        source = resolve_hf_dataset_source(source)
        assert source.split == "train"
        return [
            {
                "messages": [
                    {"role": "user", "content": f"question-{idx}"},
                    {"role": "assistant", "content": f"answer-{idx}"},
                ]
            }
            for idx in range(10)
        ]

    monkeypatch.setattr(builder_mod, "load_and_adapt_hf_dataset", _fake_load)
    config = _hf_config(tmp_path)
    config.hf_validation_proportion = 0.2

    materialize_hf_dataset(config, tmp_path)

    assert len((tmp_path / "training.jsonl").read_text().splitlines()) == 8
    assert len((tmp_path / "validation.jsonl").read_text().splitlines()) == 2


def test_hf_source_reads_local_json(tmp_path):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        + "\n"
    )
    output_root = tmp_path / "output"
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(
            path_or_dataset="json",
            load_kwargs={"data_files": {"train": str(source_path)}},
        ),
        hf_output_root=output_root,
        preprocessing=ChatSFTPreprocessingConfig(),
        do_validation=False,
        do_test=False,
    )

    materialize_hf_dataset(config, output_root)

    output_row = json.loads((output_root / "training.jsonl").read_text())
    assert output_row["conversation"] == json.loads(source_path.read_text())["messages"]


def test_hf_rewrite_removes_disabled_split_jsonl(monkeypatch, tmp_path):
    (tmp_path / "validation.jsonl").write_text("stale validation")
    (tmp_path / "validation.jsonl.idx.npy").touch()
    (tmp_path / "validation.jsonl.idx.info").touch()
    (tmp_path / "test.jsonl").write_text("stale test")
    (tmp_path / "test.jsonl.idx.npy").touch()
    (tmp_path / "test.jsonl.idx.info").touch()
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        hf_output_root=tmp_path,
        hf_rewrite=True,
        do_validation=False,
        do_test=False,
    )
    monkeypatch.setattr(builder_mod, "_materialize_hf_split", lambda *_args, **_kwargs: None)

    materialize_hf_dataset(config, tmp_path)

    assert not (tmp_path / "validation.jsonl").exists()
    assert not (tmp_path / "validation.jsonl.idx.npy").exists()
    assert not (tmp_path / "validation.jsonl.idx.info").exists()
    assert not (tmp_path / "test.jsonl").exists()
    assert not (tmp_path / "test.jsonl.idx.npy").exists()
    assert not (tmp_path / "test.jsonl.idx.info").exists()


def test_hf_rewrite_invalidates_memmap_index_sidecars(monkeypatch, tmp_path):
    output_path = tmp_path / "training.jsonl"
    output_path.write_text('{"prompt": "old", "completion": "row"}\n')
    index_paths = (tmp_path / "training.jsonl.idx.npy", tmp_path / "training.jsonl.idx.info")
    for index_path in index_paths:
        index_path.touch()
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        preprocessing=PromptCompletionSFTPreprocessingConfig(separator=" "),
        hf_output_root=tmp_path,
        hf_rewrite=True,
        do_validation=False,
        do_test=False,
    )
    monkeypatch.setattr(
        builder_mod,
        "_load_hf_examples",
        lambda *_args: [{"prompt": "new", "completion": "row"}],
    )

    materialize_hf_dataset(config, tmp_path)

    assert json.loads(output_path.read_text()) == {"prompt": "new", "completion": "row"}
    assert not any(index_path.exists() for index_path in index_paths)


def test_builder_owns_runtime_materialization_and_shared_construction(monkeypatch, tmp_path):
    config = _hf_config(tmp_path)
    materialize_mock = []
    dataset_calls = []

    monkeypatch.setattr(
        builder_mod,
        "materialize_hf_dataset",
        lambda received_config, root: materialize_mock.append((received_config, root)),
    )

    def _build(path, **kwargs):
        dataset_calls.append((path, kwargs))
        return str(path)

    monkeypatch.setattr(builder_mod, "build_gpt_sft_split", _build)

    builder = GPTSFTDatasetBuilder(config=config, tokenizer=object())
    train, validation, test = builder.build()

    assert materialize_mock == [(config, tmp_path)]
    assert train.endswith("training.jsonl")
    assert validation.endswith("validation.jsonl")
    assert test is None
    assert len(dataset_calls) == 2
    assert dataset_calls[0][1]["dataset_kwargs"]["chat"] is True
    assert dataset_calls[0][1]["dataset_kwargs"]["chat_loss_mode"] == "assistant"


def test_hf_rewrite_regenerates_existing_builder_managed_packed_data(monkeypatch, tmp_path):
    specs = PackedSequenceSpecs(packed_sequence_size=128, tokenizer_model_name="test-tokenizer")
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        hf_output_root=tmp_path,
        hf_validation_proportion=0.1,
        hf_rewrite=True,
        enable_offline_packing=True,
        offline_packing_specs=specs,
        do_test=False,
    )
    builder = GPTSFTDatasetBuilder(config=config, tokenizer=SimpleNamespace(_tokenizer=object()))
    builder.train_path_packed.touch()
    builder.validation_path_packed.touch()
    builder.pack_metadata.write_text('[{"stale": true}]')
    pack_calls = []

    monkeypatch.setattr(builder_mod, "materialize_hf_dataset", lambda *_: None)
    monkeypatch.setattr(
        "megatron.bridge.data.packing.offline.prepare_gpt_sft_packed_data",
        lambda **kwargs: pack_calls.append(kwargs),
    )

    builder.prepare_data()

    assert [call["input_path"].name for call in pack_calls] == ["training.jsonl", "validation.jsonl"]
    assert json.loads(builder.pack_metadata.read_text()) == []


def test_hf_rewrite_removes_disabled_validation_pack(monkeypatch, tmp_path):
    specs = PackedSequenceSpecs(packed_sequence_size=128, tokenizer_model_name="test-tokenizer")
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        hf_output_root=tmp_path,
        hf_rewrite=True,
        enable_offline_packing=True,
        offline_packing_specs=specs,
        do_validation=False,
        do_test=False,
    )
    builder = GPTSFTDatasetBuilder(config=config, tokenizer=SimpleNamespace(_tokenizer=object()))
    builder.train_path_packed.touch()
    builder.validation_path_packed.touch()
    builder.pack_metadata.write_text('[{"stale": true}]')
    pack_calls = []

    monkeypatch.setattr(builder_mod, "materialize_hf_dataset", lambda *_: None)
    monkeypatch.setattr(
        "megatron.bridge.data.packing.offline.prepare_gpt_sft_packed_data",
        lambda **kwargs: pack_calls.append(kwargs),
    )

    builder.prepare_data()

    assert len(pack_calls) == 1
    assert not builder.validation_path_packed.exists()
    assert json.loads(builder.pack_metadata.read_text()) == []


def test_builder_requires_tokenizer(tmp_path):
    with pytest.raises(ValueError, match="requires an initialized tokenizer"):
        GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(seq_length=128, dataset_root=tmp_path),
            tokenizer=None,
        )


def test_deprecated_config_and_builder_delegate_to_canonical_types(tmp_path):
    with pytest.warns(DeprecationWarning, match="GPTSFTDatasetConfig"):
        config = FinetuningDatasetConfig(seq_length=128, dataset_root=tmp_path)
    assert isinstance(config, GPTSFTDatasetConfig)

    with pytest.warns(DeprecationWarning, match="GPTSFTDatasetBuilder"):
        builder = FinetuningDatasetBuilder(dataset_root=tmp_path, tokenizer=object(), seq_length=128)
    assert isinstance(builder, GPTSFTDatasetBuilder)
    assert isinstance(builder.config, GPTSFTDatasetConfig)


def test_deprecated_builder_preserves_legacy_prompt_kwargs(tmp_path):
    legacy_kwargs = {
        "prompt_template": "Context: {context} Question: {question} Answer: {answer}",
        "label_key": "answer",
        "truncation_field": "context,question",
        "add_sep": True,
    }

    with pytest.warns(DeprecationWarning):
        builder = FinetuningDatasetBuilder(
            dataset_root=tmp_path,
            tokenizer=object(),
            seq_length=128,
            dataset_kwargs=legacy_kwargs,
        )

    assert builder.dataset_kwargs == legacy_kwargs


def test_deprecated_builder_fingerprints_legacy_prompt_semantics(tmp_path):
    class _PackingTokenizer:
        _tokenizer = object()

    tokenizer = _PackingTokenizer()
    with pytest.warns(DeprecationWarning):
        first = FinetuningDatasetBuilder(
            dataset_root=tmp_path,
            tokenizer=tokenizer,
            seq_length=128,
            dataset_kwargs={"prompt_template": "{input} {output}"},
        )
    with pytest.warns(DeprecationWarning):
        second = FinetuningDatasetBuilder(
            dataset_root=tmp_path,
            tokenizer=tokenizer,
            seq_length=128,
            dataset_kwargs={"prompt_template": "Question: {input} Answer: {output}"},
        )

    assert first.default_pack_path != second.default_pack_path


def test_deprecated_config_rejects_runtime_objects_in_legacy_kwargs(tmp_path):
    with pytest.warns(DeprecationWarning):
        config = FinetuningDatasetConfig(
            seq_length=128,
            dataset_root=tmp_path,
            dataset_kwargs={"prompt_template": lambda value: value},
        )

    with pytest.raises(TypeError, match="declarative values"):
        config.validate()
