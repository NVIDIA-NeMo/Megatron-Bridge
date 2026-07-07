# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json

import pytest
from megatron.training.config.instantiate_utils import instantiate

from megatron.bridge.data.builders import (
    FinetuningDatasetConfig,
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
)
from megatron.bridge.data.builders import gpt_sft_dataset as builder_mod
from megatron.bridge.data.builders.gpt_sft_dataset import (
    FinetuningDatasetBuilder,
    GPTSFTDatasetBuilder,
    materialize_hf_dataset,
    normalize_gpt_sft_dataset_kwargs,
    resolve_gpt_sft_dataset_root,
)
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
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
        seed=5678,
        do_test=False,
    )


def test_config_round_trip_is_declarative_and_serializable(tmp_path):
    specs = PackedSequenceSpecs(packed_sequence_size=128, pad_seq_to_mult=8)
    config = GPTSFTDatasetConfig(
        seq_length=128,
        hf_dataset=HFDatasetSourceConfig(
            path_or_dataset="mock/squad",
            schema_adapter="squad",
        ),
        hf_output_root=str(tmp_path),
        hf_validation_proportion=0.1,
        seed=5678,
        enable_offline_packing=True,
        offline_packing_specs=specs,
    )

    serialized = ConfigContainer._convert_value_to_dict(config)
    restored = instantiate(serialized)

    assert isinstance(restored, GPTSFTDatasetConfig)
    assert isinstance(restored.hf_dataset, HFDatasetSourceConfig)
    assert restored.hf_dataset.schema_adapter == "squad"
    assert restored.offline_packing_specs.packed_sequence_size == 128
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


def test_local_jsonl_source_resolves_without_hf_defaults(monkeypatch, tmp_path):
    config = GPTSFTDatasetConfig(
        seq_length=256,
        dataset_root=tmp_path,
        dataset_kwargs={"prompt_template": "{input} {output}"},
    )
    monkeypatch.setattr(
        builder_mod,
        "load_and_adapt_hf_dataset",
        lambda _source: pytest.fail("local JSONL mode must not load a Hugging Face source"),
    )

    assert resolve_gpt_sft_dataset_root(config) == tmp_path
    assert normalize_gpt_sft_dataset_kwargs(config) == {"prompt_template": "{input} {output}"}


def test_hf_source_gets_chat_normalization_defaults(tmp_path):
    config = _hf_config(tmp_path)
    config.dataset_kwargs = {"pad_to_max_length": True}

    assert normalize_gpt_sft_dataset_kwargs(config) == {
        "chat": True,
        "use_hf_tokenizer_chat_template": True,
        "pad_to_max_length": True,
    }


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


def test_hf_source_materializes_requested_jsonl_splits(monkeypatch, tmp_path):
    calls = []

    def _fake_load(source):
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
    assert training_row["messages"][0]["content"] == "train"
    assert validation_row["messages"][0]["content"] == "train[90%:]"
    assert not (tmp_path / "test.jsonl").exists()


def test_hf_source_can_split_validation_from_training(monkeypatch, tmp_path):
    def _fake_load(source):
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
        do_validation=False,
        do_test=False,
    )

    materialize_hf_dataset(config, output_root)

    assert json.loads((output_root / "training.jsonl").read_text()) == json.loads(source_path.read_text())


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

    monkeypatch.setattr(builder_mod, "build_gpt_sft_dataset", _build)

    builder = GPTSFTDatasetBuilder(config=config, tokenizer=object())
    train, validation, test = builder.build()

    assert materialize_mock == [(config, tmp_path)]
    assert train.endswith("training.jsonl")
    assert validation.endswith("validation.jsonl")
    assert test is None
    assert len(dataset_calls) == 2
    assert dataset_calls[0][1]["dataset_kwargs"]["chat"] is True


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
