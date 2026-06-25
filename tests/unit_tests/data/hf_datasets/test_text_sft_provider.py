# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json

import pytest

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_datasets.text_sft_provider import HFTextSFTDatasetProvider
from megatron.bridge.training.config import DatasetBuildContext


class _FakeBuilder:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def build(self):
        return "train", "validation", "test"


def test_hf_text_sft_provider_writes_jsonl_and_delegates_to_finetuning_builder(monkeypatch, tmp_path):
    from megatron.bridge.data.hf_datasets import text_sft_provider as provider_mod

    def _fake_get_maker(name):
        assert name == "squad"

        def _maker(**kwargs):
            return [
                {
                    "messages": [
                        {"role": "user", "content": kwargs["split"]},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            ]

        return _maker

    packed_specs = PackedSequenceSpecs(packed_sequence_size=128, pad_seq_to_mult=8)
    monkeypatch.setattr(provider_mod, "get_hf_dataset_maker", _fake_get_maker)
    monkeypatch.setattr(provider_mod, "FinetuningDatasetBuilder", _FakeBuilder)

    provider = HFTextSFTDatasetProvider(
        seq_length=128,
        dataset_root=tmp_path,
        maker_name="squad",
        maker_kwargs={"path_or_dataset": "mock/squad", "split": "train[:90%]"},
        val_maker_kwargs={"split": "train[90%:]"},
        do_test=False,
        enable_offline_packing=True,
        offline_packing_specs=packed_specs,
        dataset_kwargs={"pad_to_max_length": True},
    )

    train_ds, valid_ds, test_ds = provider.build_datasets(DatasetBuildContext(1, 1, 0, tokenizer=object()))

    assert (train_ds, valid_ds, test_ds) == ("train", "validation", "test")
    assert (tmp_path / "training.jsonl").exists()
    assert (tmp_path / "validation.jsonl").exists()
    assert not (tmp_path / "test.jsonl").exists()

    training_row = json.loads((tmp_path / "training.jsonl").read_text().splitlines()[0])
    validation_row = json.loads((tmp_path / "validation.jsonl").read_text().splitlines()[0])
    assert training_row["messages"][0]["content"] == "train[:90%]"
    assert validation_row["messages"][0]["content"] == "train[90%:]"

    builder_kwargs = _FakeBuilder.init_kwargs
    assert builder_kwargs["dataset_root"] == tmp_path
    assert builder_kwargs["enable_offline_packing"] is True
    assert builder_kwargs["offline_packing_specs"] is packed_specs
    assert builder_kwargs["dataset_kwargs"] == {
        "chat": True,
        "use_hf_tokenizer_chat_template": True,
        "pad_to_max_length": True,
    }
    assert builder_kwargs["do_validation"] is True
    assert builder_kwargs["do_test"] is False


def test_hf_text_sft_provider_can_split_validation_from_training(monkeypatch, tmp_path):
    from megatron.bridge.data.hf_datasets import text_sft_provider as provider_mod

    def _fake_get_maker(name):
        assert name == "squad"

        def _maker(**kwargs):
            assert kwargs["split"] == "train"
            return [
                {
                    "messages": [
                        {"role": "user", "content": f"question-{idx}"},
                        {"role": "assistant", "content": f"answer-{idx}"},
                    ]
                }
                for idx in range(10)
            ]

        return _maker

    monkeypatch.setattr(provider_mod, "get_hf_dataset_maker", _fake_get_maker)
    monkeypatch.setattr(provider_mod, "FinetuningDatasetBuilder", _FakeBuilder)

    provider = HFTextSFTDatasetProvider(
        seq_length=128,
        dataset_root=tmp_path,
        maker_name="squad",
        maker_kwargs={"path_or_dataset": "mock/squad", "split": "train"},
        val_proportion=0.2,
        do_test=False,
    )

    provider.build_datasets(DatasetBuildContext(1, 1, 0, tokenizer=object()))

    train_rows = (tmp_path / "training.jsonl").read_text().splitlines()
    validation_rows = (tmp_path / "validation.jsonl").read_text().splitlines()
    assert len(train_rows) == 8
    assert len(validation_rows) == 2


def test_hf_text_sft_provider_requires_context_tokenizer(tmp_path):
    provider = HFTextSFTDatasetProvider(seq_length=128, dataset_root=tmp_path, maker_name="squad")

    with pytest.raises(ValueError, match="requires a tokenizer"):
        provider.build_datasets(DatasetBuildContext(1, 0, 0))
