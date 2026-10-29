# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MockMegatronMIMOProvider."""

import pytest
import torch

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.megatron_mimo.mock_provider import MockMegatronMIMOProvider


class DummyProcessor:
    def __call__(self, inputs, return_tensors="pt"):
        del inputs, return_tensors
        return {"pixel_values": torch.ones(1, 3, 2, 2)}


class DummyTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"

    def __call__(self, text, truncation=True, max_length=128, return_tensors="pt"):
        del text, truncation, max_length, return_tensors
        return {"input_ids": torch.tensor([[11, 12, 13]])}


def _make_provider(modality_presence_prob=None) -> MockMegatronMIMOProvider:
    provider = MockMegatronMIMOProvider(
        seq_length=16,
        processor_paths={},
        tokenizer_path="",
        special_token_ids={"vision": 32000},
        encoder_seq_lengths={"vision": 2},
        modality_configs={"vision": {"type": "image", "width": 2, "height": 2}},
        modality_presence_prob=modality_presence_prob or {},
        random_seed=123,
    )
    object.__setattr__(provider, "_processors", {"vision": DummyProcessor()})
    object.__setattr__(provider, "_tokenizer", DummyTokenizer())
    return provider


def _build_train_dataset(provider: MockMegatronMIMOProvider, samples: int = 4):
    train_ds, valid_ds, test_ds = provider.build_datasets(
        DatasetBuildContext(train_samples=samples, valid_samples=0, test_samples=0)
    )
    assert train_ds is not None
    assert valid_ds is None
    assert test_ds is None
    return train_ds


def test_default_modality_presence_keeps_all_modalities():
    dataset = _build_train_dataset(_make_provider())

    assert all("vision" in example for example in dataset.examples)
    item = dataset[0]
    assert "vision" in item["modality_inputs"]
    assert torch.count_nonzero(item["input_ids"] == 32000).item() == 2


def test_zero_modality_presence_creates_text_only_samples():
    dataset = _build_train_dataset(_make_provider({"vision": 0.0}))

    assert all("vision" not in example for example in dataset.examples)
    item = dataset[0]
    assert item["modality_inputs"] == {}
    assert torch.count_nonzero(item["input_ids"] == 32000).item() == 0


def test_invalid_modality_presence_probability_raises():
    provider = _make_provider({"vision": 1.1})

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        _build_train_dataset(provider)


def test_unknown_modality_presence_probability_raises():
    provider = _make_provider({"audio": 0.5})

    with pytest.raises(ValueError, match="unknown modality"):
        _build_train_dataset(provider)
