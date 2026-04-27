# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for HFMegatronMIMODatasetProvider."""

from dataclasses import dataclass

import torch

from megatron.bridge.data.megatron_mimo.hf_provider import HFMegatronMIMODatasetProvider
from megatron.bridge.training.config import DatasetBuildContext


class DummyProcessor:
    def __call__(self, inputs, return_tensors="pt"):
        del inputs, return_tensors
        return {"pixel_values": torch.randn(1, 3, 224, 224)}


class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=128, return_tensors="pt"):
        del text, truncation, max_length, return_tensors
        return {"input_ids": torch.tensor([[1, 2, 3]])}


@dataclass
class Calls:
    load_dataset: int = 0
    auto_processor: int = 0
    auto_tokenizer: int = 0
    is_safe_repo: int = 0


def _make_provider() -> HFMegatronMIMODatasetProvider:
    return HFMegatronMIMODatasetProvider(
        seq_length=32,
        hf_dataset_path="org/dataset",
        hf_tokenizer_path="org/tokenizer",
        processor_paths={"vision": "org/processor"},
        special_token_ids={"vision": 32000},
        encoder_seq_lengths={"vision": 1},
        modality_columns={"vision": "image"},
    )


def test_build_datasets_happy_path(monkeypatch):
    calls = Calls()

    def fake_is_safe_repo(trust_remote_code, hf_path):
        del trust_remote_code, hf_path
        calls.is_safe_repo += 1
        return False

    def fake_load_dataset(path, name=None, split=None, trust_remote_code=None, data_files=None):
        del path, name, trust_remote_code, data_files
        calls.load_dataset += 1
        if split == "validation":
            raise ValueError("missing split")
        return [{"text": "hello", "image": "image_0.jpg"}]

    class _AutoImageProcessor:
        @staticmethod
        def from_pretrained(path, trust_remote_code=None):
            del path, trust_remote_code
            calls.auto_processor += 1
            return DummyProcessor()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path, trust_remote_code=None):
            del path, trust_remote_code
            calls.auto_tokenizer += 1
            return DummyTokenizer()

    monkeypatch.setattr("megatron.bridge.data.megatron_mimo.hf_provider.is_safe_repo", fake_is_safe_repo)
    monkeypatch.setattr("megatron.bridge.data.megatron_mimo.hf_provider.load_dataset", fake_load_dataset)
    monkeypatch.setattr("megatron.bridge.data.megatron_mimo.hf_provider.AutoImageProcessor", _AutoImageProcessor)
    monkeypatch.setattr("megatron.bridge.data.megatron_mimo.hf_provider.AutoTokenizer", _AutoTokenizer)

    provider = _make_provider()
    context = DatasetBuildContext(train_samples=4, valid_samples=4, test_samples=2)
    train_ds, valid_ds, test_ds = provider.build_datasets(context)

    assert train_ds is not None
    assert valid_ds is None  # missing split propagates as None
    assert test_ds is not None
    assert len(train_ds) == 1
    assert len(test_ds) == 1
    assert calls.auto_processor == 1
    assert calls.auto_tokenizer == 1
    assert calls.load_dataset == 3
    assert calls.is_safe_repo >= 3


def test_get_collate_fn_returns_partial():
    provider = _make_provider()
    collate_fn = provider.get_collate_fn()
    assert callable(collate_fn)
    assert collate_fn.keywords["modality_names"] == ["vision"]
