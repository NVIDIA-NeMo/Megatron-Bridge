# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for HFMimoDatasetProvider."""

from dataclasses import dataclass

import torch

from megatron.bridge.data.mimo.hf_provider import HFMimoDatasetProvider
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


def _make_provider() -> HFMimoDatasetProvider:
    return HFMimoDatasetProvider(
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
        """
        Test helper that records an invocation and always reports the repository as not safe.
        
        Parameters:
            trust_remote_code: Ignored.
            hf_path: Ignored.
        
        Returns:
            False: Indicates the repository is not considered safe. Also increments `calls.is_safe_repo` as a side effect.
        """
        del trust_remote_code, hf_path
        calls.is_safe_repo += 1
        return False

    def fake_load_dataset(path, name=None, split=None, trust_remote_code=None, data_files=None):
        """
        Test stub that simulates loading a dataset for unit tests.
        
        Parameters:
            path (str): Ignored.
            name (str | None): Ignored.
            split (str | None): If equal to "validation", a `ValueError` is raised to simulate a missing split.
            trust_remote_code: Ignored.
            data_files: Ignored.
        
        Side effects:
            Increments `calls.load_dataset` to record an invocation.
        
        Returns:
            list[dict]: A single-record dataset: [{"text": "hello", "image": "image_0.jpg"}].
        
        Raises:
            ValueError: If `split` is "validation".
        """
        del path, name, trust_remote_code, data_files
        calls.load_dataset += 1
        if split == "validation":
            raise ValueError("missing split")
        return [{"text": "hello", "image": "image_0.jpg"}]

    class _AutoProcessor:
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

    monkeypatch.setattr("megatron.bridge.data.mimo.hf_provider.is_safe_repo", fake_is_safe_repo)
    monkeypatch.setattr("megatron.bridge.data.mimo.hf_provider.load_dataset", fake_load_dataset)
    monkeypatch.setattr("megatron.bridge.data.mimo.hf_provider.AutoProcessor", _AutoProcessor)
    monkeypatch.setattr("megatron.bridge.data.mimo.hf_provider.AutoTokenizer", _AutoTokenizer)

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
