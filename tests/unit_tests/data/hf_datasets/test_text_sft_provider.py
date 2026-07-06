# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.hf_datasets import text_sft_provider as provider_mod
from megatron.bridge.data.hf_datasets.text_sft_provider import HFTextSFTDatasetProvider
from megatron.bridge.training.config import DatasetBuildContext, GPTSFTDatasetConfig


pytestmark = pytest.mark.unit


class _FakeBuilder:
    config = None

    def __init__(self, *, config, tokenizer):
        type(self).config = config
        assert tokenizer is not None

    def build(self):
        return "train", "validation", "test"


def test_deprecated_provider_delegates_to_config_and_builder(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_mod, "_get_gpt_sft_dataset_builder", lambda: _FakeBuilder)

    with pytest.warns(DeprecationWarning, match="GPTSFTDatasetConfig"):
        provider = HFTextSFTDatasetProvider(
            seq_length=128,
            maker_name="squad",
            maker_kwargs={"path_or_dataset": "mock/squad"},
            dataset_root=tmp_path,
            do_test=False,
        )

    assert provider.build_datasets(DatasetBuildContext(1, 1, 0, tokenizer=object())) == (
        "train",
        "validation",
        "test",
    )
    assert isinstance(_FakeBuilder.config, GPTSFTDatasetConfig)
    assert _FakeBuilder.config.hf_dataset.maker_name == "squad"
    assert _FakeBuilder.config.hf_dataset.output_root == tmp_path


def test_deprecated_provider_requires_tokenizer():
    with pytest.warns(DeprecationWarning):
        provider = HFTextSFTDatasetProvider(seq_length=128, maker_name="squad")

    with pytest.raises(ValueError, match="requires a tokenizer"):
        provider.build_datasets(DatasetBuildContext(1, 0, 0))
