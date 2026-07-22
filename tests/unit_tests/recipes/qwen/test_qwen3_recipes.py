# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.recipes.qwen.h100 import qwen3


pytestmark = pytest.mark.unit


class _FakeModelConfig:
    pass


class _FakeBridge:
    def to_megatron_provider(self, load_weights=False):
        return _FakeModelConfig()

    @staticmethod
    def from_hf_pretrained(hf_path, **kwargs):
        return _FakeBridge()


def test_yarn_128k_recipe_uses_disjoint_train_and_validation_slices(monkeypatch):
    monkeypatch.setattr(qwen3, "AutoBridge", _FakeBridge)

    config = qwen3.qwen3_600m_sft_8gpu_h100_bf16_yarn_128k_config()

    assert isinstance(config.dataset, GPTSFTDatasetConfig)
    assert config.dataset.hf_dataset.split == "train[1%:]"
    assert config.dataset.hf_validation_dataset.split == "train[:1%]"
    assert config.dataset.hf_dataset.path_or_dataset == config.dataset.hf_validation_dataset.path_or_dataset
    assert config.dataset.hf_dataset.subset == config.dataset.hf_validation_dataset.subset == "math"
    assert config.dataset.hf_dataset.load_kwargs == config.dataset.hf_validation_dataset.load_kwargs
