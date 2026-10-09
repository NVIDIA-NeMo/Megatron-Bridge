# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.builders import DPODatasetConfig, GPTSFTDatasetConfig
from megatron.bridge.recipes import common
from megatron.bridge.recipes.qwen.h100 import qwen3, qwen3_moe


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


def test_qwen3_30b_a3b_dpo_config_applies_the_dpo_contract(monkeypatch):
    monkeypatch.setattr(common, "AutoBridge", _FakeBridge)

    cfg = qwen3_moe.qwen3_30b_a3b_dpo_8gpu_h100_bf16_config()

    assert isinstance(cfg.dataset, DPODatasetConfig)
    assert cfg.dataset.tokenizer_name == cfg.tokenizer.tokenizer_model
    assert cfg.dataset.seq_length == cfg.model.seq_length
    assert cfg.dataset.ref_artifact is None  # every run must point at its scored artifact
    assert cfg.dpo is not None

    # What validate_dpo_run_config enforces at startup.
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.model.context_parallel_size == 1
    assert cfg.train.global_batch_size % 2 == 0  # row-denominated: one pair == two rows
    assert cfg.train.micro_batch_size % 2 == 0
