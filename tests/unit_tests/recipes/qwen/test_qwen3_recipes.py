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
    assert cfg.dataset.tokenizer_name == cfg.tokenizer.tokenizer_model == qwen3_moe._QWEN3_30B_A3B_MODEL_ID
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": qwen3_moe._QWEN3_30B_A3B_MODEL_REVISION}
    assert cfg.dataset.seq_length == cfg.model.seq_length == 4096
    assert cfg.dataset.ref_artifact is None  # every run must point at its scored artifact
    assert cfg.dpo is not None

    # What validate_dpo_run_config enforces at startup.
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.model.cross_entropy_loss_fusion is False
    assert cfg.ddp.average_in_collective is False
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.mtp_num_layers is None

    # One node fills both meshes (TP4 x PP2 = PP2 x EP4 x ETP1 = 8), matching a
    # --tp 4 --sequence-parallel --ep 4 --etp 1 scoring run; PP is not part of the artifact.
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.sequence_parallel is True
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.recompute_granularity is None

    # Row-denominated batch sizes must be even (one pair == two rows).
    assert cfg.train.global_batch_size % 2 == 0
    assert cfg.train.micro_batch_size % 2 == 0
    assert cfg.validation.eval_iters == 0
