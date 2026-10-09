# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from typing import Callable

import pytest
from transformers import Qwen3Config

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.recipes.qwen.h100 import qwen3
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


pytestmark = pytest.mark.unit

_YARN_RECIPES = {"qwen3_600m_sft_8gpu_h100_bf16_yarn_128k_config"}
_ALL_QWEN3_RECIPE_FUNCS = [getattr(qwen3, name) for name in qwen3.__all__ if name.endswith("_config")]


class _FakeModelConfig:
    pass


class _FakeBridge:
    def get_model_config(self):
        return _FakeModelConfig()

    @staticmethod
    def from_hf_pretrained(hf_path, **kwargs):
        return _FakeBridge()


def _tiny_qwen3_config() -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=40960,
        vocab_size=151936,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    return config


class _BuilderOnlyBridge:
    """Return a real strict ModelConfig while rejecting the provider API."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs) -> "_BuilderOnlyBridge":
        return _BuilderOnlyBridge()

    def get_model_config(self) -> BridgeGPTModelConfig:
        model_config = AutoBridge.from_hf_config(_tiny_qwen3_config()).get_model_config()
        assert isinstance(model_config, BridgeGPTModelConfig)
        return model_config

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("Qwen3 recipes must use get_model_config(), not the legacy provider API")


@pytest.mark.parametrize("recipe_func", _ALL_QWEN3_RECIPE_FUNCS, ids=lambda func: func.__name__)
def test_each_qwen3_recipe_uses_strict_builder_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every Qwen3 recipe should configure a ModelConfig without provider fallback."""
    patch_recipe_module_global(monkeypatch, recipe_func, "AutoBridge", _BuilderOnlyBridge)

    if "peft" in recipe_func.__name__:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()

    assert isinstance(cfg.model, BridgeGPTModelConfig)
    assert cfg.model.normalization == "RMSNorm"
    assert cfg.model.qk_layernorm is True
    expected_position_embedding = "yarn" if recipe_func.__name__ in _YARN_RECIPES else "rope"
    assert cfg.model.position_embedding_type == expected_position_embedding


def test_yarn_128k_recipe_sets_yarn_on_transformer_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """YaRN settings must reach the nested transformer config that Megatron Core reads."""
    patch_recipe_module_global(
        monkeypatch, qwen3.qwen3_600m_sft_8gpu_h100_bf16_yarn_128k_config, "AutoBridge", _BuilderOnlyBridge
    )

    cfg = qwen3.qwen3_600m_sft_8gpu_h100_bf16_yarn_128k_config()

    transformer = cfg.model.transformer
    assert cfg.model.position_embedding_type == "yarn"
    assert transformer.yarn_original_max_position_embeddings == 40960
    assert transformer.yarn_rotary_scaling_factor == cfg.model.seq_length / 40960
    assert transformer.yarn_beta_fast == 32.0
    assert transformer.yarn_beta_slow == 1.0
    assert transformer.yarn_mscale == 1.0
    assert transformer.yarn_mscale_all_dim == 1.0
    assert transformer.yarn_correction_range_round_to_int is False


def test_yarn_128k_recipe_uses_disjoint_train_and_validation_slices(monkeypatch):
    monkeypatch.setattr(qwen3, "AutoBridge", _FakeBridge)

    config = qwen3.qwen3_600m_sft_8gpu_h100_bf16_yarn_128k_config()

    assert isinstance(config.dataset, GPTSFTDatasetConfig)
    assert config.dataset.hf_dataset.split == "train[1%:]"
    assert config.dataset.hf_validation_dataset.split == "train[:1%]"
    assert config.dataset.hf_dataset.path_or_dataset == config.dataset.hf_validation_dataset.path_or_dataset
    assert config.dataset.hf_dataset.subset == config.dataset.hf_validation_dataset.subset == "math"
    assert config.dataset.hf_dataset.load_kwargs == config.dataset.hf_validation_dataset.load_kwargs
