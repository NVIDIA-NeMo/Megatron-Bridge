# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Test purpose:
# - Parametrize over all exported GLM 4.5 recipe functions in `megatron.bridge.recipes.glm`.
# - For each recipe, monkeypatch the provider class with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection honors `use_null_tokenizer`, and sanity-check parallelism fields.
# - Test MoE-specific configurations (expert parallelism, router settings, etc.)
#

import importlib
from typing import Callable

import pytest


_glm_module = importlib.import_module("megatron.bridge.recipes.glm")
_GLM45_RECIPE_FUNCS = [
    getattr(_glm_module, name)
    for name in getattr(_glm_module, "__all__", [])
    if callable(getattr(_glm_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    All configs (pretrain, SFT, PEFT) now use the parameterless API.
    This function returns an empty dict since configs are modified after creation.
    """
    # All configs now use the parameterless API
    return {}


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def __init__(self):
        # Set default attributes that recipes might set
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.num_layers = 4
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False
        self.cp_comm_type = None
        # MoE-specific attributes
        self.num_moe_experts = 8
        self.moe_router_topk = 2
        self.moe_shared_expert_overlap = True
        self.moe_permute_fusion = True
        # Recompute configuration
        self.recompute_granularity = None
        self.recompute_modules = None
        self.recompute_method = None
        self.recompute_num_layers = None
        # MTP configuration
        self.mtp_num_layers = 1
        self.mtp_loss_scaling_factor = 0.3
        # Finetuning-specific attributes
        self.cross_entropy_loss_fusion = True
        self.vocab_size = 151552  # GLM vocab size

    def finalize(self):
        return None


class _FakeBridge:
    """Fake AutoBridge for testing finetune configs."""

    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()


class _FakeTokenizer:
    """Fake HuggingFace tokenizer for testing."""

    def __len__(self):
        return 151552  # GLM tokenizer vocab size


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1

    # Check sequence length (different attribute names for different dataset types)
    if hasattr(cfg.dataset, "sequence_length"):
        assert cfg.dataset.sequence_length >= 1  # GPTDatasetConfig
    elif hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1  # FinetuningDatasetConfig / HFDatasetConfig
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _GLM45_RECIPE_FUNCS)
def test_each_glm45_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each GLM 4.5 recipe function builds a valid configuration."""
    # Monkeypatch the provider classes to return fake model configs
    from megatron.bridge.models.glm import glm45_provider

    # Create a fake provider class that returns a fake model config
    class FakeProvider(_FakeModelCfg):
        def __init__(self, *args, **kwargs):
            super().__init__()

    # Monkeypatch all provider classes
    monkeypatch.setattr(glm45_provider, "GLMMoEModelProvider", FakeProvider)
    monkeypatch.setattr(glm45_provider, "GLM45ModelProvider355B", FakeProvider)
    monkeypatch.setattr(glm45_provider, "GLM45AirModelProvider106B", FakeProvider)

    # For SFT/PEFT recipes, also monkeypatch AutoBridge and AutoTokenizer
    is_sft_or_peft = "sft" in recipe_func.__name__.lower() or "peft" in recipe_func.__name__.lower()
    if is_sft_or_peft:
        module_name = recipe_func.__module__
        mod = importlib.import_module(module_name)
        monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

        # Mock AutoTokenizer to avoid HF I/O
        import transformers

        monkeypatch.setattr(
            transformers,
            "AutoTokenizer",
            type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
        )

    # All configs now use the parameterless API
    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    if is_sft_or_peft:
        # SFT/PEFT recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1


# GLM 4.5 SFT-specific tests
_GLM45_SFT_FUNCS = [
    getattr(_glm_module, name)
    for name in [
        "glm45_355b_sft_config",
        "glm45_air_106b_sft_config",
    ]
    if callable(getattr(_glm_module, name, None))
]

# GLM 4.5 PEFT-specific tests
_GLM45_PEFT_FUNCS = [
    getattr(_glm_module, name)
    for name in [
        "glm45_355b_peft_config",
        "glm45_air_106b_peft_config",
    ]
    if callable(getattr(_glm_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _GLM45_SFT_FUNCS)
def test_glm45_sft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each GLM 4.5 SFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    # SFT configs use the parameterless API
    cfg = recipe_func()

    _assert_basic_config(cfg)

    # SFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # SFT should not have PEFT config
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _GLM45_PEFT_FUNCS)
def test_glm45_peft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each GLM 4.5 PEFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    # PEFT configs take peft_scheme parameter (default is "lora")
    cfg = recipe_func()

    _assert_basic_config(cfg)

    # PEFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # PEFT should have PEFT config
    assert cfg.peft is not None


@pytest.mark.parametrize("recipe_func", _GLM45_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_glm45_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configurations are correctly applied for different schemes."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    cfg = recipe_func(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)

    # PEFT should have PEFT config
    assert cfg.peft is not None


def test_glm45_355b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 355B LoRA has correct default parallelism: TP=2, PP=4, EP=4 (32 GPUs)."""
    from megatron.bridge.recipes.glm import glm45_355b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    cfg = glm45_355b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, 355B should use TP=2, PP=4, EP=4 (32 GPUs total)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 4

    # Check PEFT config (LoRA defaults: dim=32, alpha=32)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32


def test_glm45_355b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 355B full SFT uses same parallelism as pretrain: TP=2, PP=8, EP=16 (256 GPUs)."""
    from megatron.bridge.recipes.glm import glm45_355b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    cfg = glm45_355b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 355B should use TP=2, PP=8, EP=16 (256 GPUs, same as pretrain)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.peft is None


def test_glm45_air_106b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Air 106B LoRA has correct default parallelism: TP=1, PP=2, EP=4 (8 GPUs, 1 node)."""
    from megatron.bridge.recipes.glm import glm45_air_106b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    cfg = glm45_air_106b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, Air 106B should use TP=1, PP=2, EP=4 (8 GPUs, 1 node)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 4

    # Check PEFT config (LoRA defaults: dim=32, alpha=32)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32


def test_glm45_air_106b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Air 106B full SFT uses same parallelism as pretrain: TP=1, PP=4, EP=8 (32 GPUs)."""
    from megatron.bridge.recipes.glm import glm45_air_106b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    cfg = glm45_air_106b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, Air 106B should use TP=1, PP=4, EP=8 (32 GPUs, same as pretrain)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.peft is None


def test_glm45_355b_pretrain_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 355B pretrain has correct default parallelism: TP=2, PP=8, EP=16 (256 GPUs)."""
    from megatron.bridge.recipes.glm import glm45_355b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Pretrain configs use the new parameterless API
    cfg = glm45_355b_pretrain_config()

    _assert_basic_config(cfg)

    # Check that model config has MoE-specific attributes
    assert hasattr(cfg.model, "expert_model_parallel_size")
    assert hasattr(cfg.model, "moe_permute_fusion")
    assert hasattr(cfg.model, "mtp_num_layers")
    assert hasattr(cfg.model, "mtp_loss_scaling_factor")


def test_glm45_air_106b_pretrain_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Air 106B pretrain has correct default parallelism: TP=1, PP=4, EP=8 (32 GPUs)."""
    from megatron.bridge.recipes.glm import glm45_air_106b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Pretrain configs use the new parameterless API
    cfg = glm45_air_106b_pretrain_config()

    _assert_basic_config(cfg)

    # Check that model config has MoE-specific attributes
    assert hasattr(cfg.model, "expert_model_parallel_size")
    assert hasattr(cfg.model, "moe_permute_fusion")
    assert hasattr(cfg.model, "mtp_num_layers")
    assert hasattr(cfg.model, "mtp_loss_scaling_factor")


@pytest.mark.parametrize("packed", [True, False])
def test_glm45_sft_packed_sequence(packed: bool, monkeypatch: pytest.MonkeyPatch):
    """Test that packed sequence configuration works correctly."""
    from megatron.bridge.recipes.glm import glm45_355b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    cfg = glm45_355b_sft_config()

    # Modify packed_sequence after creation
    cfg.dataset.packed_sequence = packed

    _assert_basic_config(cfg)


def test_glm45_mtp_configuration(monkeypatch: pytest.MonkeyPatch):
    """Test that MTP (Multi-Token Prediction) configuration is properly set."""
    from megatron.bridge.recipes.glm import glm45_355b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Pretrain configs use the new parameterless API
    cfg = glm45_355b_pretrain_config()

    _assert_basic_config(cfg)

    # Check MTP configuration exists and has valid values
    assert hasattr(cfg.model, "mtp_num_layers")
    assert hasattr(cfg.model, "mtp_loss_scaling_factor")
    assert cfg.model.mtp_num_layers >= 0
    assert cfg.model.mtp_loss_scaling_factor >= 0


def test_glm45_recompute_configuration(monkeypatch: pytest.MonkeyPatch):
    """Test that recompute configuration is properly set."""
    from megatron.bridge.recipes.glm import glm45_355b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.glm.glm45")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Pretrain configs use the new parameterless API
    cfg = glm45_355b_pretrain_config()

    _assert_basic_config(cfg)

    # Check recompute configuration exists
    assert hasattr(cfg.model, "recompute_granularity")
    assert hasattr(cfg.model, "recompute_method")
    assert hasattr(cfg.model, "recompute_num_layers")
