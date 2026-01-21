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
# - Parametrize over all exported GLM-4.5V recipe functions in `megatron.bridge.recipes.glm_vl.glm_45v`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
# - Test pipeline model parallel layout for asymmetric stages.
#

import importlib
from typing import Callable

import pytest


_glm_45v_module = importlib.import_module("megatron.bridge.recipes.glm_vl.glm_45v")
_GLM_45V_RECIPE_FUNCS = [
    _glm_45v_module.glm_45v_finetune_config,
]


def _safe_overrides_for(name: str) -> dict:
    """Create safe test overrides for a given recipe function name."""
    overrides = {
        "name": f"unit_{name}",
        "dir": ".",
        "dataset_type": "mock",
        "train_iters": 10,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "seq_length": 64,
        "finetune_lr": 1e-4,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "virtual_pipeline_model_parallel_size": None,
    }

    return overrides


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def __init__(self):
        # Set default attributes that recipes might set
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.expert_model_parallel_size = 1
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_vision_projection = False
        # Pipeline layout attributes
        self.pipeline_model_parallel_layout = None
        self.account_for_embedding_in_pipeline_split = True
        self.account_for_loss_in_pipeline_split = True
        self.num_layers_in_first_pipeline_stage = None
        self.num_layers_in_last_pipeline_stage = None

    def finalize(self):
        return None


class _FakeAutoBridge:
    """Fake AutoBridge for testing."""

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        """Mock from_hf_pretrained method."""
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        """Return a fake model config."""
        return _FakeModelCfg()


def _assert_basic_config(cfg):
    """Assert that a config has all required components."""
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
    assert cfg.dataset.seq_length >= 1


@pytest.mark.parametrize("recipe_func", _GLM_45V_RECIPE_FUNCS)
def test_each_glm_45v_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each GLM-4.5V recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check that NullTokenizer is used
    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Verify parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1

    # Verify freeze settings are set
    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")


@pytest.mark.parametrize("dataset_type", ["mock", "hf", "preloaded"])
def test_glm_45v_dataset_type_selection(dataset_type: str, monkeypatch: pytest.MonkeyPatch):
    """Test that different dataset_type values produce correct dataset providers."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["dataset_type"] = dataset_type

    # For preloaded, we need to provide data paths
    if dataset_type == "preloaded":
        overrides["train_data_path"] = ["/fake/train.json"]
        overrides["valid_data_path"] = ["/fake/valid.json"]
        overrides["test_data_path"] = ["/fake/test.json"]
        overrides["image_folder"] = "/fake/images"

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    # Check that appropriate dataset provider is used
    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
    from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider
    from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider

    if dataset_type == "mock":
        assert isinstance(cfg.dataset, MockVLMConversationProvider)
    elif dataset_type == "hf":
        assert isinstance(cfg.dataset, HFDatasetConversationProvider)
    elif dataset_type == "preloaded":
        assert isinstance(cfg.dataset, PreloadedVLMConversationProvider)


def test_glm_45v_freeze_options(monkeypatch: pytest.MonkeyPatch):
    """Test that freeze options are correctly passed to the model config."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = True
    overrides["freeze_vision_projection"] = False

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False


def test_glm_45v_invalid_dataset_type(monkeypatch: pytest.MonkeyPatch):
    """Test that invalid dataset_type raises ValueError."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["dataset_type"] = "invalid_type"

    with pytest.raises(ValueError, match="Unsupported dataset_type"):
        _glm_45v_module.glm_45v_finetune_config(**overrides)


# PEFT-specific tests
_GLM_45V_FINETUNE_FUNCS = [
    _glm_45v_module.glm_45v_finetune_config,
]


@pytest.mark.parametrize("recipe_func", _GLM_45V_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "dora", None])
def test_glm_45v_finetune_peft_vs_full_sft(recipe_func, peft, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT and full SFT configurations are correctly applied for GLM-4.5V models."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    if peft in ["lora", "dora"]:
        assert cfg.peft is not None
        # Verify PEFT config has expected attributes
        assert hasattr(cfg.peft, "dim")
        assert hasattr(cfg.peft, "alpha")
    elif peft is None:
        assert cfg.peft is None


def test_glm_45v_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that GLM-4.5V LoRA has correct default parallelism and learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["peft"] = "lora"
    # Remove parallelism overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)
    # Remove finetune_lr to test default
    overrides.pop("finetune_lr", None)

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, GLM-4.5V should use TP=1, PP=4, EP=2
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 2

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32

    # Check that learning rate defaults to 1e-4 for LoRA
    assert cfg.optimizer.lr == 1e-4


def test_glm_45v_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that GLM-4.5V DoRA has correct default parallelism and learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["peft"] = "dora"
    # Remove parallelism overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For DoRA, GLM-4.5V should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 2

    # Check PEFT config (DoRA has alpha=64 by default, unlike LoRA's alpha=32)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 64


def test_glm_45v_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that GLM-4.5V full SFT has correct default parallelism and learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["peft"] = None
    # Remove parallelism overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)
    # Remove finetune_lr to test default
    overrides.pop("finetune_lr", None)

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, GLM-4.5V should use TP=1, PP=4, EP=16
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.peft is None

    # Check that learning rate defaults to 5e-6 for full SFT
    assert cfg.optimizer.lr == 5e-6


def test_glm_45v_custom_finetune_lr(monkeypatch: pytest.MonkeyPatch):
    """Test that custom finetune_lr overrides default learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["peft"] = "lora"
    overrides["finetune_lr"] = 2e-4  # Custom learning rate

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check that custom learning rate is used
    assert cfg.optimizer.lr == 2e-4


def test_glm_45v_peft_with_freeze_options(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT can be combined with freeze options."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["peft"] = "lora"
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = False
    overrides["freeze_vision_projection"] = True

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config
    assert cfg.peft is not None

    # Check freeze options
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is True


# Pipeline layout tests
def test_glm_45v_pipeline_layout_pp1():
    """Test pipeline layout for PP=1."""
    model_cfg = _FakeModelCfg()
    model_cfg.pipeline_model_parallel_size = 1
    model_cfg.virtual_pipeline_model_parallel_size = 1

    _glm_45v_module.set_glm_45v_pipeline_model_parallel_layout(model_cfg)

    # PP=1 should have no layout (None)
    assert model_cfg.pipeline_model_parallel_layout is None


def test_glm_45v_pipeline_layout_pp2():
    """Test pipeline layout for PP=2."""
    model_cfg = _FakeModelCfg()
    model_cfg.pipeline_model_parallel_size = 2
    model_cfg.virtual_pipeline_model_parallel_size = 1

    _glm_45v_module.set_glm_45v_pipeline_model_parallel_layout(model_cfg)

    # PP=2 should split 46 layers: first stage 1+23=24, second stage 23
    assert model_cfg.pipeline_model_parallel_layout is not None
    assert len(model_cfg.pipeline_model_parallel_layout) == 2
    # First stage: embedding + 23 decoder layers
    assert model_cfg.pipeline_model_parallel_layout[0][0] == "embedding"
    assert model_cfg.pipeline_model_parallel_layout[0].count("decoder") == 23
    # Last stage: 23 decoder layers + loss
    assert model_cfg.pipeline_model_parallel_layout[1].count("decoder") == 23
    assert "loss" in model_cfg.pipeline_model_parallel_layout[1]


def test_glm_45v_pipeline_layout_pp4():
    """Test pipeline layout for PP=4."""
    model_cfg = _FakeModelCfg()
    model_cfg.pipeline_model_parallel_size = 4
    model_cfg.virtual_pipeline_model_parallel_size = 1

    _glm_45v_module.set_glm_45v_pipeline_model_parallel_layout(model_cfg)

    # PP=4 should have 4 stages
    assert model_cfg.pipeline_model_parallel_layout is not None
    assert len(model_cfg.pipeline_model_parallel_layout) == 4
    # First stage: embedding + 11 decoder layers
    assert model_cfg.pipeline_model_parallel_layout[0][0] == "embedding"
    # Last stage should have loss
    assert "loss" in model_cfg.pipeline_model_parallel_layout[-1]


def test_glm_45v_pipeline_layout_pp8():
    """Test pipeline layout for PP=8."""
    model_cfg = _FakeModelCfg()
    model_cfg.pipeline_model_parallel_size = 8
    model_cfg.virtual_pipeline_model_parallel_size = 1

    _glm_45v_module.set_glm_45v_pipeline_model_parallel_layout(model_cfg)

    # PP=8 should have 8 stages
    assert model_cfg.pipeline_model_parallel_layout is not None
    assert len(model_cfg.pipeline_model_parallel_layout) == 8
    # First stage: embedding + 5 decoder layers
    assert model_cfg.pipeline_model_parallel_layout[0][0] == "embedding"
    # Last stage should have loss
    assert "loss" in model_cfg.pipeline_model_parallel_layout[-1]


def test_glm_45v_pipeline_layout_pp16():
    """Test pipeline layout for PP=16."""
    model_cfg = _FakeModelCfg()
    model_cfg.pipeline_model_parallel_size = 16
    model_cfg.virtual_pipeline_model_parallel_size = 1

    _glm_45v_module.set_glm_45v_pipeline_model_parallel_layout(model_cfg)

    # PP=16 should have 16 stages
    assert model_cfg.pipeline_model_parallel_layout is not None
    assert len(model_cfg.pipeline_model_parallel_layout) == 16
    # First stage: embedding + 2 decoder layers
    assert model_cfg.pipeline_model_parallel_layout[0][0] == "embedding"
    assert model_cfg.pipeline_model_parallel_layout[0].count("decoder") == 2
    # Last stage should have loss
    assert "loss" in model_cfg.pipeline_model_parallel_layout[-1]


def test_glm_45v_pipeline_layout_custom():
    """Test that custom pipeline layout overrides defaults."""
    model_cfg = _FakeModelCfg()
    model_cfg.pipeline_model_parallel_size = 2
    model_cfg.virtual_pipeline_model_parallel_size = 1

    custom_layout = [["embedding"] + ["decoder"] * 20, ["decoder"] * 26 + ["loss"]]
    _glm_45v_module.set_glm_45v_pipeline_model_parallel_layout(model_cfg, layout=custom_layout)

    # Custom layout should be used
    assert model_cfg.pipeline_model_parallel_layout == custom_layout


def test_glm_45v_pipeline_layout_in_config(monkeypatch: pytest.MonkeyPatch):
    """Test that pipeline layout is correctly set in the full config."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["pipeline_model_parallel_size"] = 2

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check that pipeline layout is set
    assert cfg.model.pipeline_model_parallel_layout is not None
    # Check that asymmetric pipeline split settings are disabled
    assert cfg.model.account_for_embedding_in_pipeline_split is False
    assert cfg.model.account_for_loss_in_pipeline_split is False


def test_glm_45v_wandb_logging(monkeypatch: pytest.MonkeyPatch):
    """Test that W&B logging options are correctly passed."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["wandb_project"] = "test_project"
    overrides["wandb_entity"] = "test_entity"
    overrides["wandb_exp_name"] = "test_exp"

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.logger.wandb_project == "test_project"
    assert cfg.logger.wandb_entity == "test_entity"
    assert cfg.logger.wandb_exp_name == "test_exp"


def test_glm_45v_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that precision config is correctly set."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Default should be bf16_mixed
    assert cfg.mixed_precision == "bf16_mixed"


def test_glm_45v_peft_none_string(monkeypatch: pytest.MonkeyPatch):
    """Test that peft='none' (string) is treated as full SFT."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["peft"] = "none"
    # Remove parallelism overrides to test recipe defaults
    overrides.pop("expert_model_parallel_size", None)
    overrides.pop("finetune_lr", None)

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # peft="none" should be treated as full SFT
    assert cfg.peft is None
    # Should use full SFT defaults: EP=16, LR=5e-6
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.optimizer.lr == 5e-6


def test_glm_45v_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that DDP config is correctly set."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check DDP settings
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"


def test_glm_45v_megatron_fsdp(monkeypatch: pytest.MonkeyPatch):
    """Test that Megatron FSDP option is correctly passed."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("glm_45v_finetune_config")
    overrides["use_megatron_fsdp"] = True

    cfg = _glm_45v_module.glm_45v_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.ddp.use_megatron_fsdp is True
