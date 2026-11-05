# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# - Parametrize over all exported Qwen3.5-VL recipe functions.
# - Monkeypatch AutoBridge and the provider to avoid I/O and heavy model init.
# - Build a config and assert it forms a valid ConfigContainer.
# - Verify dataset provider selection, parallelism fields, freeze options, and PEFT defaults.
#

import importlib
import inspect
from dataclasses import fields
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from megatron.bridge.peft.dora import DoRA
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_qwen35_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen35_vl")
_qwen35_vl_h100_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.h100.qwen35_vl")
_qwen35_vl_gb200_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.gb200.qwen35_vl")
_qwen35_vl_gb300_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.gb300.qwen35_vl")

# Pretrain mock configs (parameterless fixed configs)
_QWEN35_VL_PRETRAIN_MOCK_FUNCS = [
    _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config,
    _qwen35_vl_module.qwen35_vl_27b_pretrain_mock_config,
    _qwen35_vl_module.qwen35_vl_35b_a3b_pretrain_mock_config,
    _qwen35_vl_module.qwen35_vl_122b_a10b_pretrain_mock_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_pretrain_mock_config,
]

_QWEN35_VL_H100_PRETRAIN_MOCK_FUNCS = [
    _qwen35_vl_h100_module.qwen35_vl_9b_pretrain_4gpu_h100_bf16_mock_config,
    _qwen35_vl_h100_module.qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config,
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config,
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_pretrain_config,
    _qwen35_vl_h100_module.qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_mock_config,
    _qwen35_vl_h100_module.qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config,
]

# SFT configs (parameterless)
_QWEN35_VL_SFT_FUNCS = [
    _qwen35_vl_module.qwen35_vl_800m_sft_config,
    _qwen35_vl_module.qwen35_vl_2b_sft_config,
    _qwen35_vl_module.qwen35_vl_4b_sft_config,
    _qwen35_vl_module.qwen35_vl_9b_sft_config,
    _qwen35_vl_module.qwen35_vl_27b_sft_config,
    _qwen35_vl_module.qwen35_vl_35b_a3b_sft_config,
    _qwen35_vl_module.qwen35_vl_35b_a3b_fsdp_sft_config,
    _qwen35_vl_module.qwen35_vl_122b_a10b_sft_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_sft_config,
]

_QWEN35_VL_H100_SFT_FUNCS = [
    _qwen35_vl_h100_module.qwen35_vl_800m_sft_1gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_2b_sft_1gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_4b_sft_2gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_9b_sft_4gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_27b_sft_16gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_sft_2gpu_h100_bf16_fsdp_config,
    _qwen35_vl_h100_module.qwen35_vl_122b_a10b_sft_48gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_397b_a17b_sft_128gpu_h100_bf16_config,
]

# PEFT configs (fixed LoRA recipes)
_QWEN35_VL_PEFT_FUNCS = [
    _qwen35_vl_module.qwen35_vl_800m_peft_config,
    _qwen35_vl_module.qwen35_vl_2b_peft_config,
    _qwen35_vl_module.qwen35_vl_4b_peft_config,
    _qwen35_vl_module.qwen35_vl_9b_peft_config,
    _qwen35_vl_module.qwen35_vl_27b_peft_config,
    _qwen35_vl_module.qwen35_vl_35b_a3b_peft_config,
    _qwen35_vl_module.qwen35_vl_122b_a10b_peft_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_peft_config,
]

_QWEN35_VL_H100_PEFT_FUNCS = [
    _qwen35_vl_h100_module.qwen35_vl_800m_peft_1gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_2b_peft_1gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_4b_peft_1gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_9b_peft_1gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_27b_peft_2gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_peft_4gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_peft_16gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_122b_a10b_peft_8gpu_h100_bf16_config,
    _qwen35_vl_h100_module.qwen35_vl_397b_a17b_peft_32gpu_h100_bf16_config,
]

_QWEN35_VL_GB200_FUNCS = [
    _qwen35_vl_gb200_module.qwen35_vl_27b_pretrain_16gpu_gb200_bf16_mock_config,
    _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_8gpu_gb200_bf16_functional_config,
    _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_bf16_config,
    _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_fp8mx_config,
    _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_peft_8gpu_gb200_bf16_functional_config,
]

_QWEN35_VL_GB300_FUNCS = [
    _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_config,
    _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config,
    _qwen35_vl_gb300_module.qwen35_vl_397b_a17b_pretrain_config,
]


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.num_layers_in_first_pipeline_stage = None
        self.num_layers_in_last_pipeline_stage = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_vision_projection = False
        # Present on a Megatron-Core that carries the GDN conv/L2-norm fusion.
        # The recipe guards on this field existing, so a fake without it models
        # an older core -- see _FakeModelCfgNoGdnFusion below.
        self.gdn_pre_gated_delta_rule_fusion = False

    def finalize(self):
        return None


class _FakeAutoBridge:
    """Fake AutoBridge for testing."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, *, revision: str | None = None):
        return _FakeAutoBridge()

    def get_model_config(self):
        return _FakeModelCfg()

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("Qwen3.5-VL recipes must use get_model_config()")


def _real_moe_model_config(**overrides):
    """Return a strict builder config for the default Qwen3.5 MoE architecture (35B-A3B shape)."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge

    model_config = Qwen35VLMoEBridge().hf_config_to_model_config(Qwen3_5MoeConfig())
    for name, value in overrides.items():
        setattr(model_config, name, value)
    return model_config


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


# ---------------------------------------------------------------------------
# Basic SFT recipe building tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _QWEN35_VL_SFT_FUNCS)
def test_each_qwen35_vl_sft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen3.5-VL SFT recipe function builds a valid configuration."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")

    assert cfg.peft is None


# ---------------------------------------------------------------------------
# Basic PEFT recipe building tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _QWEN35_VL_PEFT_FUNCS)
def test_each_qwen35_vl_peft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen3.5-VL PEFT recipe function builds a valid configuration."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")

    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


def test_qwen35_vl_model_selector_supports_dora(monkeypatch: pytest.MonkeyPatch):
    """Qwen3.5-VL model selectors should honor the requested DoRA scheme."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_peft_config(peft_scheme="dora")

    assert isinstance(cfg.peft, DoRA)


# ---------------------------------------------------------------------------
# Recipe API shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recipe_func",
    _QWEN35_VL_PRETRAIN_MOCK_FUNCS
    + _QWEN35_VL_H100_PRETRAIN_MOCK_FUNCS
    + _QWEN35_VL_SFT_FUNCS
    + _QWEN35_VL_H100_SFT_FUNCS
    + [_qwen35_vl_h100_module.qwen35_vl_35b_a3b_peft_16gpu_h100_bf16_config]
    + _QWEN35_VL_GB200_FUNCS
    + _QWEN35_VL_GB300_FUNCS,
)
def test_qwen35_vl_recipe_entry_points_are_parameterless(recipe_func: Callable):
    """Qwen3.5-VL public recipe entry points should be fixed configs."""
    assert not inspect.signature(recipe_func).parameters


@pytest.mark.parametrize("recipe_func", _QWEN35_VL_PEFT_FUNCS)
def test_qwen35_vl_selectable_peft_recipes_accept_a_peft_scheme(recipe_func: Callable):
    """Model-selectable PEFT recipes should accept an optional PEFT scheme."""
    parameter = inspect.signature(recipe_func).parameters["peft_scheme"]

    assert parameter.default == "lora"


def test_qwen35_vl_h100_module_has_no_parameterized_recipe_helpers():
    """Qwen3.5-VL H100 recipes should be flattened rather than built through private templates."""
    assert not hasattr(_qwen35_vl_h100_module, "_qwen35_vl_apply_common")
    assert not hasattr(_qwen35_vl_h100_module, "_qwen35_vl_apply_moe")
    assert not hasattr(_qwen35_vl_h100_module, "_qwen35_vl_enable_recompute")


# ---------------------------------------------------------------------------
# 800M dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_800m_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """800M SFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


def test_qwen35_vl_800m_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """800M PEFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


# ---------------------------------------------------------------------------
# 2B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_2b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """2B SFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_2b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


def test_qwen35_vl_2b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """2B PEFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_2b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


# ---------------------------------------------------------------------------
# 4B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_4b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """4B SFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_4b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


def test_qwen35_vl_4b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """4B PEFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_4b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


# ---------------------------------------------------------------------------
# 9B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_9b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """9B SFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_9b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


def test_qwen35_vl_9b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """9B PEFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_9b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


# ---------------------------------------------------------------------------
# 27B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_27b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """27B SFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_27b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


def test_qwen35_vl_27b_peft_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """27B LoRA should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_27b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_dtype is None
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.optimizer.lr == 1e-4


# ---------------------------------------------------------------------------
# 35B-A3B MoE defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_397b_a17b_pretrain_64gpu_gb300_defaults(monkeypatch: pytest.MonkeyPatch):
    """The 64-GB300 library pretrain recipe should own the measured execution policy."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_gb300_module.qwen35_vl_397b_a17b_pretrain_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    # EP32 measured 370.2 TF against 211.0 TF at EP64 on 64x GB300; the HybridEP
    # NVLink rank count must track it or the domain is split wrongly.
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.recompute_granularity == "selective"
    # No "moe_act": the CuTe DSL fused grouped MLP rejects it.
    assert cfg.model.recompute_modules == ["core_attn", "gdn_norm_out"]
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["attn", "moe_router", "moe_preprocess"]
    # Vision graphs at the cap measured for micro_batch_size=1 (max 980 + 5%).
    assert cfg.model.vision_cuda_graph_impl == "transformer_engine"
    assert cfg.model.vision_cuda_graph_scope == ["attn", "mlp"]
    assert cfg.model.max_vision_cuda_graph_seq_length == 1029
    assert cfg.train.global_batch_size == 1024
    # micro_batch_size is load-bearing for the vision graph cap above.
    assert cfg.train.micro_batch_size == 1
    assert cfg.checkpoint.pretrained_checkpoint is None


def test_qwen35_vl_35b_a3b_pretrain_16gpu_gb300_defaults(monkeypatch: pytest.MonkeyPatch):
    """The 16-GB300 library pretrain recipe should own the measured execution policy."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    # EP4 on 16 GPUs; the HybridEP NVLink rank count must track it or the
    # domain is split wrongly.
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 4
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    # LayerNorm SM margin is 0 at 35B, not the 397B recipe's 20.
    assert cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 0
    assert cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 0
    assert cfg.model.recompute_granularity == "selective"
    # No "moe_act": the CuTe DSL fused grouped MLP rejects it.
    assert cfg.model.recompute_modules == ["core_attn", "gdn_norm_out"]
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["attn", "moe_router", "moe_preprocess"]
    # Vision graphs at the cap measured for micro_batch_size=4 (max 3532 + 5%).
    assert cfg.model.vision_cuda_graph_impl == "transformer_engine"
    assert cfg.model.vision_cuda_graph_scope == ["attn", "mlp"]
    assert cfg.model.max_vision_cuda_graph_seq_length == 3712
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.cross_entropy_loss_fusion is True
    assert cfg.model.cross_entropy_fusion_impl == "te"
    assert cfg.train.global_batch_size == 2048
    # micro_batch_size is load-bearing for the vision graph cap above.
    assert cfg.train.micro_batch_size == 4
    assert cfg.tokenizer.use_tokenizer_vocab_size is False
    assert cfg.checkpoint.pretrained_checkpoint is None


def test_qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_defaults(monkeypatch: pytest.MonkeyPatch):
    """The 8-GB300 MXFP8 recipe adds the measured precision, kernel and dispatcher settings to the 16-GPU recipe."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_config()
    base = _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config()

    _assert_basic_config(cfg)
    # Same parallel layout, CUDA graphs, recompute, batch shape and routing as the 16-GPU recipe.
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 4
    assert cfg.train.global_batch_size == base.train.global_batch_size == 2048
    assert cfg.train.micro_batch_size == base.train.micro_batch_size == 4
    assert cfg.model.max_vision_cuda_graph_seq_length == 3712
    assert cuda_graph_module_names(cfg.model) == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.recompute_modules == ["core_attn", "gdn_norm_out"]
    assert cfg.model.moe_router_force_load_balancing is False
    # MXFP8 with FP8 parameter all-gather.
    assert cfg.mixed_precision.fp8 == "e4m3"
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    # CuTe DSL fused grouped MLP needs all three settings together.
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_permute_fusion_into_hybridep is False
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 16
    assert cfg.model.overlap_dispatch_backward_with_experts_wgrad is True
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 128
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API"] == 64
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API"] == 64
    assert cfg.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1
    assert cfg.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    # Building the 8-GPU recipe must not leak the env gate into the 16-GPU recipe.
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in base.env_vars

    # These are the only model-field and environment-variable differences from the 16-GPU recipe; an extra model
    # field or environment variable fails here. Mixed precision is checked above on five fields only, and other
    # sections, such as train, optimizer, ddp and comm_overlap, are not compared.
    changed_model_fields = {
        name
        for name in vars(cfg.model).keys() | vars(base.model).keys()
        if getattr(cfg.model, name, None) != getattr(base.model, name, None)
    }
    assert changed_model_fields == {
        "use_transformer_engine_op_fuser",
        "moe_mlp_glu_interleave_size",
        "moe_flex_dispatcher_num_sms",
        "moe_permute_fusion_into_hybridep",
        "moe_hybridep_num_sms_preprocessing",
        "overlap_dispatch_backward_with_experts_wgrad",
    }
    changed_env_vars = {
        name
        for name in cfg.env_vars.keys() | base.env_vars.keys()
        if cfg.env_vars.get(name) != base.env_vars.get(name)
    }
    assert changed_env_vars == {
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API",
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API",
        "NVTE_NORM_FWD_USE_CUDNN",
        "NVTE_NORM_BWD_USE_CUDNN",
    }


class _FakeModelCfgWithGdnKernelBackend(_FakeModelCfg):
    """Model config from a Megatron-Core that declares gdn_kernel_backend (NVIDIA/Megatron-LM#7583)."""

    # Megatron-Core's default; a class attribute, as on the real dataclass.
    gdn_kernel_backend = "fla"


class _FakeAutoBridgeWithGdnKernelBackend(_FakeAutoBridge):
    """AutoBridge yielding a provider that declares gdn_kernel_backend."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, *, revision: str | None = None):
        return _FakeAutoBridgeWithGdnKernelBackend()

    def get_model_config(self):
        return _FakeModelCfgWithGdnKernelBackend()


def test_qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_selects_cudnn_gdn(monkeypatch: pytest.MonkeyPatch):
    """If Megatron-Core has gdn_kernel_backend, the 8-GPU recipe selects cuDNN GDN; the 16-GPU one keeps FLA."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridgeWithGdnKernelBackend)

    cfg = _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_config()
    base = _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config()

    assert cfg.model.gdn_kernel_backend == "transformer_engine"
    # The 16-GPU BF16 recipe keeps Megatron-Core's default.
    assert base.model.gdn_kernel_backend == "fla"


def test_qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_no_phantom_gdn_kernel_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    """On a Megatron-Core without gdn_kernel_backend the recipe must not invent the attribute.

    Assigning an unknown field would not raise: it would create an unused attribute, so the recipe would look as if it
    selected cuDNN GDN while the layers keep running on FLA.
    """
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_gb300_module.qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_config()

    assert not hasattr(cfg.model, "gdn_kernel_backend")


def test_qwen35_vl_35b_a3b_pretrain_16gpu_h100_defaults(monkeypatch: pytest.MonkeyPatch):
    """The 16-H100 library pretrain recipe should own the measured execution policy."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_h100_module.qwen35_vl_35b_a3b_pretrain_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.num_layers_in_first_pipeline_stage == 16
    assert cfg.model.num_layers_in_last_pipeline_stage == 24
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.checkpoint.pretrained_checkpoint is None
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.overlap_dispatch_backward_with_experts_wgrad is True
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["core_attn", "gdn_norm_out", "moe_act"]
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.vision_cuda_graph_impl == "none"
    assert cfg.model.vision_cuda_graph_scope == []
    assert cfg.model.max_vision_cuda_graph_seq_length is None
    assert cfg.model.cross_entropy_loss_fusion is True
    assert cfg.model.cross_entropy_fusion_impl == "te"
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 1
    assert cfg.tokenizer.use_tokenizer_vocab_size is False
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False


def test_qwen35_vl_35b_a3b_pretrain_name_is_exported():
    """The canonical full-pretrain recipe should be available from the public recipes package."""
    recipe_name = "qwen35_vl_35b_a3b_pretrain_config"
    recipes_package = importlib.import_module("megatron.bridge.recipes")

    assert recipe_name in _qwen35_vl_h100_module.__all__
    assert getattr(recipes_package, recipe_name) is getattr(_qwen35_vl_h100_module, recipe_name)


def test_qwen35_vl_35b_a3b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Shared Qwen3.5/Qwen3.6 35B-A3B SFT should have tuned H100 defaults."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_35b_a3b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.num_layers_in_first_pipeline_stage == 17
    assert cfg.model.num_layers_in_last_pipeline_stage == 23
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.model.sequence_parallel is False
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 16
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 16
    assert cfg.model.moe_router_fusion is True
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.moe_permute_fusion is True
    assert cfg.model.moe_permute_fusion_into_hybridep is True
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.overlap_dispatch_backward_with_experts_wgrad is True
    assert cfg.peft is None
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.model.bias_activation_fusion is True
    assert cfg.model.apply_rope_fusion is False
    assert cfg.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(cfg.model) == []
    assert cfg.model.vision_cuda_graph_impl == "none"
    assert cfg.model.vision_cuda_graph_scope == []
    assert cfg.model.max_vision_cuda_graph_seq_length == 784
    assert cfg.model.cross_entropy_loss_fusion is True
    assert cfg.model.cross_entropy_fusion_impl == "te"
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.dataset.defer_in_batch_packing_to_step is True
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.optimizer.overlap_param_gather_with_optimizer_step is False
    assert cfg.optimizer.min_lr == 2e-6
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.average_in_collective is True
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.comm_overlap.overlap_grad_reduce is False
    assert cfg.comm_overlap.overlap_param_gather is False
    assert cfg.comm_overlap.overlap_param_gather_with_optimizer_step is False
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False
    assert cfg.model.batch_p2p_sync is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 0
    assert cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 0
    assert cfg.optimizer.lr == 2e-5


def test_qwen35_vl_35b_a3b_long_context_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Shared Qwen3.5/Qwen3.6 long-context SFT should own packing and CP defaults."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_h100_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_h100_bf16_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.calculate_per_token_loss is True
    # Qwen3.5/3.6 hybrid layers include GatedDeltaNet, which owns its CP
    # communication and does not accept the attention-only cp_comm_type kwarg.
    # Standard Transformer Engine attention defaults an unset value to p2p.
    assert getattr(cfg.model, "cp_comm_type", None) is None
    assert cfg.model.seq_length == 8192
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 2
    assert cfg.dataset.seq_length == 8192
    assert cfg.dataset.enable_in_batch_packing is True
    assert cfg.dataset.defer_in_batch_packing_to_step is True
    assert cfg.dataset.in_batch_packing_pad_to_multiple_of == 4
    assert cfg.ddp.average_in_collective is False


def test_qwen35_vl_35b_a3b_gb200_long_context_precision_pair(monkeypatch: pytest.MonkeyPatch):
    """The GB200 BF16 and MXFP8 recipes should share one 128K execution topology."""
    from megatron.bridge.data.builders import EnergonDatasetConfig, QwenVLEnergonTaskEncoderConfig

    monkeypatch.setattr(
        _FakeAutoBridge,
        "get_model_config",
        lambda self: _real_moe_model_config(bias_activation_fusion=False),
    )
    patch_recipe_module_global(monkeypatch, _qwen35_vl_gb200_module, "AutoBridge", _FakeAutoBridge)

    bf16_cfg = _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_bf16_config()
    fp8mx_cfg = _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_fp8mx_config()

    for cfg in (bf16_cfg, fp8mx_cfg):
        _assert_basic_config(cfg)
        assert cfg.model.seq_length == 131072
        assert cfg.model.bias_activation_fusion is True
        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.pipeline_model_parallel_size == 1
        assert cfg.model.pipeline_dtype is None
        assert cfg.model.virtual_pipeline_model_parallel_size is None
        assert cfg.model.context_parallel_size == 8
        assert cfg.model.expert_model_parallel_size == 8
        assert cfg.model.expert_tensor_parallel_size == 1
        assert cfg.model.sequence_parallel is False
        assert cfg.model.moe_token_dispatcher_type == "flex"
        assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
        assert cfg.model.moe_flex_dispatcher_num_sms == 32
        assert cfg.model.moe_hybridep_pad_uneven_dispatch_inputs is True
        assert cfg.model.gdn_kernel_backend == "transformer_engine"
        assert cfg.model.gdn_pre_gated_delta_rule_fusion is True
        assert cfg.model.gdn_gated_output_norm_fusion is False
        assert cfg.model.apply_rope_fusion is False
        assert cfg.model.cuda_graph_impl == "none"
        assert cfg.model.moe_router_force_load_balancing is False
        assert cfg.model.cross_entropy_fusion_impl == "te"
        assert cfg.model.recompute_granularity == "selective"
        assert cfg.model.recompute_modules == ["gdn_norm_out", "moe"]
        assert cfg.model.recompute_method is None
        assert cfg.model.recompute_num_layers is None
        assert cfg.model.vision_recompute_granularity == "full"
        assert cfg.model.vision_recompute_method == "uniform"
        assert cfg.model.vision_recompute_num_layers == 1
        assert cfg.model.vision_recompute_modules is None
        assert cfg.model.attention_backend.name == "auto"
        for variable in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
            assert cfg.env_vars[variable] == 1
        assert cfg.train.global_batch_size == 32
        assert cfg.train.micro_batch_size == 1
        assert cfg.dataset.seq_length == 131072
        assert cfg.dataset.enable_in_batch_packing is False
        assert cfg.dataset.defer_in_batch_packing_to_step is False
        assert isinstance(cfg.dataset, EnergonDatasetConfig)
        assert cfg.dataset.path is None  # Caller supplies prepared CLEVR2 shards.
        assert cfg.dataset.micro_batch_size == 1
        assert cfg.dataset.num_workers == 2
        assert cfg.dataset.num_val_workers == 1
        assert cfg.dataset.shuffle_buffer_size == 16
        assert cfg.dataset.packing_buffer_size == 16
        assert cfg.dataset.dataset_kwargs == {"image_decode_spec": "pilrgb"}
        encoder = cfg.dataset.task_encoder
        assert isinstance(encoder, QwenVLEnergonTaskEncoderConfig)
        assert encoder.hf_processor_path == "Qwen/Qwen3.5-35B-A3B"
        assert encoder.hf_processor_revision == _qwen35_vl_gb200_module._QWEN35_35B_A3B_REVISION
        assert encoder.min_pixels == 200704
        assert encoder.max_pixels == 401408
        assert encoder.max_num_images == 4
        assert encoder.max_visual_tokens == 2048
        assert cfg.mixed_precision.grad_reduce_in_fp32 is True
        assert cfg.ddp.grad_reduce_in_fp32 is True
        assert cfg.ddp.overlap_grad_reduce is False
        assert cfg.ddp.overlap_param_gather is False
        assert cfg.optimizer.overlap_param_gather is False
        assert cfg.comm_overlap.overlap_grad_reduce is False
        assert cfg.comm_overlap.overlap_param_gather is False
        assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
        assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1
        assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8

    assert bf16_cfg.mixed_precision.fp8 is None
    assert bf16_cfg.mixed_precision.fp8_param_gather is False
    assert bf16_cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is False
    assert bf16_cfg.checkpoint.load_main_params_from_ckpt is False
    assert bf16_cfg.model.moe_single_grouped_weight is False
    assert fp8mx_cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert fp8mx_cfg.mixed_precision.fp8_param_gather is True
    assert fp8mx_cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert fp8mx_cfg.checkpoint.load_main_params_from_ckpt is True
    assert fp8mx_cfg.checkpoint.load_optim is False
    assert fp8mx_cfg.checkpoint.load_rng is False

    assert bf16_cfg.model.use_transformer_engine_op_fuser is False
    assert bf16_cfg.model.moe_use_grouped_tensor is False
    assert bf16_cfg.model.moe_mlp_glu_interleave_size is None
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in bf16_cfg.env_vars
    assert fp8mx_cfg.model.use_transformer_engine_op_fuser is True
    assert fp8mx_cfg.model.moe_use_grouped_tensor is True
    assert fp8mx_cfg.model.moe_mlp_glu_interleave_size == 32
    assert fp8mx_cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert fp8mx_cfg.env_vars["CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL"] == 1
    assert fp8mx_cfg.model.moe_single_grouped_weight is True
    assert fp8mx_cfg.model.moe_single_grouped_bias is False
    assert fp8mx_cfg.env_vars["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] == 1


def test_qwen35_vl_35b_a3b_fsdp_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """35B-A3B FSDP SFT should have FSDP-specific parallelism and settings."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_35b_a3b_fsdp_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 2
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.sequence_parallel is False
    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.moe_router_fusion is True
    assert cfg.model.cross_entropy_loss_fusion is True
    assert cfg.model.cross_entropy_fusion_impl == "te"
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.fsdp_double_buffer is True
    assert cfg.ddp.megatron_fsdp_max_pool_double_buffer is True
    assert cfg.ddp.nccl_ub is False
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.ddp.num_distributed_optimizer_instances == 1


def test_qwen35_vl_35b_a3b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Shared Qwen3.5/Qwen3.6 35B-A3B PEFT should have safe recipe defaults."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_35b_a3b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.peft is not None
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.dataset.defer_in_batch_packing_to_step is True
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    assert cfg.optimizer.lr == 2e-4
    assert cfg.optimizer.min_lr == 3e-5


def test_qwen35_vl_35b_a3b_peft_16gpu_h100_defaults(monkeypatch: pytest.MonkeyPatch):
    """The 16-H100 LoRA recipe should combine the prior optimizer contract with tuned execution."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_h100_module.qwen35_vl_35b_a3b_peft_16gpu_h100_bf16_config()

    _assert_basic_config(cfg)
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 2e-4
    assert cfg.optimizer.min_lr == 3e-5
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.num_layers_in_first_pipeline_stage == 17
    assert cfg.model.num_layers_in_last_pipeline_stage == 23
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.overlap_dispatch_backward_with_experts_wgrad is False
    assert cfg.model.recompute_granularity is None
    assert cuda_graph_module_names(cfg.model) == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.vision_cuda_graph_scope == ["attn", "mlp"]


def test_qwen35_vl_27b_gb200_pretrain_defaults(monkeypatch: pytest.MonkeyPatch):
    """The dense GB200 pretrain recipe should retain its measured execution policy."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_gb200_module.qwen35_vl_27b_pretrain_16gpu_gb200_bf16_mock_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_dtype is None
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.calculate_per_token_loss is True

    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 2

    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.apply_rope_fusion is False
    assert cfg.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(cfg.model) == []

    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.average_in_collective is False
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.comm_overlap.overlap_grad_reduce is False
    assert cfg.comm_overlap.overlap_param_gather is False

    assert cfg.dataset.do_validation is False
    assert cfg.dataset.pad_to_max_length is True
    assert cfg.train.eval_interval == 0
    assert cfg.train.eval_iters == 0
    assert cfg.validation.eval_interval == 0
    assert cfg.validation.eval_iters == 0
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.save is None
    assert cfg.logger.log_interval == 1
    assert cfg.logger.log_throughput is True
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert cfg.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1


@pytest.mark.parametrize(
    ("recipe_func", "expected_lr", "is_peft"),
    [
        (_qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_8gpu_gb200_bf16_functional_config, 2e-5, False),
        (_qwen35_vl_gb200_module.qwen35_vl_35b_a3b_peft_8gpu_gb200_bf16_functional_config, 2e-4, True),
    ],
)
def test_qwen35_vl_35b_a3b_gb200_functional_defaults(
    recipe_func: Callable,
    expected_lr: float,
    is_peft: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """GB200 SFT and PEFT should share the measured functional execution policy."""
    from megatron.bridge.utils.cuda_graph import cuda_graph_module_names

    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)
    assert (cfg.peft is not None) is is_peft
    assert cfg.optimizer.lr == expected_lr

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_dtype is None
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.num_layers_in_first_pipeline_stage is None
    assert cfg.model.num_layers_in_last_pipeline_stage is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.empty_unused_memory_level == 2

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.high_priority_a2a_comm_stream is True

    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(cfg.model) == []
    assert cfg.model.use_te_rng_tracker is False
    assert cfg.rng.te_rng_tracker is False

    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.optimizer.overlap_param_gather is False
    assert cfg.optimizer.overlap_param_gather_with_optimizer_step is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.check_for_large_grads is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True

    assert cfg.dataset.do_validation is False
    assert cfg.dataset.pad_to_max_length is True
    assert cfg.validation.eval_interval == 0
    assert cfg.validation.eval_iters == 0
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.save is None
    assert cfg.logger.log_interval == 1
    assert cfg.logger.log_throughput is True
    assert cfg.logger.tensorboard_dir is None
    assert cfg.dist.distributed_timeout_minutes == 30

    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.comm_overlap.overlap_grad_reduce is False
    assert cfg.comm_overlap.overlap_param_gather is False
    assert cfg.comm_overlap.overlap_param_gather_with_optimizer_step is False
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False

    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1
    assert cfg.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert cfg.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1


# ---------------------------------------------------------------------------
# 122B-A10B MoE defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_122b_a10b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """122B-A10B SFT should have correct parallelism, batch size, and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_122b_a10b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 6
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert (
        cfg.model.pipeline_model_parallel_size
        * cfg.model.expert_model_parallel_size
        * cfg.model.expert_tensor_parallel_size
        == 48
    )
    data_parallel_size = cfg.get_data_parallel_size(48)
    samples_per_micro_step = cfg.train.micro_batch_size * data_parallel_size
    assert data_parallel_size == 4
    assert cfg.train.global_batch_size == 36
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size % samples_per_micro_step == 0
    assert cfg.train.global_batch_size // samples_per_micro_step == 9
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.recompute_granularity == "full"


def test_qwen35_vl_122b_a10b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """122B-A10B PEFT should have correct parallelism, batch size, and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_122b_a10b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    data_parallel_size = cfg.get_data_parallel_size(8)
    samples_per_micro_step = cfg.train.micro_batch_size * data_parallel_size
    assert data_parallel_size == 4
    assert cfg.train.global_batch_size == 36
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size % samples_per_micro_step == 0
    assert cfg.train.global_batch_size // samples_per_micro_step == 9
    assert cfg.model.pipeline_dtype is None
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 2e-4


# ---------------------------------------------------------------------------
# 397B-A17B MoE defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_397b_a17b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """397B-A17B SFT should have correct parallelism, batch size, and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 32
    data_parallel_size = cfg.get_data_parallel_size(128)
    samples_per_micro_step = cfg.train.micro_batch_size * data_parallel_size
    assert data_parallel_size == 16
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size % samples_per_micro_step == 0
    assert cfg.train.global_batch_size // samples_per_micro_step == 2
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.recompute_granularity == "full"


def test_qwen35_vl_397b_a17b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """397B-A17B PEFT should have correct parallelism, batch size, and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 32
    data_parallel_size = cfg.get_data_parallel_size(32)
    samples_per_micro_step = cfg.train.micro_batch_size * data_parallel_size
    assert data_parallel_size == 16
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size % samples_per_micro_step == 0
    assert cfg.train.global_batch_size // samples_per_micro_step == 2
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 2e-4
    assert cfg.model.pipeline_dtype is None


# ---------------------------------------------------------------------------
# Common config properties
# ---------------------------------------------------------------------------


def test_qwen35_vl_sft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs use DirectHFSFTDatasetConfig by default."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    from megatron.bridge.data.builders import DirectHFSFTDatasetConfig

    assert isinstance(cfg.dataset, DirectHFSFTDatasetConfig)


def test_qwen35_vl_peft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs use DirectHFSFTDatasetConfig by default."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_peft_config()

    from megatron.bridge.data.builders import DirectHFSFTDatasetConfig

    assert isinstance(cfg.dataset, DirectHFSFTDatasetConfig)


def test_qwen35_vl_sft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs have freeze options set to False by default."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_qwen35_vl_peft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs have freeze options set to False by default."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_peft_config()

    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_qwen35_vl_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that precision config is correctly set."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.mixed_precision == "bf16_mixed"


def test_qwen35_vl_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that DDP config is correctly set for VLMs."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_qwen35_vl_optimizer_precision_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that optimizer precision settings are correctly configured."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32


def test_qwen35_vl_training_config(monkeypatch: pytest.MonkeyPatch):
    """Test that training configuration is correctly set."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.train.train_iters == 300000
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 4
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100


def test_qwen35_vl_validation_config(monkeypatch: pytest.MonkeyPatch):
    """Test that validation configuration is correctly set."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.validation.eval_interval == 500
    assert cfg.validation.eval_iters == 32


def test_qwen35_vl_sft_learning_rate(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT has lower learning rate than PEFT."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    sft_cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()
    peft_cfg = _qwen35_vl_module.qwen35_vl_800m_peft_config()

    assert sft_cfg.optimizer.lr < peft_cfg.optimizer.lr


def test_qwen35_vl_kernel_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that kernel settings are correctly configured."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.attention_backend == "auto"
    assert cfg.model.gradient_accumulation_fusion is True
    assert cfg.model.cross_entropy_loss_fusion is True
    assert cfg.model.cross_entropy_fusion_impl == "native"


def test_qwen35_vl_cuda_graph_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that CUDA graph settings are correctly configured."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == "full"
    assert cfg.model.cuda_graph_warmup_steps == 3


def test_qwen35_vl_transformer_impl(monkeypatch: pytest.MonkeyPatch):
    """Test that transformer implementation is set correctly."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.transformer_impl == "transformer_engine"


def test_qwen35_vl_memory_saving_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that memory saving settings are disabled by default."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_800m_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules is None


# =============================================================================
# Qwen3.5-VL Pretrain Mock Config Tests
# =============================================================================


@pytest.mark.parametrize("recipe_func", _QWEN35_VL_PRETRAIN_MOCK_FUNCS)
def test_each_qwen35_vl_pretrain_mock_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen3.5-VL pretrain mock recipe builds a valid ConfigContainer."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == cfg.dataset.hf_processor_path
    assert cfg.tokenizer.use_tokenizer_vocab_size is True
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    expected_language_freeze = recipe_func is not _qwen35_vl_module.qwen35_vl_27b_pretrain_mock_config
    assert cfg.model.freeze_language_model is expected_language_freeze
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False

    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _QWEN35_VL_PRETRAIN_MOCK_FUNCS)
def test_qwen35_vl_pretrain_mock_uses_mock_dataset(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock configs use the declarative mock VLM config."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    from megatron.bridge.data.builders import MockVLMSFTDatasetConfig

    assert isinstance(cfg.dataset, MockVLMSFTDatasetConfig)


def test_qwen35_vl_9b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 9B pretrain mock has correct default parallelism."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_dtype is None
    assert cfg.model.expert_model_parallel_size == 1

    assert cfg.train.train_iters == 300000
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 2
    assert cfg.optimizer.lr == 3e-4
    assert cfg.mixed_precision == "bf16_mixed"


def test_qwen35_vl_27b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 27B pretrain mock has correct default parallelism."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_27b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.pipeline_dtype is not None
    assert cfg.model.expert_model_parallel_size == 1
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False


def test_qwen35_vl_35b_a3b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 35B-A3B pretrain mock has correct MoE parallelism."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_35b_a3b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.pipeline_dtype is not None
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 2


class _FakeModelCfgWithoutExpertTensorParallel(_FakeModelCfg):
    """Fake model configuration that leaves expert tensor parallelism unset."""

    def __init__(self):
        super().__init__()
        # Megatron-Core resolves an unset value to the dense tensor parallel size.
        self.expert_tensor_parallel_size = None


class _FakeAutoBridgeWithoutExpertTensorParallel:
    """Fake AutoBridge whose provider leaves expert tensor parallelism unset."""

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeAutoBridgeWithoutExpertTensorParallel()

    def get_model_config(self):
        return _FakeModelCfgWithoutExpertTensorParallel()


@pytest.mark.parametrize(
    ("recipe_func", "world_size"),
    [
        (_qwen35_vl_h100_module.qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config, 8),
        (_qwen35_vl_h100_module.qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_mock_config, 128),
        (_qwen35_vl_h100_module.qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config, 512),
    ],
)
def test_qwen35_vl_moe_pretrain_mock_grids_divide_their_named_world_size(
    recipe_func: Callable,
    world_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """MoE pretrain mock recipes own dense and expert grids that divide the world size they name."""
    patch_recipe_module_global(
        monkeypatch,
        _qwen35_vl_h100_module,
        "AutoBridge",
        _FakeAutoBridgeWithoutExpertTensorParallel,
    )

    cfg = recipe_func()

    assert cfg.model.expert_tensor_parallel_size == 1
    dense_grid = (
        cfg.model.tensor_model_parallel_size * cfg.model.context_parallel_size * cfg.model.pipeline_model_parallel_size
    )
    expert_grid = (
        cfg.model.expert_tensor_parallel_size
        * cfg.model.expert_model_parallel_size
        * cfg.model.pipeline_model_parallel_size
    )
    assert world_size % dense_grid == 0, f"dense grid {dense_grid} does not divide {world_size}"
    assert world_size % expert_grid == 0, f"expert grid {expert_grid} does not divide {world_size}"


def test_qwen35_vl_122b_a10b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 122B-A10B pretrain mock has correct large MoE parallelism."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_122b_a10b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.model.pipeline_dtype is not None
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.sequence_parallel is True
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False


def test_qwen35_vl_397b_a17b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 397B-A17B pretrain mock has correct large MoE parallelism."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 16
    assert cfg.model.pipeline_dtype is not None
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.sequence_parallel is True
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False


def test_qwen35_vl_pretrain_mock_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock DDP config is correctly set."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config()

    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_qwen35_vl_pretrain_mock_overrides_after_instantiation(monkeypatch: pytest.MonkeyPatch):
    """Test that callers can override fixed pretrain configs after instantiation."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config()
    cfg.train.train_iters = 500
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.optimizer.lr = 1e-5

    _assert_basic_config(cfg)

    assert cfg.train.train_iters == 500
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.optimizer.lr == 1e-5


def test_qwen35_vl_pretrain_mock_checkpoint_config(monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock checkpoint config is correctly set."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config()

    assert cfg.checkpoint.ckpt_format == "torch_dist"
    assert cfg.checkpoint.save_interval == 500
    assert cfg.checkpoint.fully_parallel_save is True


def test_qwen35_vl_pretrain_mock_rng_seed(monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock RNG seed is set."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config()

    assert cfg.rng.seed == 1234


class _FakeModelCfgNoGdnFusion(_FakeModelCfg):
    """Model config from a Megatron-Core predating gdn_pre_gated_delta_rule_fusion."""

    def __init__(self):
        super().__init__()
        del self.gdn_pre_gated_delta_rule_fusion


class _FakeAutoBridgeNoGdnFusion(_FakeAutoBridge):
    """AutoBridge yielding a provider without the GDN fusion field."""

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeAutoBridgeNoGdnFusion()

    def get_model_config(self):
        return _FakeModelCfgNoGdnFusion()


@pytest.mark.parametrize(
    "recipe_name",
    [
        "qwen35_vl_9b_pretrain_mock_config",
        "qwen35_vl_35b_a3b_pretrain_config",
        "qwen35_vl_397b_a17b_pretrain_mock_config",
        # SFT too: the default lives at the shared construction point, so it is
        # not confined to pretraining recipes.
        "qwen35_vl_397b_a17b_sft_config",
    ],
)
def test_qwen35_vl_library_recipes_enable_gdn_conv_fusion(monkeypatch: pytest.MonkeyPatch, recipe_name: str):
    """Qwen3.5-VL library recipes fuse the GatedDeltaNet pre-gated-delta-rule path.

    Every Qwen3.5-VL recipe in the module applies ``_enable_gdn_conv_fusion`` right
    after building its model config, so it covers pretrain, SFT and PEFT alike rather
    than only the perf recipes.
    """
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = getattr(_qwen35_vl_module, recipe_name)()

    assert cfg.model.gdn_pre_gated_delta_rule_fusion is True


def test_qwen35_vl_gdn_conv_fusion_skipped_on_older_core(monkeypatch: pytest.MonkeyPatch):
    """On a core without the field the recipe must not invent the attribute.

    Assigning an unknown field would not raise -- it would silently create an
    unused attribute, leaving the recipe looking enabled while running unfused.
    """
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridgeNoGdnFusion)

    cfg = _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config()

    assert not hasattr(cfg.model, "gdn_pre_gated_delta_rule_fusion")


@pytest.mark.parametrize(
    "recipe_func",
    [
        _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_bf16_config,
        _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_fp8mx_config,
    ],
)
def test_gb200_long_context_validates_with_real_model_config(monkeypatch, recipe_func):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(major=10, name="NVIDIA GB200")
    )

    config_module = importlib.import_module("megatron.bridge.training.config")
    monkeypatch.setattr(config_module, "get_world_size_safe", lambda: 32)
    monkeypatch.setattr(_FakeAutoBridge, "get_model_config", lambda self: _real_moe_model_config())
    patch_recipe_module_global(monkeypatch, _qwen35_vl_gb200_module, "AutoBridge", _FakeAutoBridge)
    config = recipe_func()

    config.dataset.path = "prepared-clevr2-energon"
    config.dataset.validate()
    # Mirror runtime_config_update: mixed precision sets the parameter dtype before validation.
    config.mixed_precision.finalize()
    config.mixed_precision.setup(config.model, config.optimizer, config.ddp)
    config.validate()
    config.comm_overlap.finalize()
    config.comm_overlap.setup(config.model, config.optimizer, config.ddp)
    assert config.dataset.pad_to_max_length is True  # Derived for HybridEP.
    assert {
        "gdn_kernel_backend",
        "gdn_pre_gated_delta_rule_fusion",
        "gdn_gated_output_norm_fusion",
    } <= {field.name for field in fields(config.model.transformer)}
    assert config.model.gdn_kernel_backend == "transformer_engine"
    assert config.ddp.overlap_grad_reduce is False
    assert config.ddp.overlap_param_gather is False
    assert config.optimizer.overlap_param_gather is False
    assert config.optimizer.overlap_param_gather_with_optimizer_step is False
    assert config.comm_overlap.overlap_grad_reduce is False
    assert config.comm_overlap.overlap_param_gather is False
    assert config.comm_overlap.overlap_param_gather_with_optimizer_step is False
    assert config.model.overlap_moe_expert_parallel_comm is False
    assert config.model.delay_wgrad_compute is False

    data_parallel_size = 32 // (
        config.model.tensor_model_parallel_size
        * config.model.pipeline_model_parallel_size
        * config.model.context_parallel_size
    )
    assert data_parallel_size == 4
    assert config.train.global_batch_size % (data_parallel_size * config.train.micro_batch_size) == 0
    config.dataset.enable_in_batch_packing = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.dataset.validate()


def test_gb200_long_context_pins_model_and_processor_together(monkeypatch):
    calls = []

    def from_hf_pretrained(hf_path, **kwargs):
        calls.append((hf_path, kwargs))
        return _FakeAutoBridge()

    monkeypatch.setattr(_FakeAutoBridge, "from_hf_pretrained", from_hf_pretrained)
    patch_recipe_module_global(monkeypatch, _qwen35_vl_gb200_module, "AutoBridge", _FakeAutoBridge)
    recipe = _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_bf16_config
    cfg = recipe()
    encoder = cfg.dataset.task_encoder
    assert calls == [(encoder.hf_processor_path, {"revision": encoder.hf_processor_revision})]


# ---------------------------------------------------------------------------
# Builder-backed model config
# ---------------------------------------------------------------------------


class _BuilderOnlyAutoBridge:
    """AutoBridge returning a strict Qwen3.5-VL builder config and rejecting the provider API."""

    def __init__(self, hf_path: str):
        self._hf_path = hf_path

    @staticmethod
    def from_hf_pretrained(hf_path: str, *, revision: str | None = None):
        return _BuilderOnlyAutoBridge(hf_path)

    def get_model_config(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

        from megatron.bridge import AutoBridge

        # MoE checkpoints are named by active parameters, e.g. Qwen3.5-35B-A3B.
        if "-A" in self._hf_path:
            hf_config = Qwen3_5MoeConfig(architectures=["Qwen3_5MoeForConditionalGeneration"])
        else:
            hf_config = Qwen3_5Config(architectures=["Qwen3_5ForConditionalGeneration"])
        return AutoBridge.from_hf_config(hf_config).get_model_config()

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("Qwen3.5-VL recipes must use get_model_config()")


_ALL_QWEN35_VL_RECIPE_FUNCS = list(
    dict.fromkeys(
        _QWEN35_VL_PRETRAIN_MOCK_FUNCS
        + _QWEN35_VL_H100_PRETRAIN_MOCK_FUNCS
        + _QWEN35_VL_SFT_FUNCS
        + _QWEN35_VL_H100_SFT_FUNCS
        + _QWEN35_VL_PEFT_FUNCS
        + _QWEN35_VL_H100_PEFT_FUNCS
        + _QWEN35_VL_GB200_FUNCS
        + _QWEN35_VL_GB300_FUNCS
    )
)


@pytest.mark.parametrize("recipe_func", _ALL_QWEN35_VL_RECIPE_FUNCS, ids=lambda func: func.__name__)
def test_qwen35_vl_recipes_build_strict_model_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Every recipe builds the builder-backed config and assigns only fields it declares."""
    from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig

    monkeypatch.setattr(_qwen35_vl_h100_module, "AutoBridge", _BuilderOnlyAutoBridge)
    monkeypatch.setattr(_qwen35_vl_gb200_module, "AutoBridge", _BuilderOnlyAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)
    assert isinstance(cfg.model, Qwen35VLModelConfig)
    assert cfg.model.position_embedding_type == "mrope"
    assert cfg.model.vision_config is not None
    assert cfg.model.experimental_attention_variant == "gdn"
