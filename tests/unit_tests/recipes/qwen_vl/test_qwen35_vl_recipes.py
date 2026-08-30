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
from typing import Callable

import pytest
import torch

from megatron.bridge.peft.dora import DoRA
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_qwen35_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen35_vl")
_qwen35_vl_h100_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.h100.qwen35_vl")
_qwen35_vl_gb200_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.gb200.qwen35_vl")

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
    _qwen35_vl_h100_module.qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_functional_config,
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
    _qwen35_vl_gb200_module.qwen35_vl_35b_a3b_peft_8gpu_gb200_bf16_functional_config,
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

    def finalize(self):
        return None


class _FakeAutoBridge:
    """Fake AutoBridge for testing."""

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
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
    + _QWEN35_VL_GB200_FUNCS,
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


def test_qwen35_vl_35b_a3b_pretrain_16gpu_h100_defaults(monkeypatch: pytest.MonkeyPatch):
    """The 16-H100 library pretrain recipe should own the measured execution policy."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_h100_module.qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_functional_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.num_layers_in_first_pipeline_stage == 16
    assert cfg.model.num_layers_in_last_pipeline_stage == 24
    assert cfg.model.expert_model_parallel_size == 8
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
    """122B-A10B SFT should have correct default parallelism and learning rate."""
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
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.recompute_granularity == "full"


def test_qwen35_vl_122b_a10b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """122B-A10B PEFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_122b_a10b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.pipeline_dtype is None
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 2e-4


# ---------------------------------------------------------------------------
# 397B-A17B MoE defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_397b_a17b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """397B-A17B SFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.recompute_granularity == "full"


def test_qwen35_vl_397b_a17b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """397B-A17B PEFT should have correct default parallelism and learning rate."""
    patch_recipe_module_global(monkeypatch, _qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 32
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
