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
# - Parametrize over all exported Qwen recipe functions in `megatron.bridge.recipes.qwen`.
# - For each recipe, monkeypatch `AutoBridge` with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection: pretrain recipes honor `use_null_tokenizer`, sft/peft recipes always use HF tokenizer.
# - Sanity-check parallelism fields and finetuning-specific requirements.
#

import importlib
from typing import Callable

import pytest

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_qwen_module = importlib.import_module("megatron.bridge.recipes.qwen")
_QWEN_RECIPE_FUNCS = [
    getattr(_qwen_module, name)
    for name in getattr(_qwen_module, "__all__", [])
    if callable(getattr(_qwen_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    All configs (pretrain, sft, peft) use the new parameterless API.
    For peft configs, only peft_scheme can be passed as a parameter.
    """
    # All configs now use parameterless API (or peft_scheme only for peft)
    return {}


class _FakeModelCfg:
    # Minimal provider to accept attribute assignments used in recipes

    def __init__(self):
        self.cross_entropy_fusion_impl = "native"
        self.context_parallel_size = 1

    def finalize(self):
        # qwen3 recipe may call finalize(); make it a no-op
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        # Ignore hf_path; return a bridge that yields a fake provider
        return _FakeBridge()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    # Required top-level sections
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    # A few critical fields
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1

    if hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _QWEN_RECIPE_FUNCS)
def test_each_qwen_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    # Always patch AutoBridge in qwen3_moe (where base configs actually call it)
    qwen3_moe_mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, qwen3_moe_mod, "AutoBridge", _FakeBridge)
    # Also patch in the recipe function's own module if it directly references AutoBridge
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    if hasattr(mod, "AutoBridge"):
        patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    # qwen3_next PEFT is intentionally not implemented.
    if recipe_func.__name__ in {
        "qwen3_next_80b_a3b_peft_config",
        "qwen3_next_80b_a3b_peft_1gpu_h100_bf16_config",
    }:
        with pytest.raises(NotImplementedError):
            recipe_func(**overrides)
        return

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer is properly configured
    recipe_name = recipe_func.__name__.lower()
    is_sft_or_peft = "sft" in recipe_name or "peft" in recipe_name
    if is_sft_or_peft:
        # SFT and PEFT recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        # depending on the model (qwen2/qwen25 use NullTokenizer, qwen3 uses HuggingFaceTokenizer)
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    # Parallelism and shaping
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    if "qwen3" in recipe_name and "pretrain" in recipe_name and "next" not in recipe_name:
        assert cfg.model.cross_entropy_fusion_impl == "te"

    # SFT and PEFT-specific assertions
    if is_sft_or_peft:
        # New parameterless API - pretrained_checkpoint is set by user after config creation
        # Just verify the checkpoint config exists
        assert cfg.checkpoint is not None
        # Should have PEFT config (or None if SFT)
        assert hasattr(cfg, "peft")  # peft field should exist
        # Dataset should be configured (SQuAD by default)
        assert cfg.dataset is not None


# Qwen3 MoE SFT and PEFT-specific tests
_QWEN3_MOE_SFT_FUNCS = [
    getattr(_qwen_module, name)
    for name in [
        "qwen3_30b_a3b_sft_config",
        "qwen3_235b_a22b_sft_config",
    ]
    if callable(getattr(_qwen_module, name, None))
]

_QWEN3_MOE_PEFT_FUNCS = [
    getattr(_qwen_module, name)
    for name in [
        "qwen3_30b_a3b_peft_config",
        "qwen3_235b_a22b_peft_config",
    ]
    if callable(getattr(_qwen_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _QWEN3_MOE_SFT_FUNCS)
def test_qwen3_moe_sft_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that full SFT configurations are correctly applied for Qwen3 MoE models."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Full SFT should not have PEFT config
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _QWEN3_MOE_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_qwen3_moe_peft_config(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configurations are correctly applied for Qwen3 MoE models."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)

    # PEFT config should be present
    assert cfg.peft is not None


def test_qwen3_30b_a3b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, 30B-A3B should use TP=4, PP=1, EP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]


def test_qwen3_30b_a3b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B DoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, 30B-A3B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]


def test_qwen3_30b_a3b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 30B-A3B should use TP=4, PP=2, EP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None


def test_qwen3_30b_a3b_pretrain_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that the generic 30B-A3B pretrain recipe uses the verified 16-GPU topology."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_pretrain_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.moe_hybridep_num_sms == 32
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["moe_router", "moe_preprocess"]
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.comm_overlap.tp_comm_overlap is True


def test_qwen3_30b_a3b_bf16_perf_recipe_uses_default_functional_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the H100 BF16 perf recipe only adds benchmark-specific overrides."""
    from megatron.bridge.perf_recipes.qwen.h100.qwen3_moe import (
        qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config,
    )
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    default_cfg = qwen3_30b_a3b_pretrain_config()
    perf_cfg = qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config()

    assert perf_cfg.model.tensor_model_parallel_size == default_cfg.model.tensor_model_parallel_size
    assert perf_cfg.model.pipeline_model_parallel_size == default_cfg.model.pipeline_model_parallel_size
    assert perf_cfg.model.expert_model_parallel_size == default_cfg.model.expert_model_parallel_size
    assert perf_cfg.model.sequence_parallel == default_cfg.model.sequence_parallel
    assert perf_cfg.train.global_batch_size == default_cfg.train.global_batch_size
    assert perf_cfg.train.micro_batch_size == default_cfg.train.micro_batch_size
    assert perf_cfg.model.moe_flex_dispatcher_backend == default_cfg.model.moe_flex_dispatcher_backend
    assert perf_cfg.model.moe_token_dispatcher_type == default_cfg.model.moe_token_dispatcher_type
    assert perf_cfg.model.cuda_graph_impl == default_cfg.model.cuda_graph_impl
    assert perf_cfg.model.cuda_graph_scope == default_cfg.model.cuda_graph_scope
    assert perf_cfg.comm_overlap.tp_comm_overlap == default_cfg.comm_overlap.tp_comm_overlap
    assert perf_cfg.model.moe_router_force_load_balancing is True
    assert default_cfg.model.moe_router_force_load_balancing is False


def test_qwen3_30b_a3b_perf_base_remains_legacy_8gpu_recipe():
    """Keep non-H100-BF16 perf recipes isolated from the new generic default."""
    from megatron.bridge.perf_recipes.qwen.common import qwen3_30b_a3b_pretrain_config as perf_base
    from megatron.bridge.recipes.qwen.h100.qwen3_moe import (
        qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config as legacy_base,
    )

    assert perf_base is legacy_base


def test_qwen3_30b_a3b_h100_fp8_perf_recipe_keeps_cuda_graphs_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that changing the generic default does not alter the H100 FP8 recipe."""
    from megatron.bridge.perf_recipes.qwen.h100.qwen3_moe import (
        qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config,
    )

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config()

    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == []


def test_qwen3_235b_a22b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_235b_a22b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_235b_a22b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, 235B-A22B should use TP=4, PP=4, EP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]


def test_qwen3_235b_a22b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B DoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_235b_a22b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_235b_a22b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, 235B-A22B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]


def test_qwen3_235b_a22b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_235b_a22b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_235b_a22b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 235B-A22B should use TP=4, PP=16, EP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 16
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True

    assert cfg.peft is None
