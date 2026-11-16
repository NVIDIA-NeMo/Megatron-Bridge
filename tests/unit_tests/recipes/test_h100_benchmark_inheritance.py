"""Regression coverage for library-owned settings inherited by H100 benchmarks."""

import importlib

import pytest
import torch

from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit
_LLAMA = "megatron.bridge.perf_recipes.llama.h100.llama3"
_QWEN = "megatron.bridge.perf_recipes.qwen.h100.qwen3_moe"
_CASES = [
    (
        _LLAMA,
        f"llama3_8b_{task}_{gpus}gpu_h100_{precision}_config",
        "llama3_8b_pretrain_config" if task == "pretrain" else "llama3_8b_sft_config",
    )
    for task, gpus in (("pretrain", 8), ("pretrain", 64), ("sft", 8))
    for precision in ("bf16", "fp8cs")
] + [
    (
        _QWEN,
        f"qwen3_30b_a3b_pretrain_16gpu_h100_{precision}_config",
        "_qwen3_30b_a3b_default_pretrain_config" if precision == "bf16" else "qwen3_30b_a3b_pretrain_config",
    )
    for precision in ("bf16", "fp8cs", "fp8ds")
]


@pytest.fixture(autouse=True)
def _offline_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (_LLAMA, _QWEN, "megatron.bridge.perf_recipes.qwen_vl.h100.qwen35_vl"):
        importlib.import_module(module)
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("module_name,benchmark_name,library_name", _CASES)
def test_library_environment_updates_reach_benchmark(
    module_name: str, benchmark_name: str, library_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library-owned resource setting must not be shadowed by a copied mapping."""
    module = importlib.import_module(module_name)
    library_factory = getattr(module, library_name)

    def library_with_resource_setting() -> ConfigContainer:
        cfg = library_factory()
        assert isinstance(cfg, ConfigContainer)
        cfg.env_vars["NCCL_GRAPH_REGISTER"] = 1
        return cfg

    monkeypatch.setattr(module, library_name, library_with_resource_setting)
    cfg = getattr(module, benchmark_name)()
    assert cfg.env_vars["NCCL_GRAPH_REGISTER"] == 1
    assert cfg.train.train_iters == 50
    assert cfg.ddp.check_for_nan_in_grad is False
    assert cfg.checkpoint.save is None


def test_llama_library_keeps_training_precision_and_checks() -> None:
    from megatron.bridge.recipes.llama.h100.llama3 import llama3_8b_pretrain_2gpu_h100_bf16_config

    cfg = llama3_8b_pretrain_2gpu_h100_bf16_config()
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1
    assert cfg.comm_overlap is None
    assert cfg.train.global_batch_size == 512
    assert cfg.train.train_iters == 1168251
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert getattr(cfg.model, "context_parallel_size") == 2
    assert getattr(cfg.model, "recompute_granularity") is None
    assert cfg.mixed_precision == "bf16_mixed"
    for field in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert getattr(cfg.optimizer, field) is torch.float32


def test_qwen_library_keeps_real_routing_and_optimizer_state() -> None:
    from megatron.bridge.recipes.qwen.h100.qwen3_moe import qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config

    cfg = qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config()
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0
    assert getattr(cfg.model, "moe_router_force_load_balancing") is False
    assert cfg.comm_overlap is not None
    assert not cfg.comm_overlap.overlap_moe_expert_parallel_comm
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    for field in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert getattr(cfg.optimizer, field) is torch.float32


def test_vl_library_environment_is_not_shadowed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("megatron.bridge.perf_recipes.qwen_vl.h100.qwen35_vl")
    library_policy = module._apply_qwen35_vl_35b_a3b_16gpu_h100_execution_config

    def library_with_resource_setting(cfg: ConfigContainer) -> None:
        library_policy(cfg)
        cfg.env_vars["NCCL_GRAPH_REGISTER"] = 1

    monkeypatch.setattr(module, "_apply_qwen35_vl_35b_a3b_16gpu_h100_execution_config", library_with_resource_setting)
    cfg = module.qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config()
    assert cfg.env_vars["NCCL_GRAPH_REGISTER"] == 1
    assert getattr(cfg.model, "moe_router_force_load_balancing") is True
    assert cfg.train.train_iters == 50
