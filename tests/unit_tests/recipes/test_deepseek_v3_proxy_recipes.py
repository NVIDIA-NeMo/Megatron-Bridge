"""Guard the 64-GPU debugging proxies against drift from their 256-GPU parents."""

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest

from megatron.bridge.perf_recipes import deepseek
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("precision", ["fp8mx", "nvfp4"])
def test_proxy_preserves_parent_except_global_batch(precision: str) -> None:
    parent_fn: Callable[[], ConfigContainer] = getattr(
        deepseek, f"deepseek_v3_pretrain_256gpu_vr200_{precision}_config"
    )
    proxy_fn: Callable[[], ConfigContainer] = getattr(
        deepseek, f"deepseek_v3_pretrain_64gpu_vr200_{precision}_proxy_config"
    )
    parent = parent_fn()
    proxy = proxy_fn()

    assert parent.train.global_batch_size == 4096
    assert proxy.train.global_batch_size == 1024
    assert parent.train.micro_batch_size == proxy.train.micro_batch_size == 1
    assert parent.train.train_iters == proxy.train.train_iters == 50
    assert vars(proxy.model) == vars(parent.model)
    parent.train.global_batch_size = proxy.train.global_batch_size
    for field in fields(parent):
        if field.name != "model":
            assert getattr(proxy, field.name) == getattr(parent, field.name), field.name

    model = proxy.model
    assert (model.tensor_model_parallel_size, model.context_parallel_size) == (1, 1)
    assert (model.pipeline_model_parallel_size, model.virtual_pipeline_model_parallel_size) == (2, 8)
    assert (model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (32, 1)
    for world, batch, expected_dp, expected_edp in ((64, 1024, 32, 1), (256, 4096, 128, 4)):
        dense_mesh = (
            model.tensor_model_parallel_size * model.context_parallel_size * model.pipeline_model_parallel_size
        )
        expert_mesh = (
            model.expert_tensor_parallel_size * model.expert_model_parallel_size * model.pipeline_model_parallel_size
        )
        assert world % dense_mesh == world % expert_mesh == 0
        assert world // dense_mesh == expected_dp
        assert world // expert_mesh == expected_edp
        assert batch // (world // dense_mesh * proxy.train.micro_batch_size) == 32

    assert model.cuda_graph_impl == "full_iteration"
    assert proxy.comm_overlap is not None
    assert proxy.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert proxy.comm_overlap.delay_wgrad_compute is True
    assert proxy.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert model.moe_flex_dispatcher_backend == "hybridep"
    assert model.mtp_num_layers == 1
    assert model.recompute_modules == []


@pytest.mark.parametrize("precision", ["fp8_mx", "nvfp4"])
def test_proxy_is_available_through_performance_selector(monkeypatch: pytest.MonkeyPatch, precision: str) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts" / "performance"))
    from utils.utils import get_perf_recipe_by_name, list_available_config_variants

    cfg = get_perf_recipe_by_name("deepseek_v3", "pretrain", 64, "vr200", precision, config_variant="proxy")
    assert cfg.train.global_batch_size == 1024
    assert "proxy" in list_available_config_variants(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="vr200",
        compute_dtype=precision,
        task="pretrain",
    )
