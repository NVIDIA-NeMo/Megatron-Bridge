"""NCCL EP dispatch contracts for B200 and B300 Nemotron 3 Super recipes."""

from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.environment import HYBRID_EP_ENV_NAMES
from megatron.bridge.perf_recipes.nemotronh.b200 import nemotronh as b200
from megatron.bridge.perf_recipes.nemotronh.b300 import nemotronh as b300
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("module", (b200, b300), ids=("b200", "b300"))
@pytest.mark.parametrize("precision", ("bf16", "fp8mx", "nvfp4"))
def test_nemotron_super_bseries_nccl_ep_stack(module, precision: str) -> None:
    gpu = module.__name__.split(".")[-2]
    factory: Callable[[], ConfigContainer] = getattr(
        module, f"nemotron_3_super_pretrain_64gpu_{gpu}_{precision}_config"
    )
    cfg = factory()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_use_grouped_tensor is True
    expected_ep = 32 if gpu == "b200" and precision != "nvfp4" else 8
    assert cfg.env_vars["NCCL_EP_HT_EM_PULL_PUSH"] == (0 if expected_ep > 8 else 1)
    assert cfg.env_vars.keys().isdisjoint(HYBRID_EP_ENV_NAMES)
    assert cfg.model.expert_model_parallel_size == expected_ep
    assert cfg.train.global_batch_size == 512
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert ("NCCL_IGNORE_CPU_AFFINITY" in cfg.env_vars) == (gpu == "b300")


@pytest.mark.parametrize("module", (b200, b300), ids=("b200", "b300"))
@pytest.mark.parametrize("precision", ("bf16", "fp8mx", "nvfp4"))
def test_nano_base_keeps_hybridep(module, precision: str) -> None:
    gpu = module.__name__.split(".")[-2]
    cfg = getattr(module, f"nemotron_3_nano_pretrain_8gpu_{gpu}_{precision}_config")()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.env_vars.keys() >= HYBRID_EP_ENV_NAMES
    assert "NCCL_EP_HT_EM_PULL_PUSH" not in cfg.env_vars
