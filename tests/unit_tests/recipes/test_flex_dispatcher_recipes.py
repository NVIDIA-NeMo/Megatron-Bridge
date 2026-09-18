"""Backend choices must survive recipe inheritance independently of the build host."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.bridge.training.flex_dispatcher_backend import validate_flex_dispatcher_backend
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("package", "factory_name", "backend"),
    [
        ("recipes.qwen", "qwen3_235b_a22b_pretrain_config", "deepep"),
        ("recipes.qwen", "qwen3_30b_a3b_pretrain_config", "hybridep"),
        ("recipes.qwen", "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_235b_a22b_pretrain_64gpu_gb200_bf16_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config", "hybridep"),
    ],
)
@pytest.mark.parametrize("cuda_available", [False, True])
def test_recipe_selects_backend_before_hardware_validation(
    monkeypatch, package, factory_name, backend, cuda_available
):
    factory = getattr(importlib.import_module(f"megatron.bridge.{package}"), factory_name)
    patch_recipe_construction_dependencies(monkeypatch)
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch("torch.cuda.get_device_properties", side_effect=AssertionError("recipe probed the build host")),
    ):
        cfg = factory()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == backend
    assert cfg.model.moe_shared_expert_overlap is False

    with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(major=10, name="NVIDIA GB200")):
        if backend == "deepep":
            # A generic H100 recipe no longer silently becomes alltoall on GB200.
            with pytest.raises(ValueError, match="Current GPU: NVIDIA GB200"):
                validate_flex_dispatcher_backend(cfg.model)
        else:
            # GB200 children can replace DeepEP selected by an H100 parent.
            validate_flex_dispatcher_backend(cfg.model)
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == backend
