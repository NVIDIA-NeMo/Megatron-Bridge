"""Behavior of library recipes composed with performance benchmark overrides."""

import pytest

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides import (
    build_perf_benchmark_config,
    get_perf_benchmark_recipe,
)
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


def test_unknown_benchmark_is_rejected() -> None:
    assert get_perf_benchmark_recipe("missing_pretrain_config") is None
    with pytest.raises(ValueError, match="Unknown performance benchmark overrides"):
        build_perf_benchmark_config("missing_pretrain_config")


def test_fixed_allocation_benchmark_rejects_allocation_argument() -> None:
    with pytest.raises(TypeError, match="does not accept num_gpus"):
        build_perf_benchmark_config("llama3_8b_pretrain_8gpu_h100_bf16_config", num_gpus=16)


def test_composition_constructs_library_once_and_keeps_instances_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.bridge.training.perf_benchmark_overrides import llama

    patch_recipe_construction_dependencies(monkeypatch)
    name = "llama3_8b_pretrain_8gpu_gb200_bf16_config"
    original = llama.LIBRARY_RECIPES[name]
    assert callable(original)
    constructed: list[ConfigContainer] = []

    def library() -> ConfigContainer:
        cfg = original()
        assert isinstance(cfg, ConfigContainer)
        constructed.append(cfg)
        return cfg

    monkeypatch.setitem(llama.LIBRARY_RECIPES, name, library)
    first = build_perf_benchmark_config(name)
    assert len(constructed) == 1
    assert first is constructed[0]
    first.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = 999
    first.train.train_iters = 999
    second = build_perf_benchmark_config(name)
    assert len(constructed) == 2
    assert second is constructed[1]
    assert first is not second
    assert second.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] != 999
    assert second.train.train_iters == 50
    assert original().train.train_iters != 50
