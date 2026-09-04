# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Focused tests for deterministic VLM generation configuration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


_SCRIPT_PATH = Path(__file__).parents[4] / "scripts" / "inference" / "vlm_generation.py"
_SPEC = importlib.util.spec_from_file_location("vlm_generation_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
vlm_generation = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_SCRIPT_PATH.parent))
try:
    _SPEC.loader.exec_module(vlm_generation)
finally:
    sys.path.pop(0)


pytestmark = pytest.mark.unit


def test_model_builder_requires_explicit_conversion_migration() -> None:
    legacy_bridge = SimpleNamespace(_model_bridge=SimpleNamespace(MODEL_CONFIG_CLASS=object))
    migrated_bridge = SimpleNamespace(
        _model_bridge=SimpleNamespace(MODEL_CONFIG_CLASS=object, USE_MODEL_CONFIG_FOR_CONVERSION=True)
    )

    assert vlm_generation._uses_model_builder(legacy_bridge) is False
    assert vlm_generation._uses_model_builder(migrated_bridge) is True


def test_deterministic_execution_sets_required_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_ALGO",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    ):
        monkeypatch.delenv(name, raising=False)

    with (
        patch.object(vlm_generation.torch, "use_deterministic_algorithms"),
        patch.object(vlm_generation.torch, "manual_seed"),
        patch.object(vlm_generation.torch.cuda, "manual_seed_all"),
    ):
        vlm_generation._enable_deterministic_execution()

    assert vlm_generation.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert vlm_generation.os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert vlm_generation.os.environ["NCCL_ALGO"] == "Ring"
    assert vlm_generation.os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"


def test_deterministic_execution_rejects_conflicting_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        vlm_generation._enable_deterministic_execution()


def test_deterministic_execution_seeds_and_configures_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setattr(vlm_generation.torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(vlm_generation.torch.backends.cudnn, "deterministic", False)

    with (
        patch.object(vlm_generation.torch, "use_deterministic_algorithms") as use_deterministic,
        patch.object(vlm_generation.torch, "manual_seed") as manual_seed,
        patch.object(vlm_generation.torch.cuda, "manual_seed_all") as cuda_manual_seed_all,
    ):
        vlm_generation._enable_deterministic_execution()

    use_deterministic.assert_called_once_with(True)
    manual_seed.assert_called_once_with(0)
    cuda_manual_seed_all.assert_called_once_with(0)
    assert vlm_generation.torch.backends.cudnn.benchmark is False
    assert vlm_generation.torch.backends.cudnn.deterministic is True


def test_builder_configuration_propagates_deterministic_mode() -> None:
    transformer = SimpleNamespace()
    model_config = SimpleNamespace(transformer=transformer)
    bridge = SimpleNamespace(get_model_config=MagicMock(return_value=model_config))

    result = vlm_generation._configure_builder_model(
        bridge,
        tp=8,
        pp=1,
        ep=1,
        etp=1,
        deterministic=True,
    )

    assert result is model_config
    assert transformer.deterministic_mode is True
    assert transformer.tensor_model_parallel_size == 8
