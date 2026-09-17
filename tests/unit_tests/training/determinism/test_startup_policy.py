"""CPU contracts for Bridge's early startup adapter and the real MCore policy.

Run with ``--confcutdir=tests/unit_tests/training/determinism`` to avoid the GPU
dependency stack in the parent conftest. MCORE_SOURCE_ROOT may select a Core
checkout while testing this dependent change before Bridge's submodule update.

Only the exercised Bridge definitions are loaded, without their GPU-only module
imports. Public imports, real config classes and distributed initialization are
covered separately by test_determinism_startup.py in the functional suite.
"""

import ast
import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NoReturn
from unittest.mock import Mock, patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
pytestmark = pytest.mark.unit


def _load_definitions(path: str, names: list[str], namespace: dict[str, object]) -> None:
    """Execute unmodified function bodies without importing GPU dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    definitions = []
    for name in names:
        nodes = tree.body
        if "." in name:
            owner, name = name.split(".")
            nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
        definitions.append(next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name))
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    namespace: dict[str, object] = {"os": os, "torch": torch}
    _load_definitions(
        "src/megatron/bridge/training/config.py",
        [
            "apply_environment_variables",
            "ConfigContainer._validate_and_apply_deterministic_mode",
            "ConfigContainer.validate",
            "runtime_config_update",
            "megatron_mimo_runtime_config_update",
        ],
        namespace,
    )
    _load_definitions("src/megatron/bridge/training/initialize.py", ["initialize_megatron"], namespace)
    _load_definitions(
        "src/megatron/bridge/recipes/utils/determinism_utils.py", ["apply_determinism_overrides"], namespace
    )

    class Config:
        """Stop at the first unrelated validation operation, before finalization."""

        _validate_and_apply_deterministic_mode = namespace["_validate_and_apply_deterministic_mode"]
        validate = namespace["validate"]

        def __init__(self) -> None:
            self.model = SimpleNamespace(
                deterministic_mode=True,
                cross_entropy_loss_fusion=False,
                tp_comm_overlap=False,
                moe_router_fusion=False,
                moe_router_aux_loss_fusion=None,
            )
            self.env_vars: dict[str, str | int | float | bool] = {}
            self.comm_overlap = None
            self.mixed_precision = None
            self.optimizer = self.ddp = None
            self.set_data_parallel_size = Mock()
            self._disable_native_energon_packing_moe_overlap = Mock()

        @property
        def train(self) -> NoReturn:
            raise StopAfterPolicy("Reached config validation")

    namespace["Config"] = Config
    # Isolate environment and process state even if the parent GPU suite ran first.
    monkeypatch.setattr(
        os,
        "environ",
        {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("NCCL_", "NVTE_", "CUBLAS_", "MAMBA_", "CAUSAL_CONV1D_", "TRITON_"))
        },
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return SimpleNamespace(**namespace)


class StopAfterPolicy(Exception):
    """Intentional boundary before unrelated training setup or CUDA probing."""


@pytest.fixture
def policy(adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    root = Path(os.environ.get("MCORE_SOURCE_ROOT", ROOT / "3rdparty/Megatron-LM"))
    path = root / "megatron/core/determinism.py"
    assert path.is_file(), "This integration requires an MCore checkout with the shared startup API."
    spec = importlib.util.spec_from_file_location("megatron.core.determinism", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, spec.name, module)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    yield module
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def test_defaults_and_strict_torch_policy(adapter: SimpleNamespace, policy: ModuleType) -> None:
    cfg = adapter.Config()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = True
    rng = torch.get_rng_state().clone()
    cfg._validate_and_apply_deterministic_mode()
    assert os.environ["NCCL_ALGO"] == "Ring"
    assert os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic
    assert torch.equal(rng, torch.get_rng_state())


def test_recipe_defaults_preserve_launcher_precedence(
    adapter: SimpleNamespace, policy: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = adapter.Config()
    cfg.env_vars = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0}
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    cfg._validate_and_apply_deterministic_mode()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"


@pytest.mark.parametrize(
    "name,value",
    [
        ("NCCL_ALGO", "Tree"),
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"),
        ("CUBLAS_WORKSPACE_CONFIG", "invalid"),
        ("TRITON_CACHE_AUTOTUNING", "1"),
        ("MAMBA_DETERMINISTIC", "0"),
    ],
)
def test_conflicting_launcher_fails_before_finalization(
    adapter: SimpleNamespace, policy: ModuleType, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    cfg = adapter.Config()
    adapter.apply_determinism_overrides(cfg)
    monkeypatch.setenv(name, value)
    with pytest.raises(AssertionError, match=name):
        cfg.validate()
    assert os.environ[name] == value


@pytest.mark.parametrize("name", ["cross_entropy_loss_fusion", "tp_comm_overlap", "moe_router_aux_loss_fusion"])
def test_unsupported_option_fails_before_finalization(adapter: SimpleNamespace, policy: ModuleType, name: str) -> None:
    cfg = adapter.Config()
    assert hasattr(cfg.model, name)
    setattr(cfg.model, name, True)
    with pytest.raises(AssertionError, match=name):
        cfg.validate()


@pytest.mark.parametrize(
    "entrypoint", ["validate", "runtime_config_update", "megatron_mimo_runtime_config_update", "initialize_megatron"]
)
def test_policy_precedes_validation_and_cuda_probe(
    adapter: SimpleNamespace, policy: ModuleType, monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    cfg = adapter.Config()
    monkeypatch.setattr(torch.cuda, "is_available", Mock(side_effect=StopAfterPolicy("Reached CUDA probe")))
    with pytest.raises(StopAfterPolicy):
        if entrypoint == "validate":
            cfg.validate()
        else:
            getattr(adapter, entrypoint)(cfg)
    assert os.environ["NCCL_ALGO"] == "Ring"
    assert torch.are_deterministic_algorithms_enabled()
    assert policy._configured_pid == os.getpid()


def test_resolved_comm_overlap_is_validated(adapter: SimpleNamespace, policy: ModuleType) -> None:
    cfg = adapter.Config()

    def enable_overlap(model: SimpleNamespace, optimizer: object, ddp: object) -> None:
        model.tp_comm_overlap = True

    cfg.comm_overlap = SimpleNamespace(finalize=Mock(), setup=enable_overlap)
    with pytest.raises(AssertionError, match="tp_comm_overlap"):
        adapter.runtime_config_update(cfg)


def test_missing_api_is_clear_and_default_mode_does_not_require_it(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "megatron.core.determinism", None)
    cfg = adapter.Config()
    cfg.model.deterministic_mode = False
    cfg._validate_and_apply_deterministic_mode()
    assert "NCCL_ALGO" not in os.environ
    cfg.model.deterministic_mode = True
    with pytest.raises(RuntimeError, match="requires Megatron Core's"):
        cfg._validate_and_apply_deterministic_mode()


def test_unrelated_import_error_is_not_hidden(adapter: SimpleNamespace) -> None:
    error = ModuleNotFoundError("Missing backend dependency", name="transformer_engine")
    with (
        pytest.raises(ModuleNotFoundError, match="Missing backend dependency"),
        patch("builtins.__import__", side_effect=error),
    ):
        adapter.Config()._validate_and_apply_deterministic_mode()


def test_first_late_call_is_rejected(
    adapter: SimpleNamespace, policy: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA"):
        adapter.Config()._validate_and_apply_deterministic_mode()


@pytest.mark.parametrize("drift", ["workspace", "torch", "config"])
def test_same_policy_can_repeat_but_late_drift_fails(
    adapter: SimpleNamespace, policy: ModuleType, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    cfg = adapter.Config()
    cfg._validate_and_apply_deterministic_mode()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    cfg._validate_and_apply_deterministic_mode()
    if drift == "workspace":
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    elif drift == "torch":
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cfg.model.cross_entropy_loss_fusion = True
    with pytest.raises((AssertionError, RuntimeError)):
        cfg._validate_and_apply_deterministic_mode()


@pytest.mark.parametrize("with_comm_overlap", [False, True])
def test_recipe_overrides_disable_both_overlap_sources_and_only_aux_fusion(
    adapter: SimpleNamespace, with_comm_overlap: bool
) -> None:
    cfg = adapter.Config()
    cfg.model.deterministic_mode = False
    cfg.model.cross_entropy_loss_fusion = cfg.model.tp_comm_overlap = cfg.model.moe_router_fusion = True
    if with_comm_overlap:
        cfg.comm_overlap = SimpleNamespace(tp_comm_overlap=True)
    adapter.apply_determinism_overrides(cfg)
    adapter.apply_determinism_overrides(cfg)
    assert cfg.model.deterministic_mode
    assert not cfg.model.cross_entropy_loss_fusion
    assert not cfg.model.tp_comm_overlap
    assert cfg.model.moe_router_fusion
    assert cfg.model.moe_router_aux_loss_fusion is False
    assert cfg.comm_overlap is None or cfg.comm_overlap.tp_comm_overlap is False
    assert "NCCL_ALGO" not in os.environ  # Recipe construction does not initialize policy.


def test_recipe_override_supports_core_without_separate_aux_option(adapter: SimpleNamespace) -> None:
    cfg = adapter.Config()
    del cfg.model.moe_router_aux_loss_fusion
    cfg.model.moe_router_fusion = True
    adapter.apply_determinism_overrides(cfg)
    assert cfg.model.moe_router_fusion is False
    assert not hasattr(cfg.model, "moe_router_aux_loss_fusion")
