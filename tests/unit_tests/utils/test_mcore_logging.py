"""Logging compatibility tests that do not require importing the GPU stack."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def logging_compat(monkeypatch):
    # Load the real helper against each MCore API shape without importing all
    # Bridge model providers or requiring multiple installed MCore versions.
    core_utils = ModuleType("megatron.core.utils")
    core = ModuleType("megatron.core")
    core.utils = core_utils
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", core_utils)

    helper_path = Path(__file__).resolve().parents[3] / "src/megatron/bridge/utils/mcore_logging.py"
    spec = importlib.util.spec_from_file_location("_bridge_mcore_logging_test", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, core_utils


@pytest.mark.unit
def test_legacy_rank_zero_preserves_mcore_module(logging_compat):
    compat, core_utils = logging_compat
    original_attributes = vars(core_utils).copy()

    compat.set_default_log_ranks({0})

    assert vars(core_utils) == original_attributes


@pytest.mark.unit
@pytest.mark.parametrize("ranks", [{0, 4}, {4}, set()])
def test_legacy_rejects_unsupported_logging_selection(logging_compat, ranks):
    compat, _ = logging_compat

    with pytest.raises(RuntimeError, match="requires Megatron-Core with set_default_log_ranks support"):
        compat.set_default_log_ranks(ranks)


@pytest.mark.unit
@pytest.mark.parametrize("ranks", [{0}, {0, 4}])
def test_current_mcore_receives_requested_ranks(logging_compat, ranks):
    compat, core_utils = logging_compat
    core_utils.set_default_log_ranks = Mock()

    compat.set_default_log_ranks(ranks)

    core_utils.set_default_log_ranks.assert_called_once_with(ranks)


@pytest.mark.unit
def test_current_mcore_errors_are_not_hidden(logging_compat):
    compat, core_utils = logging_compat
    core_utils.set_default_log_ranks = Mock(side_effect=ValueError("invalid rank selection"))

    with pytest.raises(ValueError, match="invalid rank selection"):
        compat.set_default_log_ranks({0, -1})
