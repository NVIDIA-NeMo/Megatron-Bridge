"""Dependency-free coverage of opt-in performance-run stack dumps."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit
MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts/performance/utils/debug_tracebacks.py"


@pytest.fixture
def debug_tracebacks(monkeypatch):
    spec = importlib.util.spec_from_file_location("debug_tracebacks", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("MBRIDGE_DEBUG_TRACEBACK_SECONDS", "MBRIDGE_DEBUG_TRACEBACK_DIR", "RANK", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", Mock())
    monkeypatch.setattr(module.faulthandler, "cancel_dump_traceback_later", Mock())
    return module


def test_disabled_has_no_side_effects(debug_tracebacks):
    with debug_tracebacks.periodic_debug_tracebacks():
        pass
    debug_tracebacks.faulthandler.dump_traceback_later.assert_not_called()
    debug_tracebacks.faulthandler.cancel_dump_traceback_later.assert_not_called()


@pytest.mark.parametrize("rank_var", ["RANK", "SLURM_PROCID"])
@pytest.mark.parametrize("raises", [False, True])
def test_rank_dump_and_cleanup(debug_tracebacks, monkeypatch, tmp_path, rank_var, raises):
    monkeypatch.setenv("MBRIDGE_DEBUG_TRACEBACK_SECONDS", "120")
    monkeypatch.setenv("MBRIDGE_DEBUG_TRACEBACK_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv(rank_var, "33")
    if rank_var == "RANK":
        monkeypatch.setenv("SLURM_PROCID", "0")
    output = None

    def cancel():
        assert not output.closed

    debug_tracebacks.faulthandler.cancel_dump_traceback_later.side_effect = cancel
    try:
        with debug_tracebacks.periodic_debug_tracebacks():
            args, kwargs = debug_tracebacks.faulthandler.dump_traceback_later.call_args
            assert args == (120,)
            assert kwargs["repeat"] is True
            output = kwargs["file"]
            assert Path(output.name) == tmp_path / "profiles/python_tracebacks_rank-33.log"
            assert not output.closed
            if raises:
                raise RuntimeError("training failed")
    except RuntimeError as error:
        assert raises
        assert str(error) == "training failed"
    assert output.closed
    debug_tracebacks.faulthandler.cancel_dump_traceback_later.assert_called_once_with()


@pytest.mark.parametrize("interval", ["0", "-1", "inf", "abc", "0.5"])
def test_invalid_interval(debug_tracebacks, monkeypatch, interval):
    monkeypatch.setenv("MBRIDGE_DEBUG_TRACEBACK_SECONDS", interval)
    with pytest.raises(ValueError), debug_tracebacks.periodic_debug_tracebacks():
        pytest.fail("Invalid interval accepted")
    debug_tracebacks.faulthandler.dump_traceback_later.assert_not_called()


def test_requires_directory(debug_tracebacks, monkeypatch):
    monkeypatch.setenv("MBRIDGE_DEBUG_TRACEBACK_SECONDS", "120")
    with pytest.raises(ValueError, match="DIR is required"), debug_tracebacks.periodic_debug_tracebacks():
        pytest.fail("Missing directory accepted")


@pytest.mark.parametrize("rank", [None, "-1", "../../x", "not-a-rank"])
def test_requires_global_rank(debug_tracebacks, monkeypatch, tmp_path, rank):
    monkeypatch.setenv("MBRIDGE_DEBUG_TRACEBACK_SECONDS", "120")
    monkeypatch.setenv("MBRIDGE_DEBUG_TRACEBACK_DIR", str(tmp_path))
    if rank is not None:
        monkeypatch.setenv("RANK", rank)
    with pytest.raises(ValueError, match="RANK or SLURM_PROCID"), debug_tracebacks.periodic_debug_tracebacks():
        pytest.fail("Invalid rank accepted")
