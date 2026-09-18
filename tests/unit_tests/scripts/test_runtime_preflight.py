"""Dependency-free coverage of runtime import diagnostic gates."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit
MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts/performance/utils/runtime_preflight.py"


@pytest.fixture
def preflight(monkeypatch):
    spec = importlib.util.spec_from_file_location("runtime_preflight", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("MBRIDGE_RUNTIME_PREFLIGHT", "MBRIDGE_RUNTIME_PREFLIGHT_DIR", "RANK", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "_provenance", Mock(return_value={"machine": "test"}))
    monkeypatch.setattr(module, "_run_probe", Mock(side_effect=lambda name: {"module": name, "returncode": 0}))
    return module


def test_disabled_is_noop(preflight):
    preflight.run_from_environment()
    preflight._provenance.assert_not_called()
    preflight._run_probe.assert_not_called()


@pytest.mark.parametrize("mode", ["only", "check"])
@pytest.mark.parametrize("failed", [False, True])
def test_gate_and_reports(preflight, monkeypatch, tmp_path, mode, failed):
    monkeypatch.setenv("MBRIDGE_RUNTIME_PREFLIGHT", mode)
    monkeypatch.setenv("MBRIDGE_RUNTIME_PREFLIGHT_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_PROCID", "7")
    path = tmp_path / "runtime_preflight_rank-7.json"

    def probe(name):
        assert path.exists()
        return {"module": name, "returncode": 1 if failed and name == "transformer_engine.pytorch" else 0}

    preflight._run_probe.side_effect = probe
    if mode == "only" or failed:
        with pytest.raises(SystemExit) as error:
            preflight.run_from_environment()
        assert error.value.code == int(failed)
    else:
        assert preflight.run_from_environment() is None
    report = json.loads(path.read_text())
    assert report["rank"] == 7
    assert [p["module"] for p in report["probes"]] == (
        ["torch", "transformer_engine.pytorch"]
        if mode == "only"
        else ["torch", "transformer_engine.pytorch", "deep_ep", "modelopt.torch"]
    )


@pytest.mark.parametrize(
    "mode,rank,directory", [("bad", "0", True), ("only", "-1", True), ("only", None, True), ("only", "0", False)]
)
def test_invalid_settings(preflight, monkeypatch, tmp_path, mode, rank, directory):
    monkeypatch.setenv("MBRIDGE_RUNTIME_PREFLIGHT", mode)
    if rank is not None:
        monkeypatch.setenv("RANK", rank)
    if directory:
        monkeypatch.setenv("MBRIDGE_RUNTIME_PREFLIGHT_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        preflight.run_from_environment()
    preflight._run_probe.assert_not_called()


def test_bootstrap_preflight_precedes_dependency_imports(tmp_path):
    # -S hides site packages: the preflight must be reached without nemo_run or Torch.
    env = {**os.environ, "MBRIDGE_RUNTIME_PREFLIGHT": "invalid"}
    result = subprocess.run(
        [sys.executable, "-S", str(MODULE_PATH.parents[1] / "bootstrap.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "MBRIDGE_RUNTIME_PREFLIGHT must be only or check" in result.stderr
    assert "nemo_run" not in result.stderr


@pytest.mark.parametrize("error", [OSError("missing executable"), subprocess.TimeoutExpired(["python"], 120)])
def test_probe_process_failure_is_reported(monkeypatch, error):
    spec = importlib.util.spec_from_file_location("runtime_preflight_real", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", Mock(side_effect=error))
    result = module._run_probe("torch")
    assert result["returncode"] in {124, 127}
