"""Dependency-free tests for opt-in base runtime preservation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def guard(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "preserve_base_runtime", ROOT / "docker/common/preserve_base_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("MBRIDGE_PRESERVE_BASE_RUNTIME", "True")

    def distribution(name):
        if name not in module.REQUIRED:
            raise module.importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(version="base-version", locate_file=lambda path: Path("/base/site-packages") / path)

    monkeypatch.setattr(module.importlib.metadata, "distribution", distribution)
    return module


@pytest.mark.parametrize("value", [None, "False"])
def test_disabled_does_not_read_packages_or_manifest(guard, monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv("MBRIDGE_PRESERVE_BASE_RUNTIME")
    else:
        monkeypatch.setenv("MBRIDGE_PRESERVE_BASE_RUNTIME", value)
    monkeypatch.setattr(guard, "package_state", Mock(side_effect=AssertionError("must not read metadata")))
    guard.verify_runtime(tmp_path / "absent", record=False)


def test_records_selected_required_packages_and_accepts_unchanged_runtime(guard, tmp_path):
    path = tmp_path / "manifest.json"
    guard.verify_runtime(path, record=True)
    assert set(json.loads(path.read_text())) == set(guard.REQUIRED)
    guard.verify_runtime(path, record=False)


@pytest.mark.parametrize("field,value", [("version", "replacement-version"), ("location", "/shadow/site-packages")])
def test_rejects_replacement_or_shadowing(guard, monkeypatch, tmp_path, field, value):
    path = tmp_path / "manifest.json"
    guard.verify_runtime(path, record=True)
    actual = guard.package_state()
    actual["transformer-engine"][field] = value
    monkeypatch.setattr(guard, "package_state", lambda: actual)
    with pytest.raises(RuntimeError, match="Base runtime packages changed"):
        guard.verify_runtime(path, record=False)


def test_missing_required_package_fails(guard, monkeypatch):
    monkeypatch.setattr(
        guard.importlib.metadata,
        "distribution",
        Mock(side_effect=guard.importlib.metadata.PackageNotFoundError("torch")),
    )
    with pytest.raises(guard.importlib.metadata.PackageNotFoundError):
        guard.package_state()


def test_missing_manifest_fails(guard, tmp_path):
    with pytest.raises(FileNotFoundError):
        guard.verify_runtime(tmp_path / "absent", record=False)


def test_invalid_setting_fails(guard, monkeypatch):
    monkeypatch.setenv("MBRIDGE_PRESERVE_BASE_RUNTIME", "typo")
    with pytest.raises(ValueError):
        guard.enabled()


def test_docker_stages_preserve_runtime_only_when_enabled():
    ci = (ROOT / "docker/Dockerfile.ci").read_text()
    final = (ROOT / "docker/Dockerfile.fw_final").read_text()
    assert "ARG PRESERVE_BASE_RUNTIME=False" in ci
    assert "MBRIDGE_PRESERVE_BASE_RUNTIME=${PRESERVE_BASE_RUNTIME}" in ci
    for dockerfile in (ci, final):
        syncs = [line for line in dockerfile.splitlines() if "uv sync " in line and not line.lstrip().startswith("#")]
        assert syncs
        assert all("${BASE_RUNTIME_UV_ARGS}" in line for line in syncs)
        assert "--no-install-package nvidia-cudnn-frontend" in dockerfile
        assert "--no-install-package transformer-engine" in dockerfile
        assert "preserve_base_runtime.py check /opt/base-runtime-packages.json" in dockerfile
    assert 'if [ "$MBRIDGE_PRESERVE_BASE_RUNTIME" != "True" ]; then' in ci
    # FW-final may also be built on an older, non-opt-in Bridge image.
    assert "COPY docker/common/preserve_base_runtime.py /opt/preserve_base_runtime.py" in final
