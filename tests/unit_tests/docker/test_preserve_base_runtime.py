# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-free tests for opt-in base runtime preservation."""

import importlib.util
import json
import re
import subprocess
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
    assert spec is not None and spec.loader is not None
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


@pytest.mark.parametrize("name", ["transformer-engine-torch", "nvidia-cutlass-dsl-libs-cu13"])
def test_optional_companion_is_recorded_and_removal_is_rejected(guard, monkeypatch, tmp_path, name):
    original_distribution = guard.importlib.metadata.distribution

    def distribution(package):
        if package == name:
            return SimpleNamespace(version="companion-version", locate_file=lambda path: Path("/base/site-packages"))
        return original_distribution(package)

    monkeypatch.setattr(guard.importlib.metadata, "distribution", distribution)
    path = tmp_path / "manifest.json"
    guard.verify_runtime(path, record=True)
    assert json.loads(path.read_text())[name]["version"] == "companion-version"
    guard.verify_runtime(path, record=False)
    monkeypatch.setattr(guard.importlib.metadata, "distribution", original_distribution)
    with pytest.raises(RuntimeError, match="Base runtime packages changed"):
        guard.verify_runtime(path, record=False)


def test_added_companion_is_rejected(guard, monkeypatch, tmp_path):
    path = tmp_path / "manifest.json"
    guard.verify_runtime(path, record=True)
    actual = guard.package_state()
    actual["transformer-engine-torch"] = {"version": "new-version", "location": "/shadow/site-packages"}
    monkeypatch.setattr(guard, "package_state", lambda: actual)
    with pytest.raises(RuntimeError, match="Base runtime packages changed"):
        guard.verify_runtime(path, record=False)


def test_malformed_manifest_fails(guard, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        guard.verify_runtime(path, record=False)


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
    assert ci.index("preserve_base_runtime.py record") < ci.index("uv sync ")
    # The final checks also cover installation layers after the main dependency syncs.
    assert ci.rindex("preserve_base_runtime.py check") > ci.index("bash scripts/install_diffusion_deps.sh")
    assert final.rindex("preserve_base_runtime.py check") > final.index("pip install ")


@pytest.mark.parametrize("filename", ["Dockerfile.ci", "Dockerfile.fw_final"])
def test_modified_docker_run_shell_syntax(filename):
    commands = []
    command = []
    for line in (ROOT / "docker" / filename).read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        if line.startswith("RUN "):
            command = [line.removeprefix("RUN ")]
        elif command:
            command.append(line)
        if command and not line.endswith("\\"):
            body = "\n".join(command)
            if "BASE_RUNTIME_UV_ARGS" in body:
                commands.append(re.sub(r"--mount=\S+", "", body))
            command = []

    assert len(commands) == (2 if filename == "Dockerfile.ci" else 1)
    for body in commands:
        result = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
