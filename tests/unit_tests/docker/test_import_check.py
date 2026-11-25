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

"""Tests for the image import check's module discovery."""

import importlib.util
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def import_check(monkeypatch):
    spec = importlib.util.spec_from_file_location("import_check", ROOT / "docker/common/import_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve string annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _dist(name, *files):
    return SimpleNamespace(metadata={"Name": name}, files=[PurePosixPath(f) for f in files])


@pytest.fixture
def installed(import_check, monkeypatch):
    dists = [
        _dist("sentencepiece", "sentencepiece/__init__.py", "sentencepiece/sentencepiece_model_pb2.py"),
        _dist("other", "other.py"),
    ]
    monkeypatch.setattr(import_check.md, "distributions", lambda: dists)
    return import_check


def test_sentencepiece_protobuf_module_is_checked(installed):
    assert installed.discover_modules(set()) == {
        "other": ["other"],
        "sentencepiece": ["sentencepiece"],
        "sentencepiece.sentencepiece_model_pb2": ["sentencepiece"],
    }


@pytest.mark.parametrize(
    ("skip", "expected"),
    [
        ({"sentencepiece"}, {"other"}),
        ({"sentencepiece.sentencepiece_model_pb2"}, {"other", "sentencepiece"}),
    ],
)
def test_skip_list_applies_to_extra_submodules(installed, skip, expected):
    assert set(installed.discover_modules(skip)) == expected


def test_extra_submodules_need_their_package(import_check, monkeypatch):
    monkeypatch.setattr(import_check.md, "distributions", lambda: [_dist("other", "other.py")])
    assert import_check.discover_modules(set()) == {"other": ["other"]}


def test_extra_submodules_without_a_file_list(import_check, monkeypatch):
    dist = SimpleNamespace(
        metadata={"Name": "sentencepiece"},
        files=None,
        read_text=lambda name: "sentencepiece\n" if name == "top_level.txt" else None,
    )
    monkeypatch.setattr(import_check.md, "distributions", lambda: [dist])
    assert import_check.discover_modules(set()) == {
        "sentencepiece": ["sentencepiece"],
        "sentencepiece.sentencepiece_model_pb2": ["sentencepiece"],
    }


def _opentelemetry_dists():
    return [
        _dist("opentelemetry-api", "opentelemetry/attributes/__init__.py"),
        _dist("opentelemetry-sdk", "opentelemetry/sdk/metrics/__init__.py"),
    ]


def test_opentelemetry_sdk_metrics_module_is_checked(import_check, monkeypatch):
    monkeypatch.setattr(import_check.md, "distributions", _opentelemetry_dists)
    assert import_check.discover_modules(set()) == {
        "opentelemetry": ["opentelemetry-api", "opentelemetry-sdk"],
        "opentelemetry.sdk.metrics": ["opentelemetry-sdk"],
    }


def test_opentelemetry_api_alone_checks_no_sdk_module(import_check, monkeypatch):
    monkeypatch.setattr(
        import_check.md, "distributions", lambda: [_dist("opentelemetry-api", "opentelemetry/attributes/__init__.py")]
    )
    assert import_check.discover_modules(set()) == {"opentelemetry": ["opentelemetry-api"]}


def test_check_fails_when_opentelemetry_api_is_older_than_sdk(import_check, monkeypatch, tmp_path, capsys):
    # opentelemetry-sdk 1.45 imports a helper that opentelemetry-api 1.43 does not define.
    attributes = tmp_path / "opentelemetry" / "attributes"
    metrics = tmp_path / "opentelemetry" / "sdk" / "metrics"
    attributes.mkdir(parents=True)
    metrics.mkdir(parents=True)
    (attributes / "__init__.py").write_text("def _clean_attribute(key, value, max_len):\n    return value\n")
    (metrics / "__init__.py").write_text("from opentelemetry.attributes import _clean_attribute_value\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(import_check.md, "distributions", _opentelemetry_dists)

    assert import_check.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL  opentelemetry.sdk.metrics" in out
    assert "cannot import name '_clean_attribute_value'" in out
    assert "1 ok, 1 failed" in out


def test_check_fails_when_only_the_protobuf_module_is_skewed(import_check, monkeypatch, tmp_path, capsys):
    package = tmp_path / "sentencepiece"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "sentencepiece_model_pb2.py").write_text("raise RuntimeError('gencode 7.36.1 runtime 6.33.5')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        import_check.md,
        "distributions",
        lambda: [_dist("sentencepiece", "sentencepiece/__init__.py", "sentencepiece/sentencepiece_model_pb2.py")],
    )

    assert import_check.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL  sentencepiece.sentencepiece_model_pb2" in out
    assert "gencode 7.36.1 runtime 6.33.5" in out
    assert "1 ok, 1 failed" in out
