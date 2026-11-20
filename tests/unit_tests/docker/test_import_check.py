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


def test_failing_submodule_is_reported_although_package_imports(import_check, monkeypatch, tmp_path):
    package = tmp_path / "skewed"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "generated_pb2.py").write_text("raise RuntimeError('gencode newer than runtime')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    assert import_check.import_one("skewed", timeout=60).ok
    result = import_check.import_one("skewed.generated_pb2", timeout=60)
    assert not result.ok
    assert "gencode newer than runtime" in result.detail
