# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Static contracts for functional-test launch scripts."""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCH_ROOT = REPO_ROOT / "tests" / "functional_tests" / "launch_scripts"
LAUNCH_NAME_RE = re.compile(r"^L[0-2]_[A-Za-z0-9_]+\.sh$")
TEST_GROUP_REF_RE = re.compile(r"(?:tests/functional_tests/)?test_groups/[^\s\"'\\]+")


def _launch_scripts() -> list[Path]:
    return sorted(LAUNCH_ROOT.glob("*/*/*.sh"))


def _repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _format_failures(paths: list[Path]) -> str:
    return "\n".join(f"- {_repo_relative(path)}" for path in paths)


def _referenced_test_group_paths(script: Path) -> list[Path]:
    refs = []
    for match in TEST_GROUP_REF_RE.findall(script.read_text()):
        normalized = match
        prefix = "tests/functional_tests/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
        normalized = normalized.split("::", maxsplit=1)[0]
        normalized = normalized.rstrip("),;")
        refs.append(REPO_ROOT / "tests" / "functional_tests" / normalized)
    return refs


def test_launch_scripts_use_tiered_names():
    bad_names = [path for path in _launch_scripts() if not LAUNCH_NAME_RE.fullmatch(path.name)]

    assert not bad_names, "Launch scripts must use {L0,L1,L2}_*.sh names:\n" + _format_failures(bad_names)


def test_launch_scripts_are_executable():
    not_executable = [path for path in _launch_scripts() if not path.stat().st_mode & stat.S_IXUSR]

    assert not not_executable, "Launch scripts must be executable:\n" + _format_failures(not_executable)


def test_launch_scripts_do_not_reference_old_functional_model_path():
    stale_refs = [
        path
        for path in _launch_scripts()
        if "tests/functional_tests/models/" in path.read_text() or "tests.functional_tests.models" in path.read_text()
    ]

    assert not stale_refs, "Launch scripts must use tests/functional_tests/test_groups/models paths:\n" + (
        _format_failures(stale_refs)
    )


def test_launch_scripts_reference_existing_test_groups_paths():
    missing = []
    for script in _launch_scripts():
        for referenced_path in _referenced_test_group_paths(script):
            if not referenced_path.exists():
                missing.append(f"{_repo_relative(script)} -> {_repo_relative(referenced_path)}")

    assert not missing, "Launch scripts reference missing test_groups paths:\n" + "\n".join(
        f"- {item}" for item in missing
    )
