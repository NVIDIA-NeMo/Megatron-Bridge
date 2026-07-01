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

"""Prevent official construction workflows from returning to provider APIs."""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

SRC_ROOT = Path(__file__).parents[3] / "src" / "megatron" / "bridge"
REPO_ROOT = SRC_ROOT.parents[2]
FUNCTIONAL_ROOT = REPO_ROOT / "tests" / "functional_tests"
PRIMARY_CONSUMER_ROOTS = (
    SRC_ROOT / "recipes",
    SRC_ROOT / "inference",
)
FORBIDDEN_CALLS = {"provider_bridge", "to_megatron_provider"}
TEMPORARY_PROVIDER_COMPATIBILITY_TESTS = {
    Path("test_groups/models/exaone/test_exaone4_provider.py"),
    Path("test_groups/models/mistral/test_mistral_provider.py"),
    Path("test_groups/models/olmoe/test_olmoe_provider.py"),
}


def test_primary_consumers_do_not_call_provider_build_apis() -> None:
    violations: list[str] = []
    for root in PRIMARY_CONSUMER_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr in FORBIDDEN_CALLS:
                    violations.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}:{node.func.attr}")

    assert not violations, "Provider build APIs remain in primary consumers: " + ", ".join(violations)


def test_functional_flows_only_use_allowlisted_provider_compatibility_calls() -> None:
    """Keep ordinary functional flows on builder APIs while compatibility tests remain explicit."""
    calls: set[Path] = set()
    for path in FUNCTIONAL_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "to_megatron_provider":
                calls.add(path.relative_to(FUNCTIONAL_ROOT))

    assert calls == TEMPORARY_PROVIDER_COMPATIBILITY_TESTS
