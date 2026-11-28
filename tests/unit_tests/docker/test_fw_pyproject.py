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

"""Tests for the NeMo Framework image's dependency overlay."""

from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def _never_installed_by_framework_layer() -> set[str]:
    overlay = tomllib.loads((ROOT / "docker/common/fw_pyproject.toml").read_text())
    overrides = [Requirement(spec) for spec in overlay["tool"]["uv"]["override-dependencies"]]
    return {
        req.name for req in overrides if req.marker is not None and not req.marker.evaluate({"sys_platform": "linux"})
    }


def test_framework_layer_leaves_opentelemetry_to_bridge():
    # OpenTelemetry packages require each other at exact releases, and the Framework layer syncs into the
    # Megatron-Bridge environment without removing packages. If it resolved any of them on its own, it would
    # replace that one and leave Megatron-Bridge's others unmatched.
    locked = {package["name"] for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    bridge_opentelemetry = {name for name in locked if name.startswith("opentelemetry-")}
    assert {"opentelemetry-api", "opentelemetry-sdk"} <= bridge_opentelemetry
    resolved_by_framework_layer = bridge_opentelemetry - _never_installed_by_framework_layer()
    assert not resolved_by_framework_layer, (
        f"docker/common/fw_pyproject.toml must leave {sorted(resolved_by_framework_layer)} to Megatron-Bridge's lock"
    )
