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

"""Smoke tests for the perf config dump debugging tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_DUMP_TOOL = Path(__file__).resolve().parents[4] / "scripts" / "performance" / "dump_perf_configs.py"


def _load_dump_tool():
    spec = importlib.util.spec_from_file_location("dump_perf_configs_under_test", _DUMP_TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dump_perf_configs_tool_is_kept_for_debugging() -> None:
    """Keep the branch-to-branch perf config dump tool available."""
    assert _DUMP_TOOL.exists()

    module = _load_dump_tool()

    assert module.COMBOS
    assert callable(module.load_old_recipe)
    assert callable(module.load_new_recipe)
