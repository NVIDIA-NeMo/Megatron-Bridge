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

"""Tests for scripts/performance/perf_plugins.py PerfEnvPlugin wiring."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run  # noqa: F401

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

pytestmark = pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")

if HAS_NEMO_RUN:
    import perf_plugins
    from perf_plugins import PerfEnvPlugin


def test_set_determinism_env_vars_writes_three_keys():
    plugin = PerfEnvPlugin(
        deterministic=True,
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        gpu="h100",
        compute_dtype="bf16",
        train_task="pretrain",
    )
    executor = MagicMock()
    executor.env_vars = {}

    plugin._set_determinism_env_vars(executor)

    assert executor.env_vars["NCCL_ALGO"] == "Ring"
    assert executor.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert executor.env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_setup_adds_peak_mem_clock_slurm_command(monkeypatch):
    class FakeSlurmExecutor:
        def __init__(self):
            self.tunnel = SimpleNamespace(job_dir="/tmp/job")
            self.setup_lines = ""

    monkeypatch.setattr(perf_plugins, "SlurmExecutor", FakeSlurmExecutor)
    plugin = PerfEnvPlugin(
        enable_manual_gc=False,
        peak_mem_clk=2600,
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        gpu="h100",
        compute_dtype="bf16",
        train_task="pretrain",
    )
    executor = FakeSlurmExecutor()

    plugin.setup(MagicMock(), executor)

    assert "nvidia-smi -lmi" in executor.setup_lines
    assert "sudo nvidia-smi -lmc 2600,2600" in executor.setup_lines
    assert executor.setup_lines.index("nvidia-smi -lmi") < executor.setup_lines.index("sudo nvidia-smi -lmc")
    assert "--output /tmp/job/peak_mem_clock.out" in executor.setup_lines
    assert "--error /tmp/job/peak_mem_clock.err" in executor.setup_lines
