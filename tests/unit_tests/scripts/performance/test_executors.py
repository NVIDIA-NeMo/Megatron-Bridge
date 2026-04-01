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

"""Tests for scripts/performance/utils/executors.py — container_env on SlurmExecutor."""

import sys
from pathlib import Path

import pytest

# scripts/performance is not an installed package; add it to sys.path so we
# can import ``utils.executors`` the same way the scripts themselves do.
_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run  # noqa: F401

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from utils.executors import PERF_ENV_VARS, slurm_executor


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_container_env_includes_perf_vars(tmp_path):
    """PERF_ENV_VARS keys must appear in container_env so they override container defaults."""
    executor = slurm_executor(
        gpu="h100", account="test", partition="test",
        log_dir=str(tmp_path), nodes=1, num_gpus_per_node=8,
    )
    assert executor.container_env is not None, "container_env is None — was the field removed from the executor?"
    missing = set(PERF_ENV_VARS) - set(executor.container_env)
    assert not missing, f"PERF_ENV_VARS keys missing from container_env: {missing}"


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_custom_env_vars_in_container_env(tmp_path):
    """Vars passed via custom_env_vars must also appear in container_env."""
    executor = slurm_executor(
        gpu="h100", account="test", partition="test",
        log_dir=str(tmp_path), nodes=1, num_gpus_per_node=8,
        custom_env_vars={"MY_CUSTOM_VAR": "1"},
    )
    assert "MY_CUSTOM_VAR" in executor.container_env


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_container_env_param_forwarded(tmp_path):
    """Keys passed via the container_env parameter must appear in container_env."""
    executor = slurm_executor(
        gpu="h100", account="test", partition="test",
        log_dir=str(tmp_path), nodes=1, num_gpus_per_node=8,
        container_env=["UPSTREAM_SET_VAR"],
    )
    assert "UPSTREAM_SET_VAR" in executor.container_env
