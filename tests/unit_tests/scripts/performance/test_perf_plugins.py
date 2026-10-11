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

"""Tests for scripts/performance/perf_plugins.py — NsysPlugin output filenames."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# scripts/performance is not an installed package; add it to sys.path so we
# can import ``perf_plugins`` the same way the scripts themselves do.
_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run as run

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

try:
    from nemo_run.core.execution.nvcre import NvcreExecutor

    HAS_NVCRE = True
except ImportError:
    HAS_NVCRE = False

if HAS_NEMO_RUN:
    from perf_plugins import NsysPlugin


def _run_nsys_setup(executor: MagicMock) -> str:
    task = MagicMock(spec=run.Script)
    task.args = []
    NsysPlugin(profile_step_start=10, profile_step_end=20).setup(task, executor)
    return executor.get_launcher.return_value.nsys_filename


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_nsys_filename_uses_slurm_placeholders_for_slurm_executor():
    """NsysPlugin must set Slurm %q{} placeholders for the nsys output filename."""
    nsys_filename = _run_nsys_setup(MagicMock(spec=run.SlurmExecutor))

    assert "%q{SLURM_JOB_ID}" in nsys_filename
    assert "%q{SLURM_PROCID}" in nsys_filename
    assert "${" not in nsys_filename


@pytest.mark.skipif(not HAS_NVCRE, reason="nemo_run nvcre executor not available")
def test_nsys_filename_uses_nsys_placeholders_for_nvcre_executor():
    """NsysPlugin must use nsys %q{} placeholders (not bash ${}) for NVCRE executors."""
    nsys_filename = _run_nsys_setup(MagicMock(spec=NvcreExecutor))

    assert "%q{PET_NODE_RANK}" in nsys_filename
    assert "${PET_NODE_RANK}" not in nsys_filename
