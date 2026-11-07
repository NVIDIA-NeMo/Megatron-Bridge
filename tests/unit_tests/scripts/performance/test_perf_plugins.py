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
    executor = MagicMock(spec=NvcreExecutor)
    executor.capabilities = []  # the plugin requests a capability from NVCRE executors, so the mock must hold one
    nsys_filename = _run_nsys_setup(executor)

    assert "%q{PET_NODE_RANK}" in nsys_filename
    assert "${PET_NODE_RANK}" not in nsys_filename


# ── NVCRE container capability ────────────────────────────────────────────────

HAS_NVCRE_CAPABILITIES = HAS_NVCRE and "capabilities" in getattr(NvcreExecutor, "__dataclass_fields__", {})
needs_nvcre_capabilities = pytest.mark.skipif(
    not HAS_NVCRE_CAPABILITIES, reason="nemo_run NvcreExecutor without capabilities"
)


def _nvcre_executor() -> "NvcreExecutor":
    executor = NvcreExecutor(namespace="ns", container_image="img:latest", num_nodes=2, gpus_per_node=4)
    executor.job_name = "job1"
    executor.experiment_id = "exp_123456789"
    executor.job_dir = "/tmp/exp/job1"
    return executor


def _setup_nsys_plugin(executor) -> None:
    task = MagicMock(spec=run.Script)
    task.args = []
    NsysPlugin(profile_step_start=10, profile_step_end=20).setup(task, executor)


@needs_nvcre_capabilities
def test_nsys_plugin_requests_sys_admin_for_nvcre_executor():
    """Profiling on NVCRE needs SYS_ADMIN where the driver restricts GPU profiling, so the plugin asks for it."""
    executor = _nvcre_executor()

    _setup_nsys_plugin(executor)

    assert executor.capabilities == ["SYS_ADMIN"]


@needs_nvcre_capabilities
def test_nsys_plugin_keeps_capabilities_already_requested_and_adds_sys_admin_once():
    executor = _nvcre_executor()
    executor.capabilities = ["IPC_LOCK"]

    _setup_nsys_plugin(executor)
    _setup_nsys_plugin(executor)

    assert executor.capabilities == ["IPC_LOCK", "SYS_ADMIN"]


@needs_nvcre_capabilities
def test_nsys_plugin_leaves_an_explicit_sys_admin_alone():
    executor = _nvcre_executor()
    executor.capabilities = ["SYS_ADMIN", "IPC_LOCK"]

    _setup_nsys_plugin(executor)

    assert executor.capabilities == ["SYS_ADMIN", "IPC_LOCK"]


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_nsys_plugin_does_not_request_capabilities_for_slurm_executor():
    executor = MagicMock(spec=run.SlurmExecutor)

    _setup_nsys_plugin(executor)

    assert not hasattr(executor, "capabilities")


@pytest.mark.skipif(not HAS_NVCRE, reason="nemo_run nvcre executor not available")
def test_nsys_plugin_errors_when_nemo_run_cannot_request_capabilities():
    """Without the capability nsys records no GPU kernels, so an older nemo_run must fail instead of profiling blind."""
    executor = MagicMock(spec=NvcreExecutor)
    executor.capabilities = []
    del executor.capabilities  # an NvcreExecutor without a ``capabilities`` field

    with pytest.raises(RuntimeError, match="needs the SYS_ADMIN capability so nsys can trace GPU kernels"):
        _setup_nsys_plugin(executor)


@pytest.mark.skipif(not HAS_NVCRE, reason="nemo_run nvcre executor not available")
def test_request_nsys_capability_error_leaves_the_executor_unchanged():
    from types import SimpleNamespace

    import perf_plugins

    executor = SimpleNamespace()

    with pytest.raises(RuntimeError, match="SYS_ADMIN"):
        perf_plugins._request_nsys_capability(executor)

    assert not hasattr(executor, "capabilities")


@needs_nvcre_capabilities
def test_profiled_nvcre_workloadrun_carries_the_sys_admin_override():
    """End to end: the plugin's request ends up as the TrainingRuntime override in the submitted WorkloadRun."""
    executor = _nvcre_executor()
    assert "overrides" not in executor.build_workloadrun_yaml(["python"])["spec"]

    _setup_nsys_plugin(executor)
    manifest = executor.build_workloadrun_yaml(["python"])

    [override] = manifest["spec"]["overrides"]
    [runtime] = override["dependencies"]
    assert runtime["metadata"]["name"] == f"{manifest['metadata']['name']}-runtime"
    [job] = runtime["spec"]["template"]["spec"]["replicatedJobs"]
    [container] = job["template"]["spec"]["template"]["spec"]["containers"]
    assert container["securityContext"] == {"capabilities": {"add": ["SYS_ADMIN"]}}
