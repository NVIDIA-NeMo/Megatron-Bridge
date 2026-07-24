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

"""Tests for scripts/performance/utils/csp_plugins.py — CSP fabric plugins."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# scripts/performance is not an installed package; add it to sys.path so we can
# import ``utils.csp_plugins`` the same way the scripts themselves do.
_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run  # noqa: F401
    from nemo_run.core.execution.kubeflow import KubeflowExecutor

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from utils.csp_plugins import AWS_OFI_NCCL_NET_PLUGIN, EKSEnvPlugin, GKEEnvPlugin


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_eks_plugin_pins_aws_ofi_and_enables_efa():
    """The EFA plugin must pin the aws-ofi net plugin and request EFA fabric."""
    executor = KubeflowExecutor()

    EKSEnvPlugin().setup(task=None, executor=executor)

    assert executor.env_vars["NCCL_NET_PLUGIN"] == AWS_OFI_NCCL_NET_PLUGIN
    assert executor.env_vars["FI_PROVIDER"] == "efa"
    assert executor.env_vars["FI_EFA_USE_HUGE_PAGE"] == "0"
    assert executor.extra_resource_requests["vpc.amazonaws.com/efa"] == "32"
    assert executor.extra_resource_limits["vpc.amazonaws.com/efa"] == "32"
    assert executor.container_kwargs["securityContext"]["privileged"] is True
    assert any(volume.get("name") == "rdma-dev" for volume in executor.volumes)


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_eks_plugin_respects_preset_net_plugin():
    """A recipe/user override of NCCL_NET_PLUGIN must win over the aws-ofi default."""
    executor = KubeflowExecutor(env_vars={"NCCL_NET_PLUGIN": "/custom/libnccl-net.so"})

    EKSEnvPlugin().setup(task=None, executor=executor)

    assert executor.env_vars["NCCL_NET_PLUGIN"] == "/custom/libnccl-net.so"


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_eks_plugin_is_noop_off_kubeflow():
    """Selecting EKS on a non-Kubeflow executor must not mutate it."""
    executor = SimpleNamespace()

    EKSEnvPlugin().setup(task=None, executor=executor)

    assert not hasattr(executor, "env_vars")


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_gke_plugin_never_sets_net_plugin_and_selects_gib():
    """GKE fabric attaches gIB and must never touch the aws-ofi NCCL_NET_PLUGIN."""
    executor = KubeflowExecutor()

    GKEEnvPlugin(rdma_networks=["rdma-0", "rdma-1"]).setup(task=None, executor=executor)

    assert "NCCL_NET_PLUGIN" not in executor.env_vars
    assert executor.env_vars["NCCL_NET"] == "gIB"
    assert "networking.gke.io/interfaces" in executor.pod_annotations


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_gke_plugin_is_noop_without_networks():
    """A single-block GKE run (no RDMA networks) must not alter fabric env."""
    executor = KubeflowExecutor()

    GKEEnvPlugin().setup(task=None, executor=executor)

    assert "NCCL_NET" not in executor.env_vars
    assert "NCCL_NET_PLUGIN" not in executor.env_vars
