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

"""Cloud-provider (CSP) fabric plugins for Kubeflow (K8s) executors.

These NeMo-Run ``run.Plugin`` classes inject the CSP-specific networking /
fabric configuration onto a ``KubeflowExecutor`` at launch. Selecting a plugin
*is* the statement "this job runs on that CSP", so each enables its fabric
unconditionally — there is no extra ``enabled`` toggle.

Scope is deliberately CSP networking/fabric only. Arch/recipe/perf env
(``NCCL_NVLS_ENABLE``, ``NVTE_*`` FP8 amax, NCCL buffer/chunk tuning) is NOT a
CSP concern — it varies by GPU arch and recipe, not by cloud — and stays with
the recipe/perf configuration.
"""

import json
from dataclasses import dataclass, field
from typing import List, Union

import nemo_run as run
from nemo_run import Plugin
from nemo_run.core.execution.kubeflow import KubeflowExecutor


# Absolute path to the aws-ofi-nccl EFA net plugin in the FW image (installed by
# docker/common/install_aws_ofi_nccl.sh). NVIDIA PyTorch >= 26.06 base images drop
# aws-ofi-nccl and default NCCL_NET_PLUGIN=spcx (HPCX Spectrum-X), which cannot
# drive EFA. Pinning the absolute path forces NCCL back onto aws-ofi and is
# unambiguous (the image also ships HPCX libnccl-net.so on the ldconfig path).
AWS_OFI_NCCL_NET_PLUGIN = "/opt/amazon/ofi-nccl/lib/libnccl-net.so"


@dataclass(kw_only=True)
class EKSEnvPlugin(Plugin):
    """AWS EKS (EFA) fabric.

    Selecting this plugin means the job runs on an EFA-equipped EKS cluster, so
    EFA is enabled unconditionally: the pod requests EFA devices, runs
    privileged with the host RDMA device nodes mounted, and NCCL uses the
    aws-ofi / libfabric EFA provider. The plugin pins ``NCCL_NET_PLUGIN`` to the
    aws-ofi net plugin because newer base images default it to ``spcx`` (HPCX
    Spectrum-X), which cannot drive EFA and silently degrades NCCL to TCP.

    Attributes:
        efa_device_count: ``vpc.amazonaws.com/efa`` devices requested per node.
    """

    efa_device_count: int = 32

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor") -> None:
        """Layer the EFA fabric onto a Kubeflow executor (no-op otherwise)."""
        if not isinstance(executor, KubeflowExecutor):
            return
        efa = {"vpc.amazonaws.com/efa": str(self.efa_device_count)}
        executor.extra_resource_requests = {**executor.extra_resource_requests, **efa}
        executor.extra_resource_limits = {**executor.extra_resource_limits, **efa}
        # libfabric EFA provider + the aws-ofi net plugin (overriding the base
        # image's spcx default, which does not speak EFA).
        executor.env_vars.setdefault("FI_PROVIDER", "efa")
        executor.env_vars.setdefault("FI_EFA_USE_HUGE_PAGE", "0")
        executor.env_vars.setdefault("NCCL_NET_PLUGIN", AWS_OFI_NCCL_NET_PLUGIN)
        # EFA requires a privileged container and the host /dev/infiniband nodes.
        security_context = {**executor.container_kwargs.get("securityContext", {}), "privileged": True}
        executor.container_kwargs = {**executor.container_kwargs, "securityContext": security_context}
        if not any(volume.get("name") == "rdma-dev" for volume in executor.volumes):
            executor.volumes.append({"name": "rdma-dev", "hostPath": {"path": "/dev/infiniband"}})
            executor.volume_mounts.append({"name": "rdma-dev", "mountPath": "/dev/infiniband"})


@dataclass(kw_only=True)
class GKEEnvPlugin(Plugin):
    """GCP GKE (GPUDirect-RDMA / gIB) fabric.

    Attaches the gIB RDMA NICs via the ``networking.gke.io/interfaces`` pod
    annotation and selects the gIB NCCL net transport. This is only needed for
    inter-node RDMA; single-block (intra-NVLink) runs need no RDMA NICs, so the
    default (no networks) is a no-op.

    Attributes:
        rdma_networks: GKE Network names to attach, in order
            (e.g. ``["rdma-0", "rdma-1", "rdma-2", "rdma-3"]``).
        rdma_interface_prefix: Interface-name prefix for the attached NICs.
    """

    rdma_networks: List[str] = field(default_factory=list)
    rdma_interface_prefix: str = "eth"

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor") -> None:
        """Attach gIB RDMA NICs onto a Kubeflow executor (no-op without networks)."""
        if not isinstance(executor, KubeflowExecutor) or not self.rdma_networks:
            return
        interfaces = [
            {"interfaceName": f"{self.rdma_interface_prefix}{index + 1}", "network": network}
            for index, network in enumerate(self.rdma_networks)
        ]
        executor.pod_annotations = {
            **executor.pod_annotations,
            "networking.gke.io/interfaces": json.dumps(interfaces, separators=(",", ":")),
        }
        executor.env_vars.setdefault("NCCL_NET", "gIB")
