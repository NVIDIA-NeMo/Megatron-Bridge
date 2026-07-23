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
from typing import Dict, List, Optional, Union

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


@dataclass(kw_only=True)
class RunAIPlugin(Plugin):
    """NVIDIA Run:ai (RoCE / SR-IOV) fabric.

    Attaches RoCE/GDR rails as Kubernetes extended resources and Multus
    network-attachment annotations, and optionally enlarges ``/dev/shm`` via an
    ``emptyDir`` volume.  This is the on-prem / colocation equivalent of the
    cloud CSP plugins: the networking topology is expressed through Multus
    NetworkAttachmentDefinitions and SR-IOV device-plugin resources instead of
    cloud-provider device plugins (EFA, gIB).

    Typical B300 NVL72 topology exposes 8 RoCE rails (``r0-p0`` … ``r7-p0``)
    each with one SR-IOV VF. A single Multus annotation attaches all rails.

    Attributes:
        extended_resources: Kubernetes extended-resource requests per pod,
            e.g. ``{"nvidia.com/r0-p0": "1", "nvidia.com/r1-p0": "1", …}``.
            Maps directly to ``extra_resource_requests`` / ``limits``.
        annotations: Pod annotations dict, typically a single
            ``k8s.v1.cni.cncf.io/networks`` key listing comma-separated Multus
            NetworkAttachmentDefinitions.
        large_shm: Mount a memory-backed ``/dev/shm`` (``emptyDir.medium:
            Memory``) to avoid the default 64 MiB limit, which is too small for
            NCCL shared-memory collectives on multi-GPU nodes.
        pvc_claim_name: If set, attach the named PersistentVolumeClaim to the
            pod and mount it at ``pvc_mount_path``.  Typically the shared
            workspace PVC that holds model assets, HuggingFace cache, and
            experiment logs.
        pvc_mount_path: Container mount path for the PVC.
        env_vars: Additional environment variables injected into the training
            container (e.g. ``TRANSFORMERS_OFFLINE``, ``HF_HOME``).
        scheduler_name: If set, pin the workload pods to this Kubernetes
            scheduler (``spec.schedulerName``). Run:ai gang-schedules through its
            own scheduler — typically ``"runai-scheduler"`` — so raw
            PyTorchJob/TrainJob submissions (i.e. not via the ``runai`` CLI) must
            name it explicitly or the default scheduler will place the pods and
            bypass Run:ai quota/fair-share. Left unset (default) so non-Run:ai
            paths are unaffected.
        labels: Extra pod labels merged onto the workload pods. Run:ai expresses
            project/queue membership through a pod label whose key varies by
            version (e.g. ``project`` or ``kai.scheduler/queue``); pass the
            key/value your cluster expects here rather than hardcoding one.
    """

    extended_resources: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    large_shm: bool = True
    pvc_claim_name: Optional[str] = None
    pvc_mount_path: str = "/nemo-workspace"
    env_vars: Dict[str, str] = field(default_factory=dict)
    scheduler_name: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor") -> None:
        """Layer the Run:ai RoCE/SR-IOV fabric onto a Kubeflow executor."""
        if not isinstance(executor, KubeflowExecutor):
            return

        if self.scheduler_name:
            # pod_spec_overrides merges into the (v2 TrainJob) podTemplateOverrides
            # spec and the (v1 PyTorchJob) replica pod spec alike.
            executor.pod_spec_overrides = {
                **executor.pod_spec_overrides,
                "schedulerName": self.scheduler_name,
            }

        if self.labels:
            if hasattr(executor, "pod_labels"):
                executor.pod_labels = {**executor.pod_labels, **self.labels}
            else:
                executor.labels = {**executor.labels, **self.labels}

        if self.extended_resources:
            executor.extra_resource_requests = {**executor.extra_resource_requests, **self.extended_resources}
            executor.extra_resource_limits = {**executor.extra_resource_limits, **self.extended_resources}

        if self.annotations:
            if hasattr(executor, "pod_annotations"):
                executor.pod_annotations = {**executor.pod_annotations, **self.annotations}
            else:
                executor.annotations = {**executor.annotations, **self.annotations}

        if self.large_shm and not any(v.get("name") == "dshm" for v in executor.volumes):
            executor.volumes.append({"name": "dshm", "emptyDir": {"medium": "Memory"}})
            executor.volume_mounts.append({"name": "dshm", "mountPath": "/dev/shm"})

        if self.pvc_claim_name:
            vol_name = "runai-workspace"
            if not any(v.get("name") == vol_name for v in executor.volumes):
                executor.volumes.append(
                    {"name": vol_name, "persistentVolumeClaim": {"claimName": self.pvc_claim_name}}
                )
                executor.volume_mounts.append({"name": vol_name, "mountPath": self.pvc_mount_path})

        if self.env_vars:
            executor.env_vars.update(self.env_vars)
