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

import os
from typing import Dict, List

import nemo_run as run
from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup


def slurm_executor(
    account: str,
    partition: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    hf_token: str = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    env_vars = {
        "HF_TOKEN": hf_token,
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "TRANSFORMERS_OFFLINE": "0",
    }
    if custom_env_vars:
        env_vars.update(custom_env_vars)

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        container_image=container_image,
        container_mounts=custom_mounts,
        env_vars=env_vars,
        srun_args=[
            "--mpi=pmix",
            "--no-container-mount-home",
        ],
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
    )

    return executor


def kuberay_executor(
    nodes: int,
    num_gpus_per_node: int,
    dgxc_pvc_mount_path: str,
    dgxc_pvc_claim_name: str,
    namespace: str = "default",
    ray_version: str = "2.43.0",
    container_image: str = "",  # Will be set in __post_init__ if empty
    head_cpu: str = "8",
    head_memory: str = "32Gi",
    hf_token: str = None,
    custom_env_vars: Dict[str, str] = None,
):
    """
    Kuberay cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """

    env_vars = {
        "TORCH_HOME": "/nemo-workspace/.cache",
        "FI_EFA_USE_HUGE_PAGE": "0",
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_P2P_NET_CHUNKSIZE": "524288",
        "NCCL_TUNER_PLUGIN": "/opt/gcp-ofi-nccl/install/lib/libnccl-ofi-tuner.so",
        "HF_TOKEN": hf_token,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "False",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/nemo-workspace/pagaray/hf_cache",
        "RAY_enable_infeasible_task_early_exit": "true",
        "NCCL_IB_DISABLE": "1",
        "NCCL_IB_HCA": "^openib",  # Ignore OpenIB devices
        "NCCL_NET": "Socket",
        "NCCL_NET_GDR_LEVEL": "0",
        "FI_PROVIDER": "tcp",
    }
    if custom_env_vars:
        env_vars.update(custom_env_vars)

    executor = KubeRayExecutor(
        namespace=namespace,
        ray_version=ray_version,
        image=container_image,
        head_cpu=head_cpu,
        head_memory=head_memory,
        ray_head_start_params={"num-gpus": "0", "num-cpus": "0"},
        ray_worker_start_params={"num-gpus": "8", "num-cpus": "128"},
        worker_groups=[
            KubeRayWorkerGroup(
                group_name="worker",
                min_replicas=nodes,
                max_replicas=nodes,
                replicas=nodes,
                gpus_per_worker=num_gpus_per_node,
                cpu_requests="128",
                cpu_limits="128",
                memory_requests="512Gi",
                memory_limits="512Gi",
            )
        ],
        spec_kwargs={
            "schedulerName": "runai-scheduler",
            "image_pull_secrets": ["dockerregistry-dockerregistry-pagaray-ngc"],
        },  # e.g. Run:ai
        volume_mounts=[{"name": "workspace", "mountPath": dgxc_pvc_mount_path}],
        volumes=[
            {
                "name": "workspace",
                "persistentVolumeClaim": {"claimName": dgxc_pvc_claim_name},
            },
        ],
        env_vars=env_vars,
        container_kwargs={
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "runAsUser": 0,
            },
        },
    )

    executor.volumes.append({"name": "dshm", "emptyDir": {"medium": "Memory"}})
    executor.volume_mounts.append({"name": "dshm", "mountPath": "/dev/shm"})

    return executor
