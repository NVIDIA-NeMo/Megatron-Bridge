# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
    wandb_key: str = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    env_vars = {
        "WANDB_API_KEY": wandb_key,
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


def dgxc_executor(
    dgxc_base_url: str,
    dgxc_cluster: str,
    dgxc_kube_apiserver_url: str,
    dgxc_app_id: str,
    dgxc_app_secret: str,
    dgxc_project_name: str,
    dgxc_pvc_claim_name: str,
    nodes: int,
    num_gpus_per_node: int,
    wandb_key: str = None,
    hf_token: str = None,
    custom_env_vars: Dict[str, str] = None,
    dgxc_pvc_mount_path: str = "/nemo-workspace",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
):
    """
    DGXCloud cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """

    env_vars = {
        "TORCH_HOME": "/nemo-workspace/.cache",
        "FI_EFA_USE_HUGE_PAGE": "0",
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_P2P_NET_CHUNKSIZE": "524288",
        "NCCL_TUNER_PLUGIN": "/opt/gcp-ofi-nccl/install/lib/libnccl-ofi-tuner.so",
        "WANDB_API_KEY": wandb_key,
        "HF_TOKEN": hf_token,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "False",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/nemo-workspace/pagaray/hf_cache",
    }
    if custom_env_vars:
        env_vars.update(custom_env_vars)

    executor = run.DGXCloudExecutor(
        base_url=dgxc_base_url,
        kube_apiserver_url=dgxc_kube_apiserver_url,
        app_id=dgxc_app_id,
        app_secret=dgxc_app_secret,
        project_name=dgxc_project_name,
        nodes=nodes,
        gpus_per_node=num_gpus_per_node,
        container_image=container_image,
        pvc_nemo_run_dir=get_nemorun_home(),
        launched_from_cluster=True,
        pvcs=[
            {
                "name": "workspace",
                "path": dgxc_pvc_mount_path,
                "existingPvc": True,
                "claimName": dgxc_pvc_claim_name,
            }
        ],
        custom_spec=(
            {
                "annotations": [
                    {
                        "name": "runai.dgxc.nvidia.com/gcp-nccl",
                        "value": "none",
                        "exclude": False,
                    }
                ],
            }
            if dgxc_cluster == "dgxcloud-gcp" and nodes == 1
            else {}
        ),
        env_vars=env_vars,
        launcher="torchrun",
    )
    return executor


def local_executor(hf_token: str) -> run.LocalExecutor:
    # [snippet-local-executor-start]
    env_vars = {
        # required for some eval benchmarks from lm-eval-harness
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "HF_TOKEN": hf_token,  # [hf-token-local]
    }

    executor = run.LocalExecutor(env_vars=env_vars)
    # [snippet-local-executor-end]
    return executor
