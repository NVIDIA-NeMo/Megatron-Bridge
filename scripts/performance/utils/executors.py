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

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home, set_nemorun_home
from nemo_run.core.execution.launcher import SlurmTemplate


DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
DEFAULT_NEMO_HOME = os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME)
logger = logging.getLogger(__name__)

# NOTE: If you update this template,
# PLEASE test it by submitting a job to GPU/node/cluster and verifying the sbatch and bash scripts.
INLINE_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c '{{ pre_cmds }} {{ command }}'
"""

PERF_ENV_VARS = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
    "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
    "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
    "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
    "NVTE_NORM_FWD_USE_CUDNN": "1",
    "NVTE_NORM_BWD_USE_CUDNN": "1",
    "TORCH_NCCL_HIGH_PRIORITY": "1",
    "HF_HUB_OFFLINE": "0",
}


def slurm_executor(
    gpu: str,
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    hf_token: str = None,
    nemo_home: str = DEFAULT_NEMO_HOME,
    wandb_key: str = None,
    network: str = None,
    custom_bash_cmds: List[str] = None,
    additional_slurm_params: Dict[str, Any] = None,
    gres: Optional[str] = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments

    Args:
        additional_slurm_params: Dict[str, Any], optional
            Additional SLURM parameters to pass to sbatch. These will be converted to #SBATCH directives.
            Example: {"nodelist": "node001,node002", "constraint": "gpu"} will generate:
                #SBATCH --nodelist=node001,node002
                #SBATCH --constraint=gpu
    """
    custom_bash_cmds = [] if custom_bash_cmds is None else custom_bash_cmds
    mounts = []
    # Explicitly request GPU resources to ensure proper allocation
    # Without --gres=gpu:N, some clusters only allocate 1 GPU regardless of ntasks_per_node
    srun_args = custom_srun_args.copy() + [
        "--mpi=pmix",
        "--no-container-mount-home",
    ]

    if log_dir is not None:
        set_nemorun_home(log_dir)
    else:
        if os.environ.get("NEMORUN_HOME") is None:
            logger.warning(
                f"Logs will be written to {get_nemorun_home()}, which is probably not desired.  export NEMORUN_HOME in your shell environment or use the --log_dir argument"
            )

    if wandb_key is not None:
        PERF_ENV_VARS["WANDB_API_KEY"] = wandb_key

    if gpu.lower() == "gb200":
        PERF_ENV_VARS["NCCL_NET_GDR_LEVEL"] = "PHB"  # For NCCL 2.25
        PERF_ENV_VARS["NCCL_NET_GDR_C2C"] = "1"  # For NCCL 2.26

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        PERF_ENV_VARS["NEMO_HOME"] = nemo_home
        mounts.extend([f"{nemo_home}:{nemo_home}"])
    if hf_token is not None:
        PERF_ENV_VARS.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    PERF_ENV_VARS.update(custom_env_vars)
    mounts.extend(custom_mounts)

    # add --segment flag to sbatch if job uses GB200.
    segment = None
    if num_gpus_per_node == 4:
        if nodes <= 18:
            segment = nodes
        else:  # nodes > 18
            for segment_candidate in range(18, 0, -1):
                if nodes % segment_candidate == 0:
                    segment = segment_candidate
                    break

    numa_divisor = 2 if gpu.lower() in ["gb200", "gb300"] else 4
    numa_cmd = f"numactl --cpunodebind=$((SLURM_LOCALID/{numa_divisor})) --membind=$((SLURM_LOCALID/{numa_divisor}))"
    custom_bash_cmds.append(numa_cmd)

    launcher = SlurmTemplate(
        template_inline=INLINE_TEMPLATE,
        template_vars={"pre_cmds": " ; ".join(custom_bash_cmds)},
    )

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        gres=gres,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=PERF_ENV_VARS,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
        segment=segment,
        network=network,
        launcher=launcher,
        additional_parameters=additional_slurm_params,
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
                "annotations": [{"name": "runai.dgxc.nvidia.com/gcp-nccl", "value": "none", "exclude": False}],
            }
            if dgxc_cluster == "dgxcloud-gcp" and nodes == 1
            else {}
        ),
        env_vars=env_vars,
        launcher="torchrun",
    )
    return executor
