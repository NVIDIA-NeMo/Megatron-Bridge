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
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home, set_nemorun_home
from nemo_run.core.execution.launcher import SlurmTemplate


DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
DEFAULT_NEMO_HOME = os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME)
logger = logging.getLogger(__name__)

KUBEFLOW_NUMA_BINDING_ENV = "NEMO_KUBEFLOW_NUMA_BINDING"


def _kubeflow_numa_binding_script(task: run.Script) -> run.Script:
    """Wrap a task with per-rank GPU-local NUMA binding without dropping task metadata."""
    training_command = shlex.join(task.to_command(with_entrypoint=True))
    return run.Script(
        env=task.env.copy(),
        metadata=task.metadata.copy(),
        inline=f"""
set -euo pipefail

: "${{LOCAL_RANK:?LOCAL_RANK must be set by torchrun}}"
command -v nvidia-smi >/dev/null || {{ echo "[numactl_local] nvidia-smi not found" >&2; exit 1; }}
command -v numactl >/dev/null || {{ echo "[numactl_local] numactl not found" >&2; exit 1; }}

PCI_BUS=$(nvidia-smi -i "$LOCAL_RANK" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
    | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]' | sed -E 's/^00000000:/0000:/') || PCI_BUS=""
NUMA_FILE="/sys/bus/pci/devices/$PCI_BUS/numa_node"

if [[ -z "$PCI_BUS" || ! -r "$NUMA_FILE" ]]; then
    echo "[numactl_local] cannot resolve NUMA file for local_rank=$LOCAL_RANK gpu_pci=$PCI_BUS" >&2
    exit 1
fi

NUMA_NODE=$(<"$NUMA_FILE")
if [[ ! "$NUMA_NODE" =~ ^[0-9]+$ ]]; then
    echo "[numactl_local] invalid NUMA node for local_rank=$LOCAL_RANK gpu_pci=$PCI_BUS numa=$NUMA_NODE" >&2
    exit 1
fi

echo "[numactl_local] host=$(hostname) rank=${{RANK:-unknown}} local_rank=$LOCAL_RANK gpu_pci=$PCI_BUS numa=$NUMA_NODE"
exec numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" {training_command}
""",
    )


def _kubeflow_numa_binding_enabled(env_vars: Dict[str, str]) -> bool:
    """Return whether the opt-in Kubeflow NUMA wrapper is enabled."""
    return str(env_vars.get(KUBEFLOW_NUMA_BINDING_ENV, "")).lower() in {
        "1",
        "on",
        "true",
        "yes",
    }


# NOTE: If you update this template,
# PLEASE test it by submitting a job to GPU/node/cluster and verifying the sbatch and bash scripts.
INLINE_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c '{{ pre_cmds }} {{ command }}'
"""

OFFLINE_BENCHMARK_ENV_VARS = {
    "TRANSFORMERS_OFFLINE": "1",  # Default for benchmark runs that mostly use NullTokenizer.
    "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
    "HF_HUB_OFFLINE": "0",  # Keep HF Hub online by default; --offline flips this to 1.
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
    offline: bool = False,
    nemo_home: str = DEFAULT_NEMO_HOME,
    wandb_key: str = None,
    network: str = None,
    custom_bash_cmds: List[List[str]] = None,
    additional_slurm_params: Dict[str, Any] = None,
    gres: Optional[str] = None,
    packager: str = "git",
    enable_pct_binding: bool = True,
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
    custom_bash_cmds = [] if custom_bash_cmds is None else [" ".join(cmd) for cmd in custom_bash_cmds]
    mounts = []
    # Explicitly request GPU resources to ensure proper allocation
    # Without --gres=gpu:N, some clusters only allocate 1 GPU regardless of ntasks_per_node
    srun_args = custom_srun_args.copy() + [
        "--mpi=pmix",
        "--no-container-mount-home",
        "--container-writable",  # Required on clusters using Enroot defaults, where ENROOT_ROOTFS_WRITABLE=no.
    ]

    if log_dir is not None:
        set_nemorun_home(log_dir)
    else:
        if os.environ.get("NEMORUN_HOME") is None:
            logger.warning(
                f"Logs will be written to {get_nemorun_home()}, which is probably not desired.  export NEMORUN_HOME in your shell environment or use the --log_dir argument"
            )

    perf_env = OFFLINE_BENCHMARK_ENV_VARS.copy()

    if wandb_key is not None:
        perf_env["WANDB_API_KEY"] = wandb_key

    if gpu.lower() == "gb200":
        perf_env["NCCL_NET_GDR_LEVEL"] = "PHB"  # For NCCL 2.25
        perf_env["NCCL_NET_GDR_C2C"] = "1"  # For NCCL 2.26

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        perf_env["NEMO_HOME"] = nemo_home
        mounts.extend([f"{nemo_home}:{nemo_home}"])
    if hf_token is not None:
        # Enable authenticated online access for tokenizer/config paths.
        perf_env.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})
    if offline:
        # Disable HF Hub network calls. Requires a pre-populated local HF cache.
        perf_env["HF_HUB_OFFLINE"] = "1"

    perf_env.update(custom_env_vars)
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

    log_repo_status_cmd = "bash /opt/Megatron-Bridge/docker/common/print_sha.sh /nemo_run/configs/repo_status.json"
    custom_bash_cmds.append(log_repo_status_cmd)

    numa_divisor = 2 if gpu.lower() in ["gb200", "gb300"] else 4
    numa_cmd = f"numactl --cpunodebind=$((SLURM_LOCALID/{numa_divisor})) --membind=$((SLURM_LOCALID/{numa_divisor}))"
    if gpu.lower() in ["b300"] and enable_pct_binding:
        numa_cmd += " -C $((SLURM_LOCALID * 16)),$((SLURM_LOCALID * 16 + 1))"
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
        env_vars=perf_env,
        container_env=sorted(perf_env.keys()),
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(include_submodules=False) if packager == "git" else run.Packager(),
        segment=segment,
        network=network,
        launcher=launcher,
        additional_parameters=additional_slurm_params,
    )

    return executor


def kubeflow_executor(
    namespace: str,
    nodes: int,
    num_gpus_per_node: int,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    train_job_basename: Optional[str] = None,
    volumes: List[Dict[str, Any]] = None,
    volume_mounts: List[Dict[str, Any]] = None,
    workdir_pvc: Optional[str] = None,
    workdir_pvc_path: str = "/nemo_run",
    workdir_local_path: Optional[str] = None,
    image_pull_secrets: List[str] = None,
    wandb_key: str = None,
    hf_token: str = None,
    custom_env_vars: Dict[str, str] = None,
    tolerations: Optional[List[Dict[str, Any]]] = None,
    affinity: Optional[Dict[str, Any]] = None,
    env_list: Optional[List[Dict[str, Any]]] = None,
    extra_resource_requests: Optional[Dict[str, str]] = None,
    extra_resource_limits: Optional[Dict[str, str]] = None,
    pod_spec_overrides: Optional[Dict[str, Any]] = None,
    container_kwargs: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, Any]] = None,
    pod_annotations: Optional[Dict[str, Any]] = None,
) -> run.KubeflowExecutor:
    """Build a Kubeflow Training Operator executor.

    Wires NeMo container settings, secret-backed env vars (``env_list``), GPU lease
    affinity / toleration metadata, and EFA-style extra resource requests through to
    ``run.KubeflowExecutor``. The keyword arguments after ``custom_env_vars`` are pass-through
    to the underlying executor — they map onto fields of the same name on
    ``nemo_run.core.execution.kubeflow.KubeflowExecutor`` and are required when scheduling
    onto lease-allocated nodes (tolerations + node affinity) or onto AWS clusters with
    EFA devices (extra_resource_requests / extra_resource_limits + privileged container).

    Args:
        namespace: Kubernetes namespace for the TrainJob.
        nodes: Number of replica nodes.
        num_gpus_per_node: GPUs requested per replica.
        container_image: Training container image.
        volumes: Pod-level volumes.
        volume_mounts: Container volume mounts.
        workdir_pvc: PVC to sync the run workdir into.
        workdir_pvc_path: Mount path for the workdir PVC inside the container.
        workdir_local_path: Local directory whose contents nemo-run's
            ``KubeflowExecutor.package()`` rsyncs into the workdir PVC via
            a temporary alpine pod before launch. Used to overlay a
            ``--mbridge-ref`` checkout onto the trainer's
            ``/opt/Megatron-Bridge`` without rebuilding the image.
        image_pull_secrets: Image pull secret names.
        wandb_key: WANDB_API_KEY to inject as a flat env var (use ``env_list`` for
            ``secretKeyRef`` instead in production).
        hf_token: HF_TOKEN to inject as a flat env var (use ``env_list`` for
            ``secretKeyRef`` instead in production).
        custom_env_vars: Additional flat env vars merged into the container env.
        tolerations: Pod tolerations (e.g. ``gpu-wrangler.nvidia.com/lease``).
        affinity: Pod affinity dict (e.g. node affinity onto lease-allocated nodes).
        env_list: Kubernetes ``EnvVar`` dicts (supports ``valueFrom.secretKeyRef``).
        extra_resource_requests: Extra container resource requests (e.g. EFA).
        extra_resource_limits: Extra container resource limits.
        pod_spec_overrides: Dict merged into the pod spec.
        container_kwargs: Extra container fields (e.g. ``securityContext``).
        labels: Pod labels.
        pod_annotations: Annotations applied to the trainer pod template metadata
            (e.g. ``networking.gke.io/interfaces`` for GKE RDMA NIC attachment).

    Returns:
        Configured ``run.KubeflowExecutor`` instance.
    """
    # K8s/Kubeflow jobs deliberately do NOT inherit the Slurm orchestration
    # defaults in OFFLINE_BENCHMARK_ENV_VARS. Recipe-owned process settings are applied by
    # the rank-local bootstrap; the cluster supplies fabric
    # tuning explicitly via KUBEFLOW_ENV_LIST_JSON (-> custom_env_vars / env_list).
    env_vars: Dict[str, str] = {}
    if wandb_key is not None:
        env_vars["WANDB_API_KEY"] = wandb_key
    if hf_token is not None:
        env_vars.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})
    if custom_env_vars:
        env_vars.update(custom_env_vars)

    # Tag the TrainJob + its pods with their CI origin so a stray/orphaned job can
    # be traced back to (and cancelled via) its GitLab pipeline/job — e.g.
    # `kubectl get trainjob -L nemo-ci/job-id`. K8s label values must be <=63 chars
    # of [A-Za-z0-9._-]; the CI ids are numeric, so they are safe as-is.
    ci_labels = {
        f"nemo-ci/{name}": os.environ[env]
        for name, env in (
            ("pipeline-id", "CI_PIPELINE_ID"),
            ("job-id", "CI_JOB_ID"),
            ("parent-pipeline-id", "PARENT_PIPELINE_ID"),
        )
        if os.environ.get(env)
    }
    labels = {**ci_labels, **(labels or {})}

    executor = run.KubeflowExecutor(
        # Launch each replica's entrypoint under torchrun so the torch-distributed
        # ClusterTrainingRuntime's rendezvous env (MASTER_ADDR, nnodes, nproc) is
        # consumed and a single WORLD_SIZE = num_nodes * gpus_per_node process
        # group is formed. Without this the entrypoint runs as a lone python
        # process per node (WORLD_SIZE=1), failing data-parallel sizing.
        launcher=run.Torchrun(),
        # Pin the Kubeflow Trainer runtime + per-replica CPU/memory requests to
        # the same values the verified standalone launch (real_trainjob.py) uses.
        runtime_ref="torch-distributed",
        cpu_requests="8",
        memory_requests="32Gi",
        namespace=namespace,
        image=container_image,
        num_nodes=nodes,
        gpus_per_node=num_gpus_per_node,
        volumes=volumes or [],
        volume_mounts=volume_mounts or [],
        workdir_pvc=workdir_pvc,
        workdir_pvc_path=workdir_pvc_path,
        workdir_local_path=workdir_local_path,
        train_job_basename=train_job_basename,
        image_pull_secrets=image_pull_secrets or [],
        env_vars=env_vars,
        env_list=env_list or [],
        tolerations=tolerations or [],
        affinity=affinity or {},
        extra_resource_requests=extra_resource_requests or {},
        extra_resource_limits=extra_resource_limits or {},
        pod_spec_overrides=pod_spec_overrides or {},
        container_kwargs=container_kwargs or {},
        labels=labels,
        # Mirror the CI-origin labels onto the trainer pods too, so both
        # `kubectl get trainjob -l` and `kubectl get pods -l` resolve the origin.
        pod_labels=labels,
        # pod_annotations land on the trainer pod template metadata (e.g. GKE
        # networking.gke.io/interfaces to attach the RDMA NICs for gIB).
        pod_annotations=pod_annotations or {},
        # include_submodules=True: KubeflowExecutor.package() ships the packager
        # tarball to <workdir_pvc_path>/<user>/code, which the launcher overlays
        # onto /opt/Megatron-Bridge in the trainer container. The trainer
        # needs both mbridge AND the pinned 3rdparty/Megatron-LM submodule,
        # so the tarball must include both. (On SLURM where the host
        # checkout is bind-mounted directly via CUSTOM_MOUNTS, submodules
        # come from the host filesystem and the packager output is unused
        # for the /opt/Megatron-Bridge path — so the extra archive bytes
        # are harmless cost there.)
        packager=run.GitArchivePackager(include_submodules=True),
    )
    return executor
