#!/usr/bin/env python3
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
"""Submit Bridge-backed offline inference and comparison through Slurm."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
from pathlib import Path

import nemo_run as run
from nemo_run.config import get_nemorun_home
from torchx.specs.api import AppState


logger = logging.getLogger(__name__)

CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")
INFERENCE_TASKS = {
    "text-generation": Path("scripts/inference/text_generation.py"),
    "vlm-generation": Path("examples/conversion/hf_to_megatron_generate_vlm.py"),
    "model-comparison": Path("examples/conversion/compare_hf_and_megatron/compare.py"),
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the lightweight head-node parser."""
    parser = argparse.ArgumentParser(
        description="Launch Megatron Bridge offline inference or model comparison through Slurm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Example:
  ./scripts/inference/infer.sh --nodes 1 --gpus-per-node 1 \\
      --account ACCOUNT --partition PARTITION --container-image IMAGE \\
      --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge --env HF_TOKEN \\
      --hf-model-path meta-llama/Llama-3.2-1B \\
      --prompt "Megatron Bridge inference is" --max_new_tokens 32

Use --task vlm-generation for multimodal generation or --task model-comparison
for a one-step Hugging Face/Megatron comparison. Arguments not owned by this
launcher are forwarded unchanged to the selected repository entry point.
""",
    )
    execution = parser.add_argument_group("Execution")
    execution.add_argument(
        "--task",
        choices=tuple(INFERENCE_TASKS),
        default="text-generation",
        help="Repository inference task to launch (default: text-generation).",
    )
    execution.add_argument("--nodes", type=int, default=1, help="Number of Slurm nodes (default: 1).")
    execution.add_argument(
        "--gpus-per-node",
        "--gpus_per_node",
        type=int,
        dest="gpus_per_node",
        help="GPUs and inference tasks per node.",
    )
    execution.add_argument("--cpus-per-task", type=int, help="CPUs allocated to each inference task.")
    execution.add_argument("--mem", help="Optional Slurm memory request, such as 64G.")
    execution.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT"), help="Slurm account.")
    execution.add_argument("--partition", default=os.environ.get("SLURM_PARTITION"), help="Slurm partition.")
    execution.add_argument("--time", default="01:00:00", help="Slurm time limit (default: 01:00:00).")
    execution.add_argument("--gres", help="Optional Slurm GRES value.")
    execution.add_argument(
        "--exclusive",
        action="store_true",
        help="Request exclusive nodes; by default Slurm may share nodes for small inference jobs.",
    )
    execution.add_argument(
        "--no-gpu-resource-request",
        action="store_true",
        help="Do not emit a Slurm GPU/GRES request on clusters that allocate whole GPU nodes implicitly.",
    )
    execution.add_argument(
        "--container-image",
        default=os.environ.get("CONTAINER_IMAGE"),
        help="Slurm container image; defaults to CONTAINER_IMAGE.",
    )
    execution.add_argument(
        "--mount",
        action="append",
        default=[],
        help="Container mount as HOST or HOST:CONTAINER; may be repeated.",
    )
    execution.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable NAME to inherit; may be repeated. Values are not accepted.",
    )
    execution.add_argument(
        "--srun-arg",
        action="append",
        default=[],
        dest="srun_args",
        metavar="ARG",
        help="Additional cluster-specific argument passed to srun; may be repeated. Use --srun-arg=--flag.",
    )
    execution.add_argument("--experiment-name", help="NeMo-Run experiment name.")
    execution.add_argument(
        "--submission-dry-run",
        "--dry-run",
        action="store_true",
        dest="submission_dry_run",
        help="Render the NeMo-Run Slurm submission without submitting it.",
    )
    execution.add_argument(
        "--detach",
        action="store_true",
        help="Return after Slurm accepts the job instead of waiting and tailing generation output.",
    )
    return parser


def _parse_env(values: list[str]) -> list[str]:
    """Validate inherited environment variable names without materializing values."""
    env_names: list[str] = []
    for name in values:
        if "=" in name:
            raise ValueError("--env accepts NAME only; export the value before launching to keep it out of arguments.")
        if name not in os.environ:
            raise ValueError(f"Environment variable '{name}' is not set in the launcher environment.")
        if name not in env_names:
            env_names.append(name)
    return env_names


def _parse_mounts(values: list[str]) -> list[str]:
    """Normalize explicit same-path and host-to-container mounts."""
    mounts: list[str] = []
    for value in values:
        if ":" in value:
            host_path, container_path = value.split(":", 1)
        else:
            host_path = value
            container_path = value
        if not host_path or not container_path:
            raise ValueError(f"Invalid --mount value '{value}'; expected HOST or HOST:CONTAINER.")
        mount = f"{host_path}:{container_path}"
        if mount not in mounts:
            mounts.append(mount)
    return mounts


def _validate_args(args: argparse.Namespace) -> None:
    """Validate Slurm resources before creating an executor."""
    if args.nodes < 1:
        raise ValueError("--nodes must be at least 1.")
    if args.gpus_per_node is None or args.gpus_per_node < 1:
        raise ValueError("--gpus-per-node is required and must be at least 1.")
    if args.cpus_per_task is not None and args.cpus_per_task < 1:
        raise ValueError("--cpus-per-task must be at least 1.")
    if any(not value.strip() for value in args.srun_args):
        raise ValueError("--srun-arg values must not be empty.")
    if not args.account or not args.partition:
        raise ValueError("Slurm execution requires --account and --partition.")
    if not args.container_image:
        raise ValueError("Slurm execution requires --container-image or CONTAINER_IMAGE.")


def _build_executor(args: argparse.Namespace, env_names: list[str], mounts: list[str]) -> object:
    """Build the srun-native NeMo-Run Slurm executor."""
    gpu_kwargs = {} if args.no_gpu_resource_request else {"gpus_per_node": args.gpus_per_node}
    # Slurm's --export=NIL removes the site PATH used to find scontrol and srun
    # on clusters where they are not installed in /usr/bin. Always inherit PATH
    # for the generated batch script, but expose it to the container only when
    # the user explicitly requests --env PATH.
    batch_env_names = ["PATH", *(name for name in env_names if name != "PATH")]
    executor = run.SlurmExecutor(
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.gpus_per_node,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        exclusive=True if args.exclusive else None,
        time=args.time,
        gres=args.gres,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        packager=run.Packager(),
        container_image=args.container_image,
        container_mounts=mounts,
        container_env=env_names,
        additional_parameters={"export": ",".join(batch_env_names)},
        srun_args=args.srun_args,
        **gpu_kwargs,
    )
    # Values are inherited by Slurm and selected by name for the container;
    # keeping env_vars empty prevents NeMo Run from serializing secrets.
    executor.env_vars = {}
    return executor


def _build_task(task_name: str, inference_args: list[str]) -> run.Script:
    """Build the selected Bridge inference task for the submitted container."""
    task_path = CONTAINER_REPO_ROOT / INFERENCE_TASKS[task_name]
    return run.Script(
        path=str(task_path),
        entrypoint="python",
        env={
            "PYTHONPATH": f"{CONTAINER_REPO_ROOT}/src:{CONTAINER_REPO_ROOT}/3rdparty/Megatron-LM:$PYTHONPATH",
        },
        # NeMo-Run 0.10 joins Script arguments into an sbatch shell command.
        # Quote each value so prompts and paths with shell metacharacters remain one argument.
        args=[shlex.quote(argument) for argument in inference_args],
    )


def _raise_on_failed_tasks(experiment: run.Experiment) -> None:
    """Raise when a synchronous NeMo-Run inference task did not succeed."""
    unsuccessful = [f"{job.id}={job.state}" for job in experiment.jobs if job.state != AppState.SUCCEEDED]
    if unsuccessful:
        raise RuntimeError(f"Inference failed: {', '.join(unsuccessful)}")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launch arguments and preserve all text-generation arguments."""
    return _build_parser().parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and submit one Bridge text-generation experiment."""
    args, inference_args = parse_args(argv)
    _validate_args(args)
    env_names = _parse_env(args.env)
    mounts = _parse_mounts(args.mount)
    executor = _build_executor(args, env_names, mounts)
    task = _build_task(args.task, inference_args)
    task_path = CONTAINER_REPO_ROOT / INFERENCE_TASKS[args.task]

    logger.info(
        "Inference command: %s",
        shlex.join(["python", str(task_path), *inference_args]),
    )
    logger.info("Forwarded environment variables: %s", ", ".join(env_names) or "none")
    logger.info("Container mounts: %s", ", ".join(mounts) or "none")

    with run.Experiment(args.experiment_name or "inference") as experiment:
        experiment.add(task, executor=executor, name=args.task)
        if args.submission_dry_run:
            experiment.dryrun()
            return
        experiment.run(detach=args.detach, tail_logs=not args.detach)
    if not args.detach:
        _raise_on_failed_tasks(experiment)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
