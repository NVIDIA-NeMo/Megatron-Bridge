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
"""Submit a recipe-based training experiment through Slurm."""

import argparse
import logging
import os
from pathlib import Path

import nemo_run as run
from nemo_run.config import get_nemorun_home


logger = logging.getLogger(__name__)

CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")


def _build_parser() -> argparse.ArgumentParser:
    """Build the lightweight head-node parser."""
    parser = argparse.ArgumentParser(
        description="Launch Megatron Bridge training; unknown arguments are forwarded to run_recipe.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Training examples:
  ./scripts/training/train.sh --nodes 1 --gpus-per-node 8 \\
      --account ACCOUNT --partition PARTITION --container-image IMAGE \\
      --env HF_TOKEN --mount /shared/data \\
      --recipe gpt_oss_20b_sft_config --mode sft

Arguments not owned by this launcher are forwarded unchanged to run_recipe.py.
""",
    )
    execution = parser.add_argument_group("Execution")
    execution.add_argument("--nodes", type=int, default=1, help="Number of nodes.")
    execution.add_argument(
        "--gpus-per-node",
        "--gpus_per_node",
        type=int,
        dest="gpus_per_node",
        help="GPUs per node.",
    )
    execution.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT"), help="Slurm account.")
    execution.add_argument("--partition", default=os.environ.get("SLURM_PARTITION"), help="Slurm partition.")
    execution.add_argument("--time", default="04:00:00", help="Slurm time limit.")
    execution.add_argument("--gres", help="Optional Slurm GRES value.")
    execution.add_argument(
        "--container-image",
        default=os.environ.get("CONTAINER_IMAGE"),
        help="Slurm container image; defaults to CONTAINER_IMAGE.",
    )
    execution.add_argument(
        "--mount",
        action="append",
        default=[],
        help="Container mount as HOST or HOST:CONTAINER. HOST uses the same path in the container. May be repeated.",
    )
    execution.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable as NAME or NAME=VALUE. NAME inherits its launcher value. May be repeated.",
    )
    execution.add_argument(
        "--packager",
        choices=["none", "git"],
        default="none",
        help="Code packaging method for Slurm.",
    )
    execution.add_argument("--experiment-name", help="NeMo-Run experiment name.")
    execution.add_argument(
        "--submission-dry-run",
        action="store_true",
        help="Render the Slurm submission without submitting it.",
    )
    return parser


def _parse_env(values: list[str]) -> dict[str, str]:
    """Resolve explicit environment values for the training container."""
    env_vars: dict[str, str] = {}
    for value in values:
        if "=" in value:
            name, env_value = value.split("=", 1)
        else:
            name = value
            if name not in os.environ:
                raise ValueError(f"Environment variable '{name}' is not set; pass --env {name}=VALUE instead.")
            env_value = os.environ[name]
        if not name:
            raise ValueError("Environment variable names cannot be empty.")
        env_vars[name] = env_value
    return env_vars


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
    """Validate launcher requirements before creating an executor."""
    if args.nodes < 1:
        raise ValueError("--nodes must be at least 1.")
    if args.gpus_per_node is None or args.gpus_per_node < 1:
        raise ValueError("--gpus-per-node is required and must be at least 1.")
    if not args.account or not args.partition:
        raise ValueError("Slurm execution requires --account and --partition.")
    if not args.container_image:
        raise ValueError("Slurm execution requires --container-image or CONTAINER_IMAGE.")


def _build_executor(args: argparse.Namespace, env_vars: dict[str, str], mounts: list[str]) -> object:
    """Build a Slurm NeMo-Run executor."""
    packager = run.GitArchivePackager(include_submodules=False) if args.packager == "git" else run.Packager()
    executor = run.SlurmExecutor(
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.gpus_per_node,
        mem="0",
        exclusive=True,
        time=args.time,
        gres=args.gres,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        packager=packager,
    )
    executor.container_image = args.container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.container_env = sorted(env_vars)
    executor.srun_args = ["--mpi=pmix", "--no-container-mount-home", "--container-writable"]
    return executor


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launch arguments and preserve all training arguments."""
    return _build_parser().parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and launch the selected training experiment."""
    args, training_args = parse_args(argv)
    _validate_args(args)

    env_vars = _parse_env(args.env)
    mounts = _parse_mounts(args.mount)
    executor = _build_executor(args, env_vars, mounts)

    task = run.Script(
        path=str(CONTAINER_REPO_ROOT / "scripts/training/run_recipe.py"),
        entrypoint="python",
        env={
            "PYTHONPATH": f"{CONTAINER_REPO_ROOT}/src:{CONTAINER_REPO_ROOT}/3rdparty/Megatron-LM:$PYTHONPATH",
        },
        args=training_args,
    )
    experiment_name = args.experiment_name or "training"
    logger.info("Training command: %s", " ".join(task.to_command()))
    logger.info("Forwarded environment variables: %s", ", ".join(sorted(env_vars)) or "none")
    logger.info("Container mounts: %s", ", ".join(mounts) or "none")

    with run.Experiment(experiment_name) as experiment:
        experiment.add(task, executor=executor, name="training")
        if args.submission_dry_run:
            experiment.dryrun()
            return
        experiment.run(detach=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
