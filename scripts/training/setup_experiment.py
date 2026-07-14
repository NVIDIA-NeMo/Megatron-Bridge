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
import re
import shlex
from pathlib import Path

import nemo_run as run
from nemo_run.config import get_nemorun_home


logger = logging.getLogger(__name__)

CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SENSITIVE_ARGUMENT_PATTERN = re.compile(
    r"(^|[._-])(credentials?|secret|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key|"
    r"authentication|authorization)([._-]|$)",
    re.IGNORECASE,
)
SENSITIVE_TOKEN_CONTEXT_PATTERN = re.compile(
    r"(^|[._-])(access|api|auth|bearer|comet|hf|refresh|secret|wandb)[._-]?token([._-]|$)",
    re.IGNORECASE,
)
TERMINAL_TOKEN_PATTERN = re.compile(r"(^|[._-])token$", re.IGNORECASE)
BENIGN_TERMINAL_TOKEN_PATTERN = re.compile(
    r"(^|[._-])(audio|bos|cls|decoder[_-]?start|eod|eos|image|image[_-]end|image[_-]start|insert[_-]start|"
    r"mask|pad|patch[_-]end|patch[_-]start|sep|set[_-]pad|unmask[_-]last|use[_-]cls|video)[._-]token$",
    re.IGNORECASE,
)
MOUNT_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9_./+@%=-]+$")


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
        help="Container mount as HOST or HOST:CONTAINER. HOST uses the same path in the container. May be repeated.",
    )
    execution.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable NAME to inherit into the container. May be repeated.",
    )
    execution.add_argument("--experiment-name", help="NeMo-Run experiment name.")
    execution.add_argument(
        "--submission-dry-run",
        action="store_true",
        help="Render the Slurm submission without submitting it.",
    )
    return parser


def _parse_env(values: list[str]) -> list[str]:
    """Validate names inherited by the Slurm job and training container."""
    env_names: list[str] = []
    for name in values:
        if "=" in name:
            raise ValueError("--env accepts NAME only; export the value before launching to keep it out of arguments.")
        if not ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}.")
        if name not in os.environ:
            raise ValueError(f"Environment variable '{name}' is not set in the launcher environment.")
        if name not in env_names:
            env_names.append(name)
    return env_names


def _validate_mount_path(path: str) -> None:
    """Reject ambiguous mount syntax before it reaches Pyxis or a shell script."""
    if not MOUNT_PATH_PATTERN.fullmatch(path):
        raise ValueError(f"Invalid mount path {path!r}; use an absolute path without spaces or shell metacharacters.")


def _parse_mounts(values: list[str]) -> list[str]:
    """Normalize explicit same-path and host-to-container mounts."""
    mounts: list[str] = []
    for value in values:
        if ":" in value:
            host_path, container_path = value.split(":", 1)
        else:
            host_path = value
            container_path = value
        _validate_mount_path(host_path)
        _validate_mount_path(container_path)
        mount = f"{host_path}:{container_path}"
        if mount not in mounts:
            mounts.append(mount)
    return mounts


def _validate_training_args(training_args: list[str]) -> None:
    """Keep credential-like values out of rendered commands and Slurm artifacts."""
    for argument in training_args:
        option_name = argument.split("=", 1)[0].lstrip("-+")
        has_sensitive_token = bool(
            SENSITIVE_TOKEN_CONTEXT_PATTERN.search(option_name)
            or (TERMINAL_TOKEN_PATTERN.search(option_name) and not BENIGN_TERMINAL_TOKEN_PATTERN.search(option_name))
        )
        if SENSITIVE_ARGUMENT_PATTERN.search(option_name) or has_sensitive_token:
            raise ValueError(
                f"Sensitive argument '{option_name}' is not allowed; pass credentials through an exported --env NAME."
            )


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


def _build_executor(args: argparse.Namespace, env_names: list[str], mounts: list[str]) -> object:
    """Build a Slurm NeMo-Run executor."""
    gpu_kwargs = {} if args.no_gpu_resource_request else {"gpus_per_node": args.gpus_per_node}
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
        packager=run.Packager(),
        **gpu_kwargs,
    )
    executor.container_image = args.container_image
    executor.container_mounts = mounts
    # Slurm inherits these values from the launcher environment. NeMo-Run receives
    # names only so secrets are not materialized into generated sbatch scripts.
    executor.env_vars = {}
    executor.container_env = env_names
    executor.additional_parameters = {"export": ",".join(env_names) if env_names else "NIL"}
    executor.srun_args = ["--mpi=pmix", "--no-container-mount-home", "--container-writable"]
    return executor


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launch arguments and preserve all training arguments."""
    return _build_parser().parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and launch the selected training experiment."""
    args, training_args = parse_args(argv)
    _validate_args(args)
    _validate_training_args(training_args)

    env_names = _parse_env(args.env)
    mounts = _parse_mounts(args.mount)
    executor = _build_executor(args, env_names, mounts)

    task = run.Script(
        path=str(CONTAINER_REPO_ROOT / "scripts/training/run_recipe.py"),
        entrypoint="python",
        env={
            "PYTHONPATH": f"{CONTAINER_REPO_ROOT}/src:{CONTAINER_REPO_ROOT}/3rdparty/Megatron-LM:$PYTHONPATH",
        },
        # NeMo-Run 0.10 joins Script arguments into an sbatch shell command.
        # Quote each value here so spaces and metacharacters remain one argument.
        args=[shlex.quote(argument) for argument in training_args],
    )
    experiment_name = args.experiment_name or "training"
    logger.info(
        "Training command: %s",
        shlex.join(["python", str(CONTAINER_REPO_ROOT / "scripts/training/run_recipe.py"), *training_args]),
    )
    logger.info("Forwarded environment variables: %s", ", ".join(env_names) or "none")
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
