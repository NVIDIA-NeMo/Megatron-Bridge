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
import shlex
import sys
from pathlib import Path
from typing import Any

import nemo_run as run
from nemo_run.config import get_nemorun_home


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recipe_metadata import (  # noqa: E402
    BenchmarkRecipeMetadata,
    selected_benchmark_recipe,
    validate_selected_benchmark_recipe,
)


CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")


def _build_parser() -> argparse.ArgumentParser:
    """Build the lightweight head-node parser."""
    parser = argparse.ArgumentParser(
        description="Launch Megatron Bridge library or exact benchmark recipes through Slurm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Training examples:
  ./scripts/training/train.sh --nodes 1 --gpus-per-node 8 \\
      --account ACCOUNT --partition PARTITION --container-image IMAGE \\
      --env HF_TOKEN --mount /shared/data \\
      --recipe gpt_oss_20b_sft_config --mode sft

  ./scripts/training/train.sh --nodes 2 --gpus-per-node 8 \\
      --account ACCOUNT --partition PARTITION --container-image IMAGE \\
      --recipe qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config --mode pretrain

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
    execution.add_argument(
        "--srun-arg",
        action="append",
        default=[],
        dest="srun_args",
        metavar="ARG",
        help="Additional cluster-specific argument passed to srun; may be repeated. Use --srun-arg=--flag.",
    )
    execution.add_argument(
        "-lmc",
        "--peak-mem-clk",
        "--peak_mem_clk",
        type=int,
        default=None,
        dest="peak_mem_clk",
        help=(
            "Lock the GPU memory clock to a fixed frequency in MHz once per node. "
            "Defaults to 4752 for VR200 benchmark recipes and is disabled otherwise; pass -1 to disable the default."
        ),
    )
    execution.add_argument("--experiment-name", help="NeMo-Run experiment name.")
    execution.add_argument(
        "--submission-dry-run",
        "--dry-run",
        action="store_true",
        dest="submission_dry_run",
        help="Render the Slurm submission without submitting it.",
    )
    execution.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the Slurm experiment to finish and stream its logs.",
    )
    return parser


def _parse_env(values: list[str]) -> list[str]:
    """Validate names inherited by the Slurm job and training container."""
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


def _validate_args(
    args: argparse.Namespace,
    benchmark_metadata: BenchmarkRecipeMetadata | None = None,
) -> None:
    """Validate launcher requirements before creating an executor."""
    if any(not value.strip() for value in args.srun_args):
        raise ValueError("--srun-arg values must not be empty.")
    if args.nodes < 1:
        raise ValueError("--nodes must be at least 1.")
    if args.gpus_per_node is None or args.gpus_per_node < 1:
        raise ValueError("--gpus-per-node is required and must be at least 1.")
    if not args.account or not args.partition:
        raise ValueError("Slurm execution requires --account and --partition.")
    if not args.container_image:
        raise ValueError("Slurm execution requires --container-image or CONTAINER_IMAGE.")
    if benchmark_metadata is not None:
        requested_gpus = args.nodes * args.gpus_per_node
        if requested_gpus != benchmark_metadata.num_gpus:
            raise ValueError(
                f"Benchmark recipe requires exactly {benchmark_metadata.num_gpus} GPUs, but --nodes and "
                f"--gpus-per-node request {requested_gpus}."
            )


def _task_environment() -> dict[str, str]:
    """Build source-agnostic rank-local environment defaults."""
    return {
        "PYTHONPATH": f"{CONTAINER_REPO_ROOT}/src:{CONTAINER_REPO_ROOT}/3rdparty/Megatron-LM:$PYTHONPATH",
    }


def _resolve_peak_mem_clk(
    requested_peak_mem_clk: int | None,
    benchmark_metadata: BenchmarkRecipeMetadata | None,
) -> int | None:
    """Resolve the explicit memory clock or the VR200 benchmark default."""
    if requested_peak_mem_clk == -1:
        return None
    if requested_peak_mem_clk is not None:
        return requested_peak_mem_clk
    if benchmark_metadata is not None and benchmark_metadata.hardware == "vr200":
        return 4752
    return None


def _configure_slurm_peak_mem_clk(executor: Any, peak_mem_clk: int | None) -> None:
    """Add a once-per-node GPU memory-clock lock to a Slurm executor."""
    if peak_mem_clk is None:
        return

    command = "\n".join(
        [
            "",
            "# Lock GPU memory clock",
            " ".join(
                [
                    "srun",
                    f"--ntasks={executor.nodes}",
                    "--ntasks-per-node=1",
                    "--output",
                    os.path.join(executor.tunnel.job_dir, "peak_mem_clock.out"),
                    "--error",
                    os.path.join(executor.tunnel.job_dir, "peak_mem_clock.err"),
                    "bash -c",
                    shlex.quote(f"sudo nvidia-smi -lmc {peak_mem_clk},{peak_mem_clk}"),
                ]
            ),
            "",
        ]
    )
    executor.setup_lines = f"{executor.setup_lines or ''}{command}"


def _build_executor(
    args: argparse.Namespace,
    env_names: list[str],
    mounts: list[str],
    *,
    task_environment: dict[str, str] | None = None,
) -> object:
    """Build a Slurm NeMo-Run executor."""
    gpu_kwargs = {} if args.no_gpu_resource_request else {"gpus_per_node": args.gpus_per_node}
    srun_args = list(args.srun_args)

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
    task_env_names = task_environment if task_environment is not None else _task_environment()
    # Pyxis otherwise lets values baked into the image override the task
    # environment. Name every task variable here while keeping secret values in
    # the inherited Slurm environment only.
    executor.container_env = sorted(set(env_names) | set(task_env_names))
    # Keep Slurm control commands available to the batch script without
    # forwarding the host PATH into the training container.
    slurm_env_names = list(dict.fromkeys(["PATH", *env_names]))
    executor.additional_parameters = {"export": ",".join(slurm_env_names)}
    executor.srun_args = srun_args
    return executor


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launch arguments and preserve all training arguments."""
    return _build_parser().parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and launch the selected training experiment."""
    args, training_args = parse_args(argv)
    benchmark_metadata = selected_benchmark_recipe(training_args)
    if benchmark_metadata is not None:
        validate_selected_benchmark_recipe(training_args, benchmark_metadata)
    _validate_args(args, benchmark_metadata)

    env_names = _parse_env(args.env)
    mounts = _parse_mounts(args.mount)
    task_environment = _task_environment()
    executor = _build_executor(args, env_names, mounts, task_environment=task_environment)
    peak_mem_clk = _resolve_peak_mem_clk(args.peak_mem_clk, benchmark_metadata)
    _configure_slurm_peak_mem_clk(executor, peak_mem_clk)

    task = run.Script(
        path=str(CONTAINER_REPO_ROOT / "scripts/training/run_recipe.py"),
        entrypoint="python",
        env=task_environment,
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
        experiment.run(detach=not args.wait, tail_logs=args.wait)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
