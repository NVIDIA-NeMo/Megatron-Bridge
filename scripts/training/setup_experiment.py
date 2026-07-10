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
"""Submit a recipe-based training experiment locally or through Slurm."""

import argparse
import logging
import os
from pathlib import Path

import nemo_run as run


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")
INHERITED_ENV_VARS = (
    "DCLM_CACHE",
    "DCLM_DATA_DIR",
    "DCLM_DATA_PREFIX",
    "HF_HOME",
    "HF_TOKEN",
    "NEMO_DATASETS_CACHE",
    "NEMO_HOME",
    "UV_CACHE_DIR",
    "WANDB_API_KEY",
)
PATH_ENV_VARS = (
    "DCLM_CACHE",
    "DCLM_DATA_DIR",
    "DCLM_DATA_PREFIX",
    "HF_HOME",
    "NEMO_DATASETS_CACHE",
    "NEMO_HOME",
    "UV_CACHE_DIR",
)
PATH_OPTIONS = (
    "--dataset-cache",
    "--dataset-path",
    "--from",
    "--load-dir",
    "--pretrained-checkpoint",
    "--save-dir",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the lightweight head-node parser."""
    parser = argparse.ArgumentParser(
        description="Launch Megatron Bridge training; unknown arguments are forwarded to run_recipe.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Training examples:
  ./scripts/training/train.sh --local --gpus-per-node 1 \\
      --recipe llama32_1b_pretrain_config --mode pretrain --dataset mock

  ./scripts/training/train.sh --nodes 2 --gpus-per-node 8 \\
      --account ACCOUNT --partition PARTITION --container-image IMAGE \\
      --recipe gpt_oss_20b_pretrain_16gpu_h100_bf16_config \\
      --mode pretrain --dataset dclm --dataset-path /data/dclm

Full recipe mode needs no --model. Selector mode uses --family, --model,
--mode, --gpu-type, and the topology from --nodes/--gpus-per-node.
Trailing KEY=VALUE arguments override the resolved ConfigContainer.
""",
    )
    execution = parser.add_argument_group("Execution")
    execution.add_argument("--local", action="store_true", help="Run locally instead of submitting to Slurm.")
    execution.add_argument("--nodes", type=int, default=1, help="Number of nodes.")
    execution.add_argument(
        "--gpus-per-node",
        "--devices",
        type=int,
        dest="gpus_per_node",
        help="GPUs per node.",
    )
    execution.add_argument("--gpu-type", help="Recipe hardware selector, e.g. h100 or gb200.")
    execution.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT"), help="Slurm account.")
    execution.add_argument("--partition", default=os.environ.get("SLURM_PARTITION"), help="Slurm partition.")
    execution.add_argument("--time", default="04:00:00", help="Slurm time limit.")
    execution.add_argument("--gres", help="Optional Slurm GRES value.")
    execution.add_argument(
        "--container-image",
        default=os.environ.get("MB_CONTAINER_IMAGE") or os.environ.get("CONTAINER_IMAGE"),
        help="Slurm container image; defaults to MB_CONTAINER_IMAGE.",
    )
    execution.add_argument(
        "--mount",
        action="append",
        default=[],
        help="Container mount in host:container form. May be repeated.",
    )
    execution.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment override in KEY=VALUE form. May be repeated.",
    )
    execution.add_argument(
        "--packager",
        choices=["none", "git"],
        default="none",
        help="Code packaging method for Slurm.",
    )
    execution.add_argument("--experiment-name", help="NeMo-Run experiment name.")
    execution.add_argument("--dry-run", action="store_true", help="Render the launch without submitting it.")
    execution.add_argument(
        "--detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach after submission.",
    )
    execution.add_argument("--tail-logs", action="store_true", help="Tail logs for a non-detached experiment.")
    return parser


def _parse_env(values: list[str]) -> dict[str, str]:
    """Parse explicit environment values and inherit standard cache/auth variables."""
    env_vars = {name: os.environ[name] for name in INHERITED_ENV_VARS if os.environ.get(name)}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --env value '{value}'; expected KEY=VALUE.")
        name, env_value = value.split("=", 1)
        if not name:
            raise ValueError("Environment variable names cannot be empty.")
        env_vars[name] = env_value
    return env_vars


def _option_values(arguments: list[str], option_names: tuple[str, ...]) -> list[str]:
    """Return values passed through a selected set of command-line options."""
    values: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        matched = False
        for option in option_names:
            if argument == option and index + 1 < len(arguments):
                values.append(arguments[index + 1])
                index += 2
                matched = True
                break
            if argument.startswith(f"{option}="):
                values.append(argument.split("=", 1)[1])
                index += 1
                matched = True
                break
        if not matched:
            index += 1
    return values


def _mount_source(path_value: str) -> Path | None:
    """Return the existing host path that should be mounted for a path-like value."""
    path = Path(path_value).expanduser()
    if path.exists():
        return path if path.is_dir() else path.parent
    if Path(f"{path}.bin").exists() or Path(f"{path}.idx").exists():
        return path.parent
    if path.is_absolute():
        for parent in path.parents:
            if parent.exists():
                return parent
    return None


def _discover_mounts(training_args: list[str], explicit_mounts: list[str]) -> list[str]:
    """Add same-path mounts for dataset, cache, checkpoint, and output paths."""
    mounts = list(explicit_mounts)
    path_values = _option_values(training_args, PATH_OPTIONS)
    path_values.extend(os.environ[name] for name in PATH_ENV_VARS if os.environ.get(name))
    for value in path_values:
        source = _mount_source(value)
        if source is None:
            continue
        mount = f"{source}:{source}"
        if mount not in mounts:
            mounts.append(mount)
    return mounts


def _has_option(arguments: list[str], names: tuple[str, ...]) -> bool:
    """Return whether any option is present in a forwarded argument list."""
    return any(argument in names or any(argument.startswith(f"{name}=") for name in names) for argument in arguments)


def _inject_selector_topology(
    training_args: list[str],
    *,
    nodes: int,
    gpus_per_node: int,
    gpu_type: str | None,
) -> list[str]:
    """Supply selector topology while leaving full recipe invocations untouched."""
    forwarded = list(training_args)
    uses_recipe = _has_option(forwarded, ("--recipe",))
    uses_model = _has_option(forwarded, ("--model", "--model-recipe-name", "--model_recipe_name", "-mr"))
    if uses_recipe or not uses_model:
        return forwarded
    if not _has_option(forwarded, ("--gpus", "--num-gpus", "--num_gpus", "-ng")):
        forwarded.extend(["--gpus", str(nodes * gpus_per_node)])
    if not _has_option(forwarded, ("--gpu", "-g")):
        if gpu_type is None:
            raise ValueError("Selector mode requires --gpu-type; full --recipe mode does not.")
        forwarded.extend(["--gpu", gpu_type])
    return forwarded


def _training_value(arguments: list[str], option: str, default: str) -> str:
    """Read one training option for experiment naming."""
    values = _option_values(arguments, (option,))
    return values[-1] if values else default


def _experiment_name(arguments: list[str], explicit_name: str | None) -> str:
    """Build a stable experiment name from the public training selection."""
    if explicit_name:
        return explicit_name
    recipe = _training_value(arguments, "--recipe", "")
    model = _training_value(arguments, "--model", "training")
    mode = _training_value(arguments, "--mode", _training_value(arguments, "--task", "auto"))
    dataset = _training_value(arguments, "--dataset", "recipe-data")
    selection = recipe.removesuffix("_config") if recipe else model
    return f"{selection}-{mode}-{dataset}".replace("_", "-")


def _validate_args(args: argparse.Namespace, training_args: list[str]) -> None:
    """Validate launcher requirements before creating an executor."""
    if args.nodes < 1:
        raise ValueError("--nodes must be at least 1.")
    if args.gpus_per_node is None or args.gpus_per_node < 1:
        raise ValueError("--gpus-per-node is required and must be at least 1.")
    uses_recipe = _has_option(training_args, ("--recipe",))
    uses_model = _has_option(training_args, ("--model", "--model-recipe-name", "--model_recipe_name", "-mr"))
    if uses_recipe and uses_model:
        raise ValueError("--recipe already identifies the model; do not also pass --model.")
    if not uses_recipe and not uses_model:
        raise ValueError("Pass --recipe, or pass --family/--model/--mode to select a training recipe.")
    if not args.local:
        if not args.account or not args.partition:
            raise ValueError("Slurm execution requires --account and --partition.")
        if not args.container_image:
            raise ValueError("Slurm execution requires --container-image or MB_CONTAINER_IMAGE.")


def _build_executor(args: argparse.Namespace, env_vars: dict[str, str], mounts: list[str]) -> object:
    """Build a local or Slurm NeMo-Run executor."""
    if args.local:
        executor = run.LocalExecutor(ntasks_per_node=args.gpus_per_node, launcher="torchrun")
        executor.env_vars = env_vars
        return executor

    packager = run.GitArchivePackager(include_submodules=False) if args.packager == "git" else run.Packager()
    executor = run.SlurmExecutor(
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.gpus_per_node,
        gpus_per_node=args.gpus_per_node,
        mem="0",
        exclusive=True,
        time=args.time,
        gres=args.gres,
        tunnel=run.LocalTunnel(),
        packager=packager,
    )
    executor.container_image = args.container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.container_env = sorted(env_vars)
    executor.srun_args = ["--mpi=pmix", "--no-container-mount-home", "--container-writable"]
    return executor


def _ensure_experiment_succeeded(experiment_name: str) -> None:
    """Raise when any task in a synchronous experiment did not succeed."""
    statuses = run.Experiment.from_title(experiment_name).status(return_dict=True)
    failed_tasks = []
    for task_name, task_info in statuses.items():
        status = str(task_info.get("status", "UNKNOWN"))
        if status != "SUCCEEDED":
            failed_tasks.append(f"{task_name}={status}")
    if failed_tasks:
        failure_summary = ", ".join(failed_tasks)
        raise RuntimeError(f"Experiment '{experiment_name}' failed: {failure_summary}.")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launch arguments and preserve all training arguments."""
    return _build_parser().parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and launch the selected training experiment."""
    args, training_args = parse_args(argv)
    _validate_args(args, training_args)
    training_args = _inject_selector_topology(
        training_args,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        gpu_type=args.gpu_type,
    )

    env_vars = _parse_env(args.env)
    mounts = _discover_mounts(training_args, args.mount)
    executor = _build_executor(args, env_vars, mounts)

    script_path = (
        SCRIPT_DIR / "run_recipe.sh" if args.local else CONTAINER_REPO_ROOT / "scripts/training/run_recipe.sh"
    )
    task = run.Script(
        path=str(script_path),
        entrypoint="bash",
        args=training_args,
    )
    experiment_name = _experiment_name(training_args, args.experiment_name)
    logger.info("Training command: %s", " ".join(task.to_command()))
    logger.info("Inherited environment variables: %s", ", ".join(sorted(env_vars)) or "none")
    logger.info("Container mounts: %s", ", ".join(mounts) or "none")

    with run.Experiment(experiment_name) as experiment:
        experiment.add(task, executor=executor, name="training")
        if args.dry_run:
            experiment.dryrun()
            return
        experiment.run(detach=args.detach, tail_logs=args.tail_logs)
        if not args.detach:
            _ensure_experiment_succeeded(experiment_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
