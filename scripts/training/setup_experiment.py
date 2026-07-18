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
import sys
from pathlib import Path

import nemo_run as run
from nemo_run.config import get_nemorun_home


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from performance_recipe import PerformanceRecipeMetadata, selected_performance_recipe  # noqa: E402


CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")
TRAINING_ENTRYPOINT = CONTAINER_REPO_ROOT / "scripts/training/run_recipe.py"
PERFORMANCE_ENTRYPOINT = CONTAINER_REPO_ROOT / "scripts/performance/run_script.py"
TRAINING_LAUNCH_ENV = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "False",
    "NCCL_NVLS_ENABLE": "0",
    "NVTE_NORM_FWD_USE_CUDNN": "1",
    "NVTE_NORM_BWD_USE_CUDNN": "1",
    "TORCH_NCCL_HIGH_PRIORITY": "1",
    "HF_HUB_OFFLINE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_GRAPH_REGISTER": "0",
}
PERFORMANCE_GB200_LAUNCH_ENV = {
    "NCCL_NET_GDR_LEVEL": "PHB",
    "NCCL_NET_GDR_C2C": "1",
}
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PERFORMANCE_SLURM_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    bash /opt/Megatron-Bridge/docker/common/print_sha.sh /nemo_run/configs/repo_status.json
fi
command -v numactl >/dev/null || { echo "numactl is required for performance recipe CPU binding" >&2; exit 1; }
exec {{ numa_command | safe }} {{ command | safe }}
"""


def _build_parser() -> argparse.ArgumentParser:
    """Build the lightweight head-node parser."""
    parser = argparse.ArgumentParser(
        description="Launch Megatron Bridge library or exact performance recipes through Slurm.",
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
      --env HF_TOKEN \\
      --recipe qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config

Library arguments are forwarded to run_recipe.py. Exact performance recipe
names are passed unchanged to the compatibility performance runtime.
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
        "--offline",
        action="store_true",
        help="Run from a pre-populated Hugging Face cache without network access.",
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
        help="Render the Slurm submission without submitting it.",
    )
    return parser


def _parse_env(values: list[str]) -> list[str]:
    """Validate names inherited by the Slurm job and training container."""
    env_names: list[str] = []
    for name in values:
        if ENV_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"Invalid --env name '{name}'; expected a shell variable name and no value. "
                "Export the value before launching to keep it out of arguments."
            )
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
    performance_metadata: PerformanceRecipeMetadata | None = None,
    *,
    env_names: list[str] | None = None,
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
    env_names = env_names or []
    if args.offline and "HF_TOKEN" in env_names:
        raise ValueError("Choose either --env HF_TOKEN for online access or --offline, not both.")
    if performance_metadata is not None:
        requested_gpus = args.nodes * args.gpus_per_node
        if requested_gpus != performance_metadata.num_gpus:
            raise ValueError(
                f"Performance recipe requires exactly {performance_metadata.num_gpus} GPUs, but --nodes and "
                f"--gpus-per-node request {requested_gpus}."
            )


def _performance_numa_command(metadata: PerformanceRecipeMetadata) -> str:
    """Return the compatibility launcher's hardware-specific NUMA binding."""
    numa_divisor = 2 if metadata.hardware in {"gb200", "gb300"} else 4
    command = f"numactl --cpunodebind=$((SLURM_LOCALID/{numa_divisor})) --membind=$((SLURM_LOCALID/{numa_divisor}))"
    recipe_name = metadata.recipe_name
    pct_binding_enabled = metadata.hardware == "b300"
    pct_binding_enabled &= not ("_pretrain_" in recipe_name and recipe_name.startswith(("llama", "kimi_")))
    pct_binding_enabled &= not (recipe_name.startswith("nemotron_3_super_") and "_b300_bf16" in recipe_name)
    pct_binding_enabled &= not (
        recipe_name.startswith("deepseek_v3_") and not recipe_name.endswith("_large_scale_config")
    )
    if pct_binding_enabled:
        command += " -C $((SLURM_LOCALID * 16)),$((SLURM_LOCALID * 16 + 1))"
    return command


def _performance_segment(*, gpus_per_node: int, nodes: int) -> int | None:
    """Preserve Slurm segment sizing for four-GPU performance nodes."""
    if gpus_per_node != 4:
        return None
    if nodes <= 18:
        return nodes
    for segment in range(18, 0, -1):
        if nodes % segment == 0:
            return segment
    raise AssertionError("Every positive node count is divisible by one.")


def _performance_slurm_launcher(metadata: PerformanceRecipeMetadata) -> object:
    """Build the rank-local NUMA-prefixed Slurm template lazily."""
    from nemo_run.core.execution.launcher import SlurmTemplate

    return SlurmTemplate(
        template_inline=PERFORMANCE_SLURM_TEMPLATE,
        template_vars={"numa_command": _performance_numa_command(metadata)},
    )


def _task_environment(
    performance_metadata: PerformanceRecipeMetadata | None,
    *,
    inherited_env_names: list[str],
    offline: bool = False,
) -> dict[str, str]:
    """Build the rank-local environment without materializing inherited values."""
    python_paths = [
        str(CONTAINER_REPO_ROOT / "src"),
        str(CONTAINER_REPO_ROOT / "3rdparty/Megatron-LM"),
    ]
    if performance_metadata is not None:
        python_paths.insert(0, str(CONTAINER_REPO_ROOT / "scripts/performance"))
    environment = {"PYTHONPATH": f"{':'.join(python_paths)}:$PYTHONPATH"}
    launch_environment = dict(TRAINING_LAUNCH_ENV)
    if performance_metadata is not None and performance_metadata.hardware == "gb200":
        launch_environment.update(PERFORMANCE_GB200_LAUNCH_ENV)
    if "HF_TOKEN" in inherited_env_names and "TRANSFORMERS_OFFLINE" not in inherited_env_names:
        launch_environment["TRANSFORMERS_OFFLINE"] = "0"
    if offline:
        launch_environment["HF_HUB_OFFLINE"] = "1"
    for name, value in launch_environment.items():
        if name not in inherited_env_names:
            environment[name] = value
    return environment


def _build_executor(
    args: argparse.Namespace,
    env_names: list[str],
    mounts: list[str],
    *,
    performance_metadata: PerformanceRecipeMetadata | None = None,
) -> object:
    """Build a Slurm NeMo-Run executor."""
    gpu_kwargs = {} if args.no_gpu_resource_request else {"gpus_per_node": args.gpus_per_node}
    performance_kwargs = {}
    srun_args = list(args.srun_args)
    if performance_metadata is not None:
        performance_kwargs["launcher"] = _performance_slurm_launcher(performance_metadata)
        segment = _performance_segment(gpus_per_node=args.gpus_per_node, nodes=args.nodes)
        if segment is not None:
            performance_kwargs["segment"] = segment

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
        **performance_kwargs,
        **gpu_kwargs,
    )
    executor.container_image = args.container_image
    executor.container_mounts = mounts
    # Slurm inherits these values from the launcher environment. NeMo-Run receives
    # names only so secrets are not materialized into generated sbatch scripts.
    executor.env_vars = {}
    task_env_names = _task_environment(
        performance_metadata,
        inherited_env_names=env_names,
        offline=args.offline,
    )
    # Pyxis otherwise lets values baked into the image override the task
    # environment. Name every task variable here while keeping secret values in
    # the inherited Slurm environment only.
    executor.container_env = sorted(set(env_names) | set(task_env_names))
    executor.additional_parameters = {"export": ",".join(env_names) if env_names else "NIL"}
    executor.srun_args = srun_args
    return executor


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launch arguments and preserve all training arguments."""
    return _build_parser().parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and launch the selected training experiment."""
    args, training_args = parse_args(argv)
    performance_metadata = selected_performance_recipe(training_args)
    task_args = training_args
    entrypoint = TRAINING_ENTRYPOINT
    if performance_metadata is not None:
        entrypoint = PERFORMANCE_ENTRYPOINT

    env_names = _parse_env(args.env)
    mounts = _parse_mounts(args.mount)
    _validate_args(args, performance_metadata, env_names=env_names)
    executor = _build_executor(args, env_names, mounts, performance_metadata=performance_metadata)

    task = run.Script(
        path=str(entrypoint),
        entrypoint="python",
        env=_task_environment(
            performance_metadata,
            inherited_env_names=env_names,
            offline=args.offline,
        ),
        # NeMo-Run 0.10 joins Script arguments into an sbatch shell command.
        # Quote each value here so spaces and metacharacters remain one argument.
        args=[shlex.quote(argument) for argument in task_args],
    )
    experiment_name = args.experiment_name or "training"
    logger.info(
        "Training command: %s",
        shlex.join(["python", str(entrypoint), *task_args]),
    )
    logger.info("Forwarded environment variables: %s", ", ".join(env_names) or "none")
    logger.info("Container mounts: %s", ", ".join(mounts) or "none")
    if performance_metadata is not None:
        logger.info(
            "Performance recipe code is resolved from the container image; ensure it contains the same Bridge "
            "revision and exact recipe export as this launcher checkout."
        )

    with run.Experiment(experiment_name) as experiment:
        experiment.add(task, executor=executor, name="training")
        if args.submission_dry_run:
            experiment.dryrun()
            return
        experiment.run(detach=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
