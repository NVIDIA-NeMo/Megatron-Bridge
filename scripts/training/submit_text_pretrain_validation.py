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

"""Submit the PR #4805 text-only pretrain validation matrix through train.sh."""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = SCRIPT_DIR / "text_pretrain_validation.json"
CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")


@dataclass(frozen=True)
class ValidationTarget:
    """One text-only pretrain validation target."""

    id: str
    family: str
    architecture: str
    hf_model: str
    revision: str
    params_b: float
    recipe: str
    tp: int
    pp: int
    ep: int
    overrides: tuple[str, ...] = ()

    @property
    def minimum_world_size(self) -> int:
        """Return the minimum world size implied by TP, PP, and EP."""
        return max(self.tp, self.ep) * self.pp


def load_manifest(path: Path) -> list[ValidationTarget]:
    """Load and validate the text-only target manifest."""
    with path.open(encoding="utf-8") as manifest_file:
        raw_targets = json.load(manifest_file)
    if not isinstance(raw_targets, list):
        raise ValueError("Validation manifest must contain a JSON list.")

    targets: list[ValidationTarget] = []
    seen_ids: set[str] = set()
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            raise ValueError("Every validation target must be a JSON object.")
        target = ValidationTarget(
            id=str(raw_target["id"]),
            family=str(raw_target["family"]),
            architecture=str(raw_target["architecture"]),
            hf_model="".join(raw_target["hf_model"])
            if isinstance(raw_target["hf_model"], list)
            else str(raw_target["hf_model"]),
            revision=str(raw_target["revision"]).replace("-", ""),
            params_b=float(raw_target["params_b"]),
            recipe=str(raw_target["recipe"]),
            tp=int(raw_target["tp"]),
            pp=int(raw_target["pp"]),
            ep=int(raw_target["ep"]),
            overrides=tuple(str(value) for value in raw_target.get("overrides", [])),
        )
        if target.id in seen_ids:
            raise ValueError(f"Duplicate validation target id: {target.id}")
        if any(value < 1 for value in (target.tp, target.pp, target.ep)):
            raise ValueError(f"Parallelism values must be positive for {target.id}.")
        if not re.fullmatch(r"[0-9a-f]{40}", target.revision):
            raise ValueError(f"Hugging Face revision must be an immutable SHA for {target.id}.")
        seen_ids.add(target.id)
        targets.append(target)
    return targets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit text-only 100-step DCLM pretrain validations through scripts/training/train.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", action="append", default=[], help="Target id to submit; repeat as needed.")
    parser.add_argument("--limit", type=int, help="Submit only the first N selected targets.")
    parser.add_argument("--submit", action="store_true", help="Submit jobs; otherwise only render commands.")
    parser.add_argument("--submission-dry-run", action="store_true", help="Ask NeMo-Run to render Slurm scripts.")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument(
        "--no-gpu-resource-request",
        action="store_true",
        help="Use whole-node implicit GPUs instead of emitting a Slurm GPU request.",
    )
    parser.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT"), required=False)
    parser.add_argument("--partition", default=os.environ.get("SLURM_PARTITION"), required=False)
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--container-image", default=os.environ.get("CONTAINER_IMAGE"))
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--runtime-venv", type=Path, help="Shared container-compatible venv used by training tasks.")
    parser.add_argument("--wandb-netrc", type=Path, help="Host netrc file mounted at /root/.netrc.")
    parser.add_argument("--wandb-project", default="megatron-bridge-text-pretrain-validation")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="mb747-text-pretrain-dclm-20260710")
    parser.add_argument("--tokenizer-model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--mount", action="append", default=[])
    parser.add_argument("--env", action="append", default=[], help="Extra environment name to inherit.")
    return parser


def _validate_args(args: argparse.Namespace, targets: list[ValidationTarget]) -> None:
    if args.nodes != 2 or args.gpus_per_node != 8:
        raise ValueError("MB-747 validation contract requires exactly 2 nodes x 8 GPUs.")
    if not args.account or not args.partition or not args.container_image:
        raise ValueError("--account, --partition, and --container-image are required.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    world_size = args.nodes * args.gpus_per_node
    for target in targets:
        if target.minimum_world_size > world_size or world_size % target.minimum_world_size != 0:
            raise ValueError(
                f"{target.id} parallelism requires a world size divisible by {target.minimum_world_size}, "
                f"got {world_size}."
            )
    for name in args.env:
        if "=" in name:
            raise ValueError("--env accepts NAME only; export the value before launching.")
        if name not in os.environ:
            raise ValueError(f"Environment variable {name!r} is not set.")


def _select_targets(targets: list[ValidationTarget], selected_ids: list[str]) -> list[ValidationTarget]:
    if not selected_ids:
        return targets
    targets_by_id = {target.id: target for target in targets}
    unknown = sorted(set(selected_ids) - targets_by_id.keys())
    if unknown:
        raise ValueError(f"Unknown target ids: {', '.join(unknown)}")
    selected = set(selected_ids)
    return [target for target in targets if target.id in selected]


def _append_mount(command: list[str], mounts: list[str], value: str) -> None:
    if value not in mounts:
        mounts.append(value)
        command.extend(["--mount", value])


def _append_env(command: list[str], env_names: list[str], name: str) -> None:
    if name not in env_names:
        env_names.append(name)
        command.extend(["--env", name])


def _runtime_path(runtime_venv: Path) -> str:
    return ":".join(
        [
            str(runtime_venv / "bin"),
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
    )


def build_environment(args: argparse.Namespace) -> dict[str, str]:
    """Build the launcher environment without serializing values in the command."""
    environment = os.environ.copy()
    if args.hf_home:
        environment["HF_HOME"] = str(args.hf_home)
    if args.runtime_venv:
        environment["VIRTUAL_ENV"] = str(args.runtime_venv)
        environment["PATH"] = _runtime_path(args.runtime_venv)
    environment["WANDB_RUN_GROUP"] = args.wandb_group
    environment["WANDB_JOB_TYPE"] = "pretrain-validation"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return environment


def build_command(args: argparse.Namespace, target: ValidationTarget) -> list[str]:
    """Build one train.sh command using the fixed MB-747 experiment contract."""
    output_dir = args.output_root / target.id
    command = [
        str(SCRIPT_DIR / "train.sh"),
        "--nodes",
        str(args.nodes),
        "--gpus-per-node",
        str(args.gpus_per_node),
        "--account",
        args.account,
        "--partition",
        args.partition,
        "--time",
        args.time,
        "--container-image",
        args.container_image,
        "--experiment-name",
        f"mb747-{target.id}",
    ]
    if args.no_gpu_resource_request:
        command.append("--no-gpu-resource-request")
    mounts: list[str] = []
    env_names: list[str] = []
    _append_mount(command, mounts, f"{REPO_ROOT}:{CONTAINER_REPO_ROOT}")
    for path in (args.dataset_path.parent, args.dataset_cache, args.output_root):
        _append_mount(command, mounts, str(path))
    if args.hf_home:
        _append_mount(command, mounts, str(args.hf_home))
        _append_env(command, env_names, "HF_HOME")
    if args.runtime_venv:
        _append_mount(command, mounts, str(args.runtime_venv))
        _append_env(command, env_names, "VIRTUAL_ENV")
        _append_env(command, env_names, "PATH")
    if args.wandb_netrc:
        _append_mount(command, mounts, f"{args.wandb_netrc}:/root/.netrc")
    for mount in args.mount:
        _append_mount(command, mounts, mount)
    for name in args.env:
        _append_env(command, env_names, name)
    for name in ("WANDB_RUN_GROUP", "WANDB_JOB_TYPE", "PYTORCH_CUDA_ALLOC_CONF"):
        _append_env(command, env_names, name)
    command.extend(
        [
            "--recipe",
            target.recipe,
            "--mode",
            "pretrain",
            "--dataset",
            "megatron-indexed",
            "--seq_length",
            "4096",
            "--max_steps",
            "100",
            "--global_batch_size",
            "128",
            "--micro_batch_size",
            "1",
            "--warmup_iters",
            "10",
            "--save_dir",
            str(output_dir / "checkpoints"),
            "--save_interval",
            "100",
            "-tp",
            str(target.tp),
            "-pp",
            str(target.pp),
            "-cp",
            "1",
            "-ep",
            str(target.ep),
            "-etp",
            "1",
            f"dataset.data_path={args.dataset_path}",
            f"dataset.path_to_cache={args.dataset_cache}",
            "tokenizer.tokenizer_type=HuggingFaceTokenizer",
            f"tokenizer.tokenizer_model={args.tokenizer_model}",
            "train.eval_interval=100",
            "train.eval_iters=1",
            "scheduler.lr_decay_iters=100",
            "logger.log_interval=1",
            f"logger.wandb_project={args.wandb_project}",
            f"logger.wandb_exp_name=mb747-{target.id}-dclm-100steps",
            f"logger.wandb_save_dir={output_dir / 'wandb'}",
            "model.virtual_pipeline_model_parallel_size=null",
            "checkpoint.load=null",
            f"logger.tensorboard_dir={output_dir / 'tensorboard'}",
            *target.overrides,
        ]
    )
    if args.wandb_entity:
        command.append(f"logger.wandb_entity={args.wandb_entity}")
    if args.submission_dry_run:
        command.append("--submission-dry-run")
    return command


def main(argv: list[str] | None = None) -> None:
    """Render or submit selected validation targets."""
    args = _build_parser().parse_args(argv)
    targets = _select_targets(load_manifest(args.manifest), args.model)
    if args.limit is not None:
        targets = targets[: args.limit]
    _validate_args(args, targets)

    LOGGER.info("Selected %d validation target(s).", len(targets))
    for target in targets:
        command = build_command(args, target)
        LOGGER.info("[%s] %s", target.id, shlex.join(command))
        if args.submit:
            output_dir = args.output_root / target.id
            (output_dir / "wandb").mkdir(parents=True, exist_ok=True)
            (output_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, env=build_environment(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
