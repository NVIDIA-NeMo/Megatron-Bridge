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
"""Run a Megatron Bridge recipe in an existing distributed environment.

Select either a model name with --model or a complete recipe function with
--recipe. Complete recipes may come from the functional library or flat
performance namespace; select the latter explicitly with
--recipe-source performance. The public launcher accepts one training mode, a
direct dataset name, runner controls, common convenience arguments, and
trailing KEY=VALUE ConfigContainer overrides. Slurm resources, containers,
mounts, and environment forwarding are owned by setup_experiment.py.

Common ConfigContainer overrides
--------------------------------
The convenience options on the left are converted to the ConfigContainer
overrides on the right. This command accepts both forms. When both forms set
the same field, the trailing KEY=VALUE override takes precedence.

Training:
  -ms, --max_steps STEPS                 train.train_iters=STEPS
  -gb, --global_batch_size SIZE          train.global_batch_size=SIZE
  -mb, --micro_batch_size SIZE           train.micro_batch_size=SIZE

Sequence length:
  -sl, --seq_length LENGTH               dataset.seq_length=LENGTH

The selected dataset owns the sequence length. After overrides are applied,
the runner synchronizes model.seq_length from the dataset field.

Parallelism:
  -tp, --tensor_model_parallel_size N    model.tensor_model_parallel_size=N
  -pp, --pipeline_model_parallel_size N  model.pipeline_model_parallel_size=N
  -cp, --context_parallel_size N         model.context_parallel_size=N
  -vp, --virtual_pipeline_model_parallel_size N
                                           model.virtual_pipeline_model_parallel_size=N
  -ep, --expert_model_parallel_size N    model.expert_model_parallel_size=N
  -etp, --expert_tensor_parallel_size N  model.expert_tensor_parallel_size=N

Optimization:
  --lr VALUE                             optimizer.lr=VALUE
  --min_lr VALUE                         optimizer.min_lr=VALUE
  --warmup_iters STEPS                   scheduler.lr_warmup_iters=STEPS

Checkpointing:
  --pretrained_checkpoint PATH           checkpoint.pretrained_checkpoint=PATH
  --save_dir PATH                        checkpoint.save=PATH
  --load_dir PATH                        checkpoint.load=PATH
  --save_interval STEPS                  checkpoint.save_interval=STEPS

For example:
  run_recipe.py --model gpt_oss_20b --mode pretrain --dataset mock \\
    -sl 8192 -mb 1 -tp 2 model.sequence_parallel=true

  run_recipe.py --recipe-source performance \\
    --recipe qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config --mode pretrain
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from performance_recipe import PerformanceRecipeMetadata, performance_recipe_metadata  # noqa: E402
from recipe_runner import (  # noqa: E402
    apply_cli_overrides,
    apply_determinism,
    apply_runtime_environment,
    bootstrap_recipe_environment,
    load_forward_step,
    load_recipe,
    run_config,
    sync_finetuning_cp_invariants,
    sync_model_dataset_sequence_length,
    sync_offline_packing_alignment,
)

from megatron.bridge.recipes.utils.dataset_utils import (  # noqa: E402
    DATASET_PRESETS,
    build_dataset_config,
    dataset_train_mode,
)


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


PublicMode = Literal["pretrain", "sft", "lora", "dora"]
TrainMode = Literal["pretrain", "finetune"]


PERFORMANCE_SAFE_OVERRIDE_FIELDS = (
    "env_vars",
    "logger",
    "profiling",
    "train.train_iters",
)


COMMON_OVERRIDE_FIELDS = (
    ("max_steps", "train.train_iters"),
    ("global_batch_size", "train.global_batch_size"),
    ("micro_batch_size", "train.micro_batch_size"),
    ("seq_length", "dataset.seq_length"),
    ("tensor_model_parallel_size", "model.tensor_model_parallel_size"),
    ("pipeline_model_parallel_size", "model.pipeline_model_parallel_size"),
    ("context_parallel_size", "model.context_parallel_size"),
    ("virtual_pipeline_model_parallel_size", "model.virtual_pipeline_model_parallel_size"),
    ("expert_model_parallel_size", "model.expert_model_parallel_size"),
    ("expert_tensor_parallel_size", "model.expert_tensor_parallel_size"),
    ("lr", "optimizer.lr"),
    ("min_lr", "optimizer.min_lr"),
    ("warmup_iters", "scheduler.lr_warmup_iters"),
    ("pretrained_checkpoint", "checkpoint.pretrained_checkpoint"),
    ("save_dir", "checkpoint.save"),
    ("load_dir", "checkpoint.load"),
    ("save_interval", "checkpoint.save_interval"),
)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Preserve the override table while retaining argparse default annotations."""


def _build_parser() -> argparse.ArgumentParser:
    """Build the public training parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
        allow_abbrev=False,
    )

    selection = parser.add_argument_group("Selection")
    recipe_selection = selection.add_mutually_exclusive_group(required=True)
    recipe_selection.add_argument("--model", help="Model recipe stem, for example gpt_oss_20b.")
    recipe_selection.add_argument("--recipe", help="Complete recipe function name.")
    selection.add_argument(
        "--recipe-source",
        choices=["library", "performance"],
        default="library",
        help="Recipe namespace used by --recipe; --model always selects the functional library.",
    )
    selection.add_argument(
        "--mode",
        choices=["pretrain", "sft", "lora", "dora"],
        help="Training mode; inferred from a conventional --recipe name when omitted.",
    )
    selection.add_argument(
        "--step-func",
        "--step_func",
        dest="step_func",
        help="Forward-step registry name; defaults to llm_step for library recipes and gpt_step for performance.",
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS),
        help="Dataset config preset or local source selector.",
    )

    training = parser.add_argument_group("Common training overrides")
    training.add_argument("-ms", "--max_steps", type=int, metavar="STEPS", help="Set train.train_iters.")
    training.add_argument("-gb", "--global_batch_size", type=int, metavar="SIZE", help="Set train.global_batch_size.")
    training.add_argument("-mb", "--micro_batch_size", type=int, metavar="SIZE", help="Set train.micro_batch_size.")

    sequence_length = parser.add_argument_group("Sequence length override")
    sequence_length.add_argument(
        "-sl", "--seq_length", type=int, metavar="LENGTH", help="Set dataset.seq_length and synchronize the model."
    )

    parallelism = parser.add_argument_group("Common parallelism overrides")
    parallelism.add_argument(
        "-tp",
        "--tensor_model_parallel_size",
        type=int,
        metavar="N",
        help="Set model.tensor_model_parallel_size.",
    )
    parallelism.add_argument(
        "-pp",
        "--pipeline_model_parallel_size",
        type=int,
        metavar="N",
        help="Set model.pipeline_model_parallel_size.",
    )
    parallelism.add_argument(
        "-cp",
        "--context_parallel_size",
        type=int,
        metavar="N",
        help="Set model.context_parallel_size.",
    )
    parallelism.add_argument(
        "-vp",
        "--virtual_pipeline_model_parallel_size",
        type=int,
        metavar="N",
        help="Set model.virtual_pipeline_model_parallel_size.",
    )
    parallelism.add_argument(
        "-ep",
        "--expert_model_parallel_size",
        type=int,
        metavar="N",
        help="Set model.expert_model_parallel_size.",
    )
    parallelism.add_argument(
        "-etp",
        "--expert_tensor_parallel_size",
        type=int,
        metavar="N",
        help="Set model.expert_tensor_parallel_size.",
    )

    optimization = parser.add_argument_group("Common optimization overrides")
    optimization.add_argument("--lr", type=float, metavar="VALUE", help="Set optimizer.lr.")
    optimization.add_argument("--min_lr", type=float, metavar="VALUE", help="Set optimizer.min_lr.")
    optimization.add_argument("--warmup_iters", type=int, metavar="STEPS", help="Set scheduler.lr_warmup_iters.")

    checkpointing = parser.add_argument_group("Common checkpoint overrides")
    checkpointing.add_argument("--pretrained_checkpoint", metavar="PATH", help="Set checkpoint.pretrained_checkpoint.")
    checkpointing.add_argument("--save_dir", metavar="PATH", help="Set checkpoint.save.")
    checkpointing.add_argument("--load_dir", metavar="PATH", help="Set checkpoint.load.")
    checkpointing.add_argument("--save_interval", type=int, metavar="STEPS", help="Set checkpoint.save_interval.")

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--dry-run", "--dry_run", action="store_true", dest="dryrun")
    runtime.add_argument("--deterministic", action="store_true")
    runtime.add_argument("--dump-env", "--dump_env", action="store_true", dest="dump_env")
    return parser


def _collect_overrides(parser: argparse.ArgumentParser, values: list[str]) -> list[str]:
    """Validate trailing ConfigContainer overrides."""
    overrides: list[str] = []
    for value in values:
        if value.startswith("-"):
            parser.error(f"Unknown option: {value}")
        if "=" not in value:
            parser.error(f"Expected override in KEY=VALUE form, got {value!r}")
        overrides.append(value)
    return overrides


def _common_config_overrides(args: argparse.Namespace) -> list[str]:
    """Convert common convenience arguments to ConfigContainer overrides."""
    overrides = []
    for argument_name, config_field in COMMON_OVERRIDE_FIELDS:
        value = getattr(args, argument_name)
        if value is None:
            continue
        serialized_value = json.dumps(value) if isinstance(value, str) else str(value)
        overrides.append(f"{config_field}={serialized_value}")
    return overrides


def _train_mode(mode: PublicMode) -> TrainMode:
    """Map the public mode to a training loop."""
    return "pretrain" if mode == "pretrain" else "finetune"


def _recipe_task(mode: PublicMode) -> str:
    """Map the public mode to a library recipe suffix."""
    return "peft" if mode in {"lora", "dora"} else mode


def _infer_recipe_mode(recipe_name: str) -> PublicMode | None:
    """Infer a public mode from a conventional library recipe name."""
    name = f"_{recipe_name.lower().strip('_')}_"
    if "_pretrain_" in name:
        return "pretrain"
    if "_sft_" in name or "_finetune_" in name:
        return "sft"
    if "_dora_" in name:
        return "dora"
    if "_peft_" in name or "_lora_" in name:
        return "lora"
    return None


def _infer_mode(args: argparse.Namespace) -> None:
    """Infer an omitted training mode from a conventional recipe name."""
    if args.mode is None and args.recipe:
        args.mode = _infer_recipe_mode(args.recipe)
    if args.mode is None:
        raise ValueError("Unable to infer training mode; pass --mode or use a conventional --recipe name.")


def _validate_recipe_mode(recipe_name: str, mode: PublicMode) -> None:
    """Reject a conventional full recipe name that contradicts ``--mode``."""
    name = f"_{recipe_name.lower().strip('_')}_"
    recipe_mode: str | None = None
    if "_pretrain_" in name:
        recipe_mode = "pretrain"
    elif "_sft_" in name or "_finetune_" in name:
        recipe_mode = "sft"
    elif any(marker in name for marker in ("_peft_", "_lora_", "_dora_")):
        recipe_mode = "peft"

    expected = _recipe_task(mode)
    if recipe_mode is not None and recipe_mode != expected:
        raise ValueError(f"Mode '{mode}' is incompatible with recipe '{recipe_name}'.")


def _validate_performance_overrides(cli_overrides: list[str]) -> None:
    """Allow only overrides that preserve the canonical benchmark configuration."""
    unsupported_overrides = []
    for override in cli_overrides:
        field_name = override.lstrip("+~").split("=", 1)[0]
        if not any(
            field_name == safe_field or field_name.startswith(f"{safe_field}.")
            for safe_field in PERFORMANCE_SAFE_OVERRIDE_FIELDS
        ):
            unsupported_overrides.append(field_name)

    if unsupported_overrides:
        fields = ", ".join(dict.fromkeys(unsupported_overrides))
        raise ValueError(
            "Performance recipes currently require their canonical model, topology, batch, sequence length, "
            f"dispatcher, graph, precision, and dataset settings; unsupported overrides: {fields}."
        )


def _validate_performance_scope(
    args: argparse.Namespace,
    metadata: PerformanceRecipeMetadata,
) -> None:
    """Limit the first unified-launcher stage to canonical text pretraining."""
    if args.mode != "pretrain":
        raise ValueError(
            "The training launcher currently supports performance pretraining recipes only; continue using "
            "scripts/performance for SFT and PEFT benchmarks during migration."
        )
    if metadata.family in {"qwen_vl", "wan"}:
        raise ValueError(
            "The training launcher currently supports text performance recipes only; continue using "
            "scripts/performance for VLM and diffusion benchmarks during migration."
        )
    if args.step_func is not None and args.step_func.lower() != "gpt_step":
        raise ValueError(
            "Text performance recipes use the canonical gpt_step forward step; omit --step-func or pass gpt_step."
        )
    if args.deterministic:
        raise ValueError(
            "Performance recipes currently require their canonical benchmark settings; --deterministic is not "
            "supported by the training launcher during migration."
        )
    if args.dataset is not None:
        raise ValueError(
            "Performance recipes own their canonical dataset; omit --dataset or continue using scripts/performance "
            "for non-canonical benchmark data."
        )


def _current_world_size() -> int | None:
    """Return the distributed world size supplied by torchrun or Slurm, when available."""
    for variable_name in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(variable_name)
        if value is None:
            continue
        world_size = int(value)
        if world_size < 1:
            raise ValueError(f"{variable_name} must be positive, got {value!r}.")
        return world_size
    return None


def _current_local_world_size() -> int | None:
    """Return the number of rank-local workers supplied by torchrun or Slurm."""
    for variable_name in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", "SLURM_TASKS_PER_NODE"):
        value = os.environ.get(variable_name)
        if value is None:
            continue
        if variable_name == "SLURM_TASKS_PER_NODE":
            match = re.fullmatch(r"(?P<tasks>[1-9][0-9]*)(?:\(x[1-9][0-9]*\))?", value)
            if match is None:
                raise ValueError(
                    "SLURM_TASKS_PER_NODE must describe one homogeneous tasks-per-node value for performance "
                    f"recipes, got {value!r}."
                )
            local_world_size = int(match.group("tasks"))
        else:
            local_world_size = int(value)
        if local_world_size < 1:
            raise ValueError(f"{variable_name} must be positive, got {value!r}.")
        return local_world_size
    return None


def _validate_performance_world_size(metadata: PerformanceRecipeMetadata, *, dryrun: bool) -> None:
    """Require the canonical total and per-node GPU topology for an executable run."""
    if dryrun:
        return
    world_size = _current_world_size()
    local_world_size = _current_local_world_size()
    if world_size is None or local_world_size is None:
        raise ValueError(
            "Performance recipes require an existing distributed environment with total and local world sizes set "
            "by torchrun or Slurm."
        )
    if world_size != metadata.num_gpus:
        raise ValueError(
            f"Performance recipe requires exactly {metadata.num_gpus} GPUs, but the distributed world size is "
            f"{world_size}. Select a recipe matching the allocation."
        )
    if local_world_size != metadata.gpus_per_node:
        raise ValueError(
            f"Performance recipe requires exactly {metadata.gpus_per_node} GPUs per {metadata.hardware} node, but "
            f"the local world size is {local_world_size}. Use the canonical node topology."
        )


def _apply_performance_runtime_defaults(
    recipe: ConfigContainer,
    metadata: PerformanceRecipeMetadata,
) -> ConfigContainer:
    """Preserve flat-runner defaults that are not yet encoded in the recipe factory."""
    optimizer = getattr(recipe, "optimizer", None)
    if metadata.precision == "bf16" and getattr(optimizer, "optimizer", None) == "adam":
        optimizer.use_precision_aware_optimizer = True
    return recipe


def _load_selected_recipe(args: argparse.Namespace) -> ConfigContainer:
    """Load the requested recipe from its explicit namespace."""
    if args.model and args.recipe_source != "library":
        raise ValueError(
            "--recipe-source performance requires an explicit --recipe name; --model uses library recipes."
        )

    peft_scheme = args.mode if args.mode in {"lora", "dora"} else None
    recipe_name = args.recipe or f"{args.model}_{_recipe_task(args.mode)}_config"
    if args.recipe:
        _validate_recipe_mode(recipe_name, args.mode)
    load_kwargs = {"source": "performance"} if args.recipe_source == "performance" else {}
    return load_recipe(recipe_name, peft_scheme=peft_scheme, **load_kwargs)


def _apply_dataset(recipe: ConfigContainer, args: argparse.Namespace) -> ConfigContainer:
    """Apply a public dataset selection to a recipe config."""
    if args.dataset is None:
        return recipe

    recipe.dataset = build_dataset_config(recipe, args.dataset)
    requested_train_mode = _train_mode(args.mode)
    if dataset_train_mode(recipe.dataset) != requested_train_mode:
        raise ValueError(f"Mode '{args.mode}' is incompatible with dataset '{args.dataset}'.")
    return recipe


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse public arguments and ordered ConfigContainer overrides."""
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    trailing_overrides = _collect_overrides(parser, unknown)
    cli_overrides = [*_common_config_overrides(args), *trailing_overrides]
    _infer_mode(args)
    return args, cli_overrides


def main(argv: list[str] | None = None) -> None:
    """Load, configure, and execute one library or performance recipe."""
    logging.basicConfig(level=logging.INFO)
    args, cli_overrides = parse_args(argv)

    metadata = None
    if args.recipe_source == "performance":
        if args.recipe is None:
            raise ValueError(
                "--recipe-source performance requires an explicit --recipe name; --model uses library recipes."
            )
        metadata = performance_recipe_metadata(args.recipe)
        _validate_performance_scope(args, metadata)
        _validate_performance_overrides(cli_overrides)

    recipe = _load_selected_recipe(args)
    recipe = _apply_dataset(recipe, args)
    recipe = apply_determinism(recipe, deterministic=args.deterministic)
    recipe = apply_cli_overrides(recipe, cli_overrides)
    configuration_mode = _train_mode(args.mode)

    if metadata is not None:
        recipe = _apply_performance_runtime_defaults(recipe, metadata)
        _validate_performance_world_size(metadata, dryrun=args.dryrun)
        recipe = bootstrap_recipe_environment(
            recipe,
            script_path=str(Path(__file__).resolve()),
            argv=list(argv) if argv is not None else sys.argv[1:],
        )
        execution_mode = "pretrain"
        step_mode = _recipe_task(args.mode)
    else:
        recipe = apply_runtime_environment(recipe)
        execution_mode = configuration_mode
        step_mode = configuration_mode

    recipe = sync_finetuning_cp_invariants(recipe, mode=configuration_mode)
    recipe = sync_offline_packing_alignment(recipe)
    recipe = sync_model_dataset_sequence_length(recipe)

    step_func_name = args.step_func or ("gpt_step" if metadata is not None else "llm_step")
    forward_step = load_forward_step(step_func_name, mode=step_mode)
    run_config(
        config=recipe,
        mode=execution_mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        dryrun_world_size=metadata.num_gpus if metadata is not None else None,
        dump_environment=args.dump_env,
    )


if __name__ == "__main__":
    main()
