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
--recipe. The launcher discovers complete library and benchmark recipes
by their exported function name. It accepts one training mode, a direct dataset
name, runner controls, common convenience arguments, and trailing KEY=VALUE
ConfigContainer overrides. Slurm resources, containers, mounts, and environment
forwarding are owned by setup_experiment.py.

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

  run_recipe.py \\
    --recipe qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config --mode pretrain
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recipe_metadata import (  # noqa: E402
    BenchmarkRecipeMetadata,
    infer_recipe_mode,
    recipe_step,
    recipe_steps_match,
    recipe_task,
    resolved_benchmark_recipe_metadata,
    validate_benchmark_recipe_scope,
)
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
    sync_model_pipeline_layout,
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
        "--mode",
        choices=["pretrain", "sft", "lora", "dora"],
        help="Training mode; inferred from a conventional --recipe name when omitted.",
    )
    selection.add_argument(
        "--step-func",
        "--step_func",
        dest="step_func",
        help="Forward-step registry name; defaults to the selected recipe's registered modality step.",
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


def _infer_mode(args: argparse.Namespace) -> None:
    """Infer an omitted training mode from a conventional recipe name."""
    if args.mode is None and args.recipe:
        args.mode = infer_recipe_mode(args.recipe)
    if args.mode is None:
        raise ValueError("Unable to infer training mode; pass --mode or use a conventional --recipe name.")


def _validate_recipe_mode(recipe_name: str, mode: PublicMode) -> None:
    """Reject a conventional full recipe name that contradicts ``--mode``."""
    recipe_mode = infer_recipe_mode(recipe_name)
    if recipe_mode is not None and recipe_task(recipe_mode) != recipe_task(mode):
        raise ValueError(f"Mode '{mode}' is incompatible with recipe '{recipe_name}'.")


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


def _validate_benchmark_world_size(metadata: BenchmarkRecipeMetadata, *, dryrun: bool) -> None:
    """Require the recipe's total GPU count for an executable run."""
    if dryrun:
        return
    world_size = _current_world_size()
    if world_size is None:
        raise ValueError(
            "Benchmark recipes require an existing distributed environment with the world size set by torchrun "
            "or Slurm."
        )
    if world_size != metadata.num_gpus:
        raise ValueError(
            f"Benchmark recipe requires exactly {metadata.num_gpus} GPUs, but the distributed world size is "
            f"{world_size}. Select a recipe matching the allocation."
        )


def _apply_benchmark_runtime_defaults(
    recipe: ConfigContainer,
    metadata: BenchmarkRecipeMetadata,
    cli_overrides: list[str],
) -> ConfigContainer:
    """Preserve flat-runner defaults that are not yet encoded in the recipe factory."""
    optimizer = getattr(recipe, "optimizer", None)
    precision_aware_field = "optimizer.use_precision_aware_optimizer"
    precision_aware_is_explicit = any(
        override.lstrip("+~").split("=", 1)[0] in {"optimizer", precision_aware_field} for override in cli_overrides
    )
    if (
        not precision_aware_is_explicit
        and metadata.precision == "bf16"
        and getattr(optimizer, "optimizer", None) == "adam"
    ):
        optimizer.use_precision_aware_optimizer = True
    return recipe


def _apply_benchmark_dataset_defaults(
    recipe: ConfigContainer,
    metadata: BenchmarkRecipeMetadata,
) -> ConfigContainer:
    """Preserve the flat runner's mock-data default for text finetuning benchmarks."""
    if metadata.task in {"sft", "peft"} and recipe_steps_match(recipe_step(metadata.recipe_name), "llm_step"):
        source_dataset = getattr(recipe, "dataset", None)
        mock_dataset = build_dataset_config(recipe, "mock")
        for field_name in ("num_workers", "pin_memory", "persistent_workers"):
            if (
                source_dataset is not None
                and hasattr(source_dataset, field_name)
                and hasattr(mock_dataset, field_name)
            ):
                setattr(mock_dataset, field_name, getattr(source_dataset, field_name))
        if hasattr(mock_dataset, "split"):
            mock_dataset.split = "99990,8,2"
        recipe.dataset = mock_dataset
    return recipe


def _selected_recipe_name(args: argparse.Namespace) -> str:
    """Return the complete recipe function name selected by public arguments."""
    return args.recipe or f"{args.model}_{recipe_task(args.mode)}_config"


def _load_selected_recipe(args: argparse.Namespace) -> ConfigContainer:
    """Load the requested recipe by its complete name or model-derived library name."""
    peft_scheme = args.mode if args.mode in {"lora", "dora"} else None
    recipe_name = _selected_recipe_name(args)
    if args.recipe:
        _validate_recipe_mode(recipe_name, args.mode)
    return load_recipe(recipe_name, peft_scheme=peft_scheme)


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
    """Load, configure, and execute one library or benchmark recipe."""
    logging.basicConfig(level=logging.INFO)
    args, cli_overrides = parse_args(argv)

    recipe_name = _selected_recipe_name(args)
    benchmark_metadata = resolved_benchmark_recipe_metadata(recipe_name)
    if benchmark_metadata is not None:
        validate_benchmark_recipe_scope(
            benchmark_metadata,
            mode=args.mode,
            step_func=args.step_func,
            dataset=args.dataset,
        )

    recipe = _load_selected_recipe(args)
    if benchmark_metadata is not None:
        recipe = _apply_benchmark_dataset_defaults(recipe, benchmark_metadata)
    recipe = _apply_dataset(recipe, args)
    recipe = apply_determinism(recipe, deterministic=args.deterministic)
    recipe = apply_cli_overrides(recipe, cli_overrides)
    recipe = sync_model_pipeline_layout(recipe, cli_overrides=cli_overrides)
    if benchmark_metadata is not None:
        recipe = _apply_benchmark_runtime_defaults(recipe, benchmark_metadata, cli_overrides)
    configuration_mode = _train_mode(args.mode)

    if benchmark_metadata is not None:
        _validate_benchmark_world_size(benchmark_metadata, dryrun=args.dryrun)
        recipe = bootstrap_recipe_environment(
            recipe,
            script_path=str(Path(__file__).resolve()),
            argv=list(argv) if argv is not None else sys.argv[1:],
        )
        execution_mode = "pretrain"
        step_mode = benchmark_metadata.task
    else:
        recipe = apply_runtime_environment(recipe)
        execution_mode = configuration_mode
        step_mode = configuration_mode

    recipe = sync_finetuning_cp_invariants(recipe, mode=configuration_mode)
    recipe = sync_offline_packing_alignment(recipe)
    recipe = sync_model_dataset_sequence_length(recipe)

    step_func_name = args.step_func or recipe_step(recipe_name)
    forward_step = load_forward_step(step_func_name, mode=step_mode)
    run_config(
        config=recipe,
        mode=execution_mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        dryrun_world_size=benchmark_metadata.num_gpus if benchmark_metadata is not None else None,
        dump_environment=args.dump_env,
    )


if __name__ == "__main__":
    main()
