#!/usr/bin/env python3
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
"""Run a Megatron Bridge library recipe in an existing distributed environment.

Select either a model name with --model or a complete recipe function with
--recipe. The public launcher accepts one training mode, a direct dataset name,
runner controls, and trailing KEY=VALUE ConfigContainer overrides. Slurm
resources, containers, mounts, and environment forwarding are owned by
setup_experiment.py.

Common ConfigContainer overrides
--------------------------------
The scripts/performance options on the left are equivalent to the trailing
overrides on the right. They are documented as a migration aid; this command
continues to accept the KEY=VALUE form only.

Training:
  -ms, --max_steps STEPS                 train.train_iters=STEPS
  -gb, --global_batch_size SIZE          train.global_batch_size=SIZE
  -mb, --micro_batch_size SIZE           train.micro_batch_size=SIZE

Sequence length:
  -sl, --seq_length LENGTH               dataset.sequence_length=LENGTH  (pretraining)
                                           dataset.seq_length=LENGTH       (SFT/PEFT)

The selected dataset owns the sequence length. After overrides are applied,
the runner synchronizes model.seq_length from the dataset field.

Parallelism:
  -tp, --tensor_model_parallel_size N    model.tensor_model_parallel_size=N
  -pp, --pipeline_model_parallel_size N  model.pipeline_model_parallel_size=N
  -cp, --context_parallel_size N         model.context_parallel_size=N
  -vp, --virtual_pipeline_model_parallel_size N
                                           model.virtual_pipeline_model_parallel_size=N
  -ep, --expert_model_parallel_size N    model.expert_model_parallel_size=N
  -et, --expert_tensor_parallel_size N   model.expert_tensor_parallel_size=N

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
    dataset.sequence_length=8192 train.micro_batch_size=1 \\
    model.tensor_model_parallel_size=2 model.sequence_parallel=true
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recipe_runner import (  # noqa: E402
    apply_cli_overrides,
    apply_determinism,
    apply_runtime_environment,
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
from megatron.bridge.training.config import ConfigContainer  # noqa: E402


PublicMode = Literal["pretrain", "sft", "lora", "dora"]
TrainMode = Literal["pretrain", "finetune"]


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
    recipe_selection.add_argument("--recipe", help="Complete library recipe function name.")
    selection.add_argument(
        "--mode",
        choices=["pretrain", "sft", "lora", "dora"],
        help="Training mode; inferred from a conventional --recipe name when omitted.",
    )
    selection.add_argument(
        "--step-func",
        "--step_func",
        default="llm_step",
        dest="step_func",
        help="Forward-step registry name; most LLM recipes use the default.",
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS),
        help="Dataset config preset or local source selector.",
    )
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


def _load_selected_recipe(args: argparse.Namespace) -> ConfigContainer:
    """Load the requested library recipe."""
    peft_scheme = args.mode if args.mode in {"lora", "dora"} else None
    recipe_name = args.recipe or f"{args.model}_{_recipe_task(args.mode)}_config"
    if args.recipe:
        _validate_recipe_mode(recipe_name, args.mode)
    return load_recipe(
        recipe_name,
        peft_scheme=peft_scheme,
    )


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
    """Parse public arguments and trailing ConfigContainer overrides."""
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    cli_overrides = _collect_overrides(parser, unknown)
    _infer_mode(args)
    return args, cli_overrides


def main(argv: list[str] | None = None) -> None:
    """Load, configure, and execute one library recipe."""
    logging.basicConfig(level=logging.INFO)
    args, cli_overrides = parse_args(argv)

    recipe = _load_selected_recipe(args)
    recipe = _apply_dataset(recipe, args)
    recipe = apply_determinism(recipe, deterministic=args.deterministic)
    recipe = apply_cli_overrides(recipe, cli_overrides)
    recipe = apply_runtime_environment(recipe)
    mode = _train_mode(args.mode)
    recipe = sync_finetuning_cp_invariants(recipe, mode=mode)
    recipe = sync_offline_packing_alignment(recipe)
    recipe = sync_model_dataset_sequence_length(recipe)

    forward_step = load_forward_step(args.step_func, mode=mode)
    run_config(
        config=recipe,
        mode=mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        dump_environment=args.dump_env,
    )


if __name__ == "__main__":
    main()
