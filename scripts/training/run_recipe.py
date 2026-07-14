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

Select either a model name with ``--model`` or a complete recipe function with
``--recipe``. The public launcher accepts one training mode, a direct dataset
name, runner controls, and trailing ``KEY=VALUE`` ConfigContainer overrides.
Slurm resources, containers, mounts, and environment forwarding are owned by
``setup_experiment.py``.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, TypeVar, cast

from hydra.core.override_parser.overrides_parser import OverridesParser


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recipe_runner import (  # noqa: E402
    STEP_MODALITIES,
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
    PUBLIC_DATASETS,
    apply_public_dataset_override,
)


PublicMode = Literal["pretrain", "sft", "lora", "dora"]
TrainMode = Literal["pretrain", "finetune"]


DATASETS = PUBLIC_DATASETS
_T = TypeVar("_T")


def _build_parser() -> argparse.ArgumentParser:
    """Build the public training parser."""
    parser = argparse.ArgumentParser(
        description="Run a Megatron Bridge library recipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        choices=sorted(DATASETS),
        help="Dataset name.",
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


def _override_values(cli_overrides: list[str]) -> dict[str, object]:
    """Return the final parsed value for each Hydra-style config override."""
    parsed = OverridesParser.create().parse_overrides(overrides=cli_overrides)
    return {override.get_key_element(): override.value() for override in parsed}


def _optional_override(values: dict[str, object], key: str, expected_type: type[_T]) -> _T | None:
    """Read and type-check a config override used while constructing a dataset."""
    value = values.get(key)
    if value is not None and not isinstance(value, expected_type):
        raise ValueError(f"{key} must be a {expected_type.__name__} value.")
    return value


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
    """Infer an omitted training mode from the recipe or dataset name."""
    if args.mode is None and args.recipe:
        args.mode = _infer_recipe_mode(args.recipe)
    if args.mode is None and args.dataset is not None:
        args.mode = "pretrain" if DATASETS[args.dataset].train_mode == "pretrain" else "sft"
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


def _load_selected_recipe(args: argparse.Namespace) -> object:
    """Load the requested library recipe."""
    peft_scheme = args.mode if args.mode in {"lora", "dora"} else None
    recipe_name = args.recipe or f"{args.model}_{_recipe_task(args.mode)}_config"
    if args.recipe:
        _validate_recipe_mode(recipe_name, args.mode)
    return load_recipe(
        recipe_name,
        peft_scheme=peft_scheme,
    )


def _validate_dataset_options(args: argparse.Namespace) -> None:
    """Validate public dataset selection before loading a potentially expensive recipe."""
    if args.dataset is None:
        return

    selection = DATASETS[args.dataset]
    if selection.modality == "vlm" and STEP_MODALITIES.get(args.step_func.lower()) != "vlm":
        raise ValueError(f"Dataset '{args.dataset}' requires a VLM-compatible --step-func.")


def _apply_dataset(
    recipe: object,
    args: argparse.Namespace,
    cli_overrides: list[str],
) -> object:
    """Apply a public dataset selection to a recipe config."""
    if args.dataset is None:
        return recipe

    selection = DATASETS[args.dataset]
    requested_train_mode = _train_mode(args.mode)
    if selection.train_mode != requested_train_mode:
        raise ValueError(f"Mode '{args.mode}' is incompatible with dataset '{args.dataset}'.")

    values = _override_values(cli_overrides)
    sequence_lengths = {
        key: values[key]
        for key in ("dataset.seq_length", "dataset.sequence_length", "model.seq_length")
        if key in values
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in sequence_lengths.values()):
        raise ValueError("Sequence length must be an int value.")
    if len(set(sequence_lengths.values())) > 1:
        raise ValueError(f"Sequence-length overrides disagree: {sequence_lengths}")
    seq_length = cast(int | None, next(iter(sequence_lengths.values()), None))

    enable_offline_packing = _optional_override(values, "dataset.enable_offline_packing", bool)
    pad_seq_to_mult = _optional_override(values, "dataset.offline_packing_specs.pad_seq_to_mult", int)
    if enable_offline_packing and not selection.supports_offline_packing:
        raise ValueError(f"Dataset '{args.dataset}' does not support dataset.enable_offline_packing=true.")
    if pad_seq_to_mult is not None and not enable_offline_packing:
        raise ValueError("dataset.offline_packing_specs.pad_seq_to_mult requires dataset.enable_offline_packing=true.")
    if pad_seq_to_mult is not None and pad_seq_to_mult < 1:
        raise ValueError("dataset.offline_packing_specs.pad_seq_to_mult must be greater than zero.")
    recipe = apply_public_dataset_override(
        recipe,
        dataset_name=args.dataset,
        enable_offline_packing=bool(enable_offline_packing),
        pad_seq_to_mult=pad_seq_to_mult if pad_seq_to_mult is not None else 1,
        seq_length=seq_length,
        dataset_root=_optional_override(values, "dataset.dataset_root", str),
        train_data_path=_optional_override(values, "dataset.source.load_kwargs.data_files.train", str),
        validation_data_path=_optional_override(
            values, "dataset.validation_source.load_kwargs.data_files.validation", str
        ),
        test_data_path=_optional_override(values, "dataset.test_source.load_kwargs.data_files.test", str),
        media_root=_optional_override(values, "dataset.source.adapter_kwargs.video_root_path", str),
        hf_processor_path=_optional_override(values, "dataset.hf_processor_path", str),
    )
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
    _validate_dataset_options(args)

    recipe = _load_selected_recipe(args)
    recipe = _apply_dataset(recipe, args, cli_overrides)
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
