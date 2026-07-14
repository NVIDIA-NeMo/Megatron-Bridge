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
"""Run a Megatron Bridge library recipe in an existing distributed environment.

Select either a model name with ``--model`` or a complete recipe function with
``--recipe``. The public launcher accepts one training mode, a direct dataset
name, optional convenience flags, and trailing ``KEY=VALUE`` ConfigContainer
overrides. Slurm resources, containers, mounts, and environment forwarding are
owned by ``setup_experiment.py``.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recipe_runner import (  # noqa: E402
    apply_cli_overrides,
    apply_determinism,
    apply_launcher_overrides,
    load_forward_step,
    load_recipe,
    run_config,
    sync_model_dataset_sequence_length,
)

from megatron.bridge.recipes.utils.dataset_utils import apply_public_dataset_override  # noqa: E402


PublicMode = Literal["pretrain", "sft", "lora", "dora"]
TrainMode = Literal["pretrain", "finetune"]


@dataclass(frozen=True)
class _DatasetSelection:
    """Internal configuration for one public dataset name."""

    train_mode: TrainMode
    packed_sequence: bool = False
    thinking_format: bool = False
    indexed_data: bool = False


DATASETS = {
    "mock": _DatasetSelection("pretrain"),
    "dclm": _DatasetSelection("pretrain", indexed_data=True),
    "rp2": _DatasetSelection("pretrain", indexed_data=True),
    "c4": _DatasetSelection("pretrain", indexed_data=True),
    "squad": _DatasetSelection("finetune"),
    "squad-packed": _DatasetSelection(
        "finetune",
        packed_sequence=True,
    ),
    "openmathinstruct2": _DatasetSelection(
        "finetune",
    ),
    "openmathinstruct2-thinking": _DatasetSelection(
        "finetune",
        packed_sequence=True,
        thinking_format=True,
    ),
    "gsm8k": _DatasetSelection("finetune"),
    "local-jsonl": _DatasetSelection("finetune"),
    "preloaded-vlm": _DatasetSelection("finetune"),
}


def _none_or_int(value: str) -> int | None:
    """Parse an integer CLI value while accepting ``none``."""
    if value.lower() == "none":
        return None
    return int(value)


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
        required=True,
        help="Training mode.",
    )
    selection.add_argument(
        "--step-func",
        "--step_func",
        default="llm_step",
        dest="step_func",
        help="Forward-step registry name; most LLM recipes use the default.",
    )

    recipe_args = parser.add_argument_group("Recipe construction")
    recipe_args.add_argument("--seq-length", "--seq_length", type=int, dest="seq_length")
    recipe_args.add_argument("--hf-path", "--hf_path", dest="hf_path")

    training = parser.add_argument_group("Training")
    training.add_argument("--max-steps", "--max_steps", type=int, dest="max_steps")
    training.add_argument("--global-batch-size", "--global_batch_size", type=int, dest="global_batch_size")
    training.add_argument("--micro-batch-size", "--micro_batch_size", type=int, dest="micro_batch_size")
    training.add_argument("--eval-interval", "--eval_interval", type=int, dest="eval_interval")
    training.add_argument("--eval-iters", "--eval_iters", type=int, dest="eval_iters")
    training.add_argument(
        "--distributed-timeout-minutes",
        "--distributed_timeout_minutes",
        type=int,
        dest="distributed_timeout_minutes",
    )
    training.add_argument("--lr", type=float)
    training.add_argument("--min-lr", "--min_lr", type=float, dest="min_lr")
    training.add_argument("--warmup-iters", "--warmup_iters", type=int, dest="warmup_iters")
    training.add_argument("--lr-decay-iters", "--lr_decay_iters", type=int, dest="lr_decay_iters")
    training.add_argument("--log-interval", "--log_interval", type=int, dest="log_interval")

    data = parser.add_argument_group("Data")
    data.add_argument("--dataset", choices=sorted(DATASETS), help="Dataset name.")
    data.add_argument(
        "--dataset-path",
        "--dataset_path",
        action="append",
        default=[],
        dest="dataset_paths",
        help="Indexed dataset prefix or directory; repeat for multiple paths.",
    )
    data.add_argument(
        "--dataset-cache",
        "--dataset_cache",
        dest="dataset_cache",
        help="Directory for dataset index files.",
    )
    data.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        dest="tokenizer_type",
        choices=["NullTokenizer", "HuggingFaceTokenizer", "SentencePieceTokenizer"],
    )
    data.add_argument("--tokenizer-model", "--tokenizer_model", dest="tokenizer_model")
    data.add_argument("--vocab-size", "--vocab_size", type=int, default=32000, dest="vocab_size")

    parallelism = parser.add_argument_group("Parallelism")
    parallelism.add_argument("--tp", type=int, dest="tensor_model_parallel_size")
    parallelism.add_argument("--pp", type=int, dest="pipeline_model_parallel_size")
    parallelism.add_argument("--cp", type=int, dest="context_parallel_size")
    parallelism.add_argument(
        "--vp",
        type=_none_or_int,
        default=-1,
        dest="virtual_pipeline_model_parallel_size",
    )
    parallelism.add_argument("--ep", type=int, dest="expert_model_parallel_size")
    parallelism.add_argument("--etp", type=int, dest="expert_tensor_parallel_size")

    checkpoint = parser.add_argument_group("Checkpointing")
    checkpoint.add_argument("--from", dest="pretrained_checkpoint", help="Checkpoint used to initialize training.")
    checkpoint.add_argument("--save-dir", "--save_dir", dest="save_dir")
    checkpoint.add_argument("--load-dir", "--load_dir", dest="load_dir")
    checkpoint.add_argument("--save-interval", "--save_interval", type=int, dest="save_interval")
    checkpoint.add_argument("--most-recent-k", "--most_recent_k", type=int, dest="most_recent_k")

    logging_args = parser.add_argument_group("Logging")
    logging_args.add_argument("--wandb-project", "--wandb_project", dest="wandb_project")
    logging_args.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity")
    logging_args.add_argument("--wandb-name", "--wandb_name", dest="wandb_name")
    logging_args.add_argument("--wandb-dir", "--wandb_dir", dest="wandb_dir")

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--dry-run", "--dry_run", action="store_true", dest="dryrun")
    runtime.add_argument("--save-config", "--save_config", dest="save_config_filepath")
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
        seq_length=args.seq_length,
        hf_path=args.hf_path,
        source="recipes",
    )


def _indexed_prefix(path: Path) -> Path:
    """Normalize an indexed-data filename to its prefix."""
    return path.with_suffix("") if path.suffix in {".bin", ".idx"} else path


def _resolve_indexed_paths(dataset_name: str, raw_paths: list[str]) -> list[str]:
    """Validate indexed dataset paths for a named pretraining dataset."""
    prefixes: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            prefixes.extend(_indexed_prefix(bin_path) for bin_path in sorted(path.glob("*.bin")))
        else:
            prefixes.append(_indexed_prefix(path))

    valid: list[str] = []
    invalid: list[str] = []
    for prefix in dict.fromkeys(prefixes):
        if Path(f"{prefix}.bin").is_file() and Path(f"{prefix}.idx").is_file():
            valid.append(str(prefix))
        else:
            invalid.append(str(prefix))

    if invalid:
        raise ValueError(f"Indexed dataset prefixes must have matching .bin and .idx files: {', '.join(invalid)}")
    if not valid:
        raise ValueError(f"Dataset '{dataset_name}' requires at least one --dataset-path.")
    return valid


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

    context_parallel_size = args.context_parallel_size or getattr(
        getattr(recipe, "model", None), "context_parallel_size", 1
    )
    recipe = apply_public_dataset_override(
        recipe,
        dataset_name=args.dataset,
        packed_sequence=selection.packed_sequence,
        pad_seq_to_mult=max(1, 2 * context_parallel_size)
        if selection.thinking_format and context_parallel_size > 1
        else 1,
        seq_length=args.seq_length,
        cli_overrides=cli_overrides,
    )
    if selection.indexed_data:
        recipe.dataset.data_path = _resolve_indexed_paths(args.dataset, args.dataset_paths)
        if args.dataset_cache is not None:
            recipe.dataset.path_to_cache = args.dataset_cache
    return recipe


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse public arguments and trailing ConfigContainer overrides."""
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    return args, _collect_overrides(parser, unknown)


def main(argv: list[str] | None = None) -> None:
    """Load, configure, and execute one library recipe."""
    logging.basicConfig(level=logging.INFO)
    args, cli_overrides = parse_args(argv)

    recipe = _load_selected_recipe(args)
    recipe = _apply_dataset(recipe, args, cli_overrides)
    recipe = apply_launcher_overrides(recipe, args, recipe_source="recipes")
    recipe = apply_determinism(recipe, deterministic=args.deterministic)
    recipe = apply_cli_overrides(recipe, cli_overrides)
    recipe = sync_model_dataset_sequence_length(recipe)

    mode = _train_mode(args.mode)
    forward_step = load_forward_step(args.step_func, mode=mode)
    run_config(
        config=recipe,
        mode=mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        save_config_filepath=args.save_config_filepath,
        dump_environment=args.dump_env,
    )


if __name__ == "__main__":
    main()
