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
"""Unified training launcher for library recipes and flat performance recipes."""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recipe_runner import (  # noqa: E402
    PRECISION_ALIASES,
    RecipeSource,
    apply_cli_overrides,
    apply_determinism,
    apply_launcher_overrides,
    infer_train_mode,
    load_forward_step,
    load_library_recipe_by_family,
    load_perf_recipe_by_name,
    load_recipe,
    resolve_recipe_source,
    run_config,
    sync_model_dataset_sequence_length,
)

from megatron.bridge.recipes.utils.dataset_utils import (  # noqa: E402
    DATASET_TYPES,
    apply_dataset_override,
    infer_mode_from_dataset,
)
from megatron.bridge.recipes.utils.finetune_utils import (  # noqa: E402
    default_openmathinstruct2_thinking_packed_config,
)


TrainMode = Literal["pretrain", "finetune"]
UserMode = Literal["pretrain", "sft", "peft", "lora", "dora"]


@dataclass(frozen=True)
class _DatasetSelection:
    """Resolved user-facing dataset selection."""

    dataset_type: str
    preset: str | None = None
    packed_sequence: bool = False
    thinking_format: bool = False


DATASET_PRESETS = {
    "mock": _DatasetSelection("llm-pretrain-mock"),
    "dclm": _DatasetSelection("llm-pretrain"),
    "rp2": _DatasetSelection("llm-pretrain"),
    "c4": _DatasetSelection("llm-pretrain"),
    "squad": _DatasetSelection("llm-finetune", preset="squad"),
    "squad-packed": _DatasetSelection("llm-finetune", preset="squad", packed_sequence=True),
    "squad_packed": _DatasetSelection("llm-finetune", preset="squad", packed_sequence=True),
    "openmathinstruct2": _DatasetSelection("llm-finetune", preset="openmathinstruct2"),
    "openmathinstruct2-thinking": _DatasetSelection(
        "llm-finetune",
        packed_sequence=True,
        thinking_format=True,
    ),
    "openmathinstruct2_thinking": _DatasetSelection(
        "llm-finetune",
        packed_sequence=True,
        thinking_format=True,
    ),
    "gsm8k": _DatasetSelection("llm-finetune", preset="gsm8k"),
}

DATASET_ALIASES = {
    "mock": "llm-pretrain-mock",
    "rp2": "llm-pretrain",
    "c4": "llm-pretrain",
    "squad": "llm-finetune",
    "squad_packed": "llm-finetune",
}

STEP_BY_DOMAIN = {
    "llm": "gpt_step",
    "vlm": "vlm_step",
    "qwen3vl": "qwen3_vl_step",
    "diffusion": "wan_step",
}


def _none_or_int(value: str) -> int | None:
    """Parse an integer CLI value while accepting 'None'."""
    if value.lower() == "none":
        return None
    return int(value)


def _normalize_task(task: str | None) -> str:
    """Normalize task aliases used by recipes and mode inference."""
    if task is None:
        return "pretrain"
    lowered = task.lower()
    if lowered == "finetune":
        return "sft"
    if lowered in {"lora", "dora"}:
        return "peft"
    return lowered


def _mode_from_task(task: str) -> TrainMode:
    """Map a normalized task name to the Bridge training loop mode."""
    return "pretrain" if task == "pretrain" else "finetune"


def _task_from_recipe_name(recipe_name: str) -> str | None:
    """Return the public task encoded in a conventional recipe name."""
    padded_name = f"_{recipe_name.lower().strip('_')}_"
    tasks = set()
    if "_pretrain_" in padded_name:
        tasks.add("pretrain")
    if "_sft_" in padded_name or "_finetune_" in padded_name:
        tasks.add("sft")
    if any(marker in padded_name for marker in ("_peft_", "_lora_", "_dora_")):
        tasks.add("peft")
    return tasks.pop() if len(tasks) == 1 else None


def _parse_key_value(value: str) -> str:
    """Validate KEY=VALUE override syntax for --set."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {value!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the unified recipe launcher."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Megatron Bridge recipes from megatron.bridge.recipes or flat performance recipes "
            "from megatron.bridge.perf_recipes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    selection = parser.add_argument_group("Recipe selection")
    selection.add_argument("--recipe", help="Full recipe function name to load.")
    selection.add_argument(
        "--source",
        choices=["auto", "recipes", "perf_recipes"],
        default="auto",
        help="Recipe namespace for --recipe or selector mode.",
    )
    selection.add_argument(
        "--use-recipes",
        "--use_recipes",
        action="store_true",
        dest="use_recipes",
        help="Compatibility alias for --source recipes.",
    )
    selection.add_argument(
        "--family",
        "--model-family-name",
        "--model_family_name",
        "-m",
        dest="model_family_name",
        help="Library recipe family, e.g. llama, qwen, flux. Optional for perf_recipes.",
    )
    selection.add_argument(
        "--model",
        "--model-recipe-name",
        "--model_recipe_name",
        "-mr",
        dest="model_recipe_name",
        help="Model recipe stem, e.g. llama32_1b, qwen3_8b, deepseek_v3.",
    )
    selection.add_argument(
        "--task",
        choices=["pretrain", "sft", "finetune", "peft", "lora", "dora"],
        default=None,
        help="Compatibility alias for --mode.",
    )
    selection.add_argument(
        "--gpus",
        "--num-gpus",
        "--num_gpus",
        "-ng",
        dest="num_gpus",
        type=int,
        help="Total GPU count for flat perf recipe names and dry-run topology checks.",
    )
    selection.add_argument("--gpu", "-g", help="GPU target in flat recipe names, e.g. h100, gb200.")
    selection.add_argument(
        "--dtype",
        "--compute-dtype",
        "--compute_dtype",
        dest="compute_dtype",
        default="bf16",
        choices=sorted(PRECISION_ALIASES),
        help="Precision token used by flat perf recipe names.",
    )
    selection.add_argument(
        "--variant",
        "--config-variant",
        "--config_variant",
        dest="config_variant",
        help="Optional non-canonical flat perf recipe variant suffix.",
    )
    selection.add_argument(
        "--mode",
        choices=["pretrain", "sft", "finetune", "peft", "lora", "dora"],
        help="User-facing training mode. lora/dora select a PEFT recipe and scheme.",
    )
    selection.add_argument(
        "--domain",
        choices=["llm", "vlm", "qwen3vl", "diffusion"],
        default="llm",
        help="Training domain used to pick the default forward step.",
    )
    selection.add_argument(
        "--step",
        "--step-func",
        "--step_func",
        dest="step_func",
        help="Forward step key. Defaults from --domain.",
    )

    recipe_kwargs = parser.add_argument_group("Recipe constructor shortcuts")
    recipe_kwargs.add_argument("--peft-scheme", "--peft_scheme", dest="peft_scheme")
    recipe_kwargs.add_argument("--packed-sequence", "--packed_sequence", action="store_true", dest="packed_sequence")
    recipe_kwargs.add_argument("--seq-length", "--seq_length", type=int, dest="seq_length")
    recipe_kwargs.add_argument("--hf-path", "--hf_path", dest="hf_path")

    training = parser.add_argument_group("Common training overrides")
    training.add_argument("--max-steps", "--max_steps", type=int, dest="max_steps")
    training.add_argument("--global-batch-size", "--global_batch_size", type=int, dest="global_batch_size")
    training.add_argument("--micro-batch-size", "--micro_batch_size", type=int, dest="micro_batch_size")
    training.add_argument("--eval-interval", "--eval_interval", type=int, dest="eval_interval")
    training.add_argument("--eval-iters", "--eval_iters", type=int, dest="eval_iters")
    training.add_argument("--distributed-timeout-minutes", "--distributed_timeout_minutes", type=int)
    training.add_argument("--lr", type=float)
    training.add_argument("--min-lr", "--min_lr", type=float, dest="min_lr")
    training.add_argument("--warmup-iters", "--warmup_iters", type=int, dest="warmup_iters")
    training.add_argument("--lr-decay-iters", "--lr_decay_iters", type=int, dest="lr_decay_iters")
    training.add_argument("--log-interval", "--log_interval", type=int, dest="log_interval")

    data = parser.add_argument_group("Dataset and tokenizer overrides")
    data.add_argument(
        "--dataset",
        help=(
            "Dataset preset (dclm, openmathinstruct2, openmathinstruct2-thinking, squad, gsm8k, mock) "
            "or a compatibility llm-/vlm- dataset type."
        ),
    )
    data.add_argument(
        "--data",
        choices=sorted(DATASET_ALIASES),
        help="Compatibility alias for common dataset presets: mock, rp2, c4, squad, squad_packed.",
    )
    data.add_argument(
        "--dataset-preset",
        "--dataset_preset",
        dest="dataset_preset",
        choices=["squad", "openmathinstruct2", "gsm8k"],
        help="Preset for --dataset llm-finetune.",
    )
    data.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        dest="dataset_paths",
        help="Preprocessed dataset prefix or directory. Repeat for multiple DCLM prefixes.",
    )
    data.add_argument(
        "--dataset-cache",
        dest="dataset_cache",
        help="Dataset index/cache directory. DCLM also reads DCLM_CACHE when omitted.",
    )
    data.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        dest="tokenizer_type",
        choices=["NullTokenizer", "HuggingFaceTokenizer", "SentencePieceTokenizer"],
    )
    data.add_argument("--tokenizer-model", "--tokenizer_model", dest="tokenizer_model")
    data.add_argument("--vocab-size", "--vocab_size", type=int, default=32000, dest="vocab_size")

    parallelism = parser.add_argument_group("Parallelism shortcuts")
    parallelism.add_argument(
        "--tp",
        "--tensor-model-parallel-size",
        "--tensor_model_parallel_size",
        type=int,
        dest="tensor_model_parallel_size",
    )
    parallelism.add_argument(
        "--pp",
        "--pipeline-model-parallel-size",
        "--pipeline_model_parallel_size",
        type=int,
        dest="pipeline_model_parallel_size",
    )
    parallelism.add_argument(
        "--cp",
        "--context-parallel-size",
        "--context_parallel_size",
        type=int,
        dest="context_parallel_size",
    )
    parallelism.add_argument(
        "--vp",
        "--virtual-pipeline-model-parallel-size",
        "--virtual_pipeline_model_parallel_size",
        type=_none_or_int,
        default=-1,
        dest="virtual_pipeline_model_parallel_size",
    )
    parallelism.add_argument(
        "--ep",
        "--expert-model-parallel-size",
        "--expert_model_parallel_size",
        type=int,
        dest="expert_model_parallel_size",
    )
    parallelism.add_argument(
        "--etp",
        "--expert-tensor-parallel-size",
        "--expert_tensor_parallel_size",
        type=int,
        dest="expert_tensor_parallel_size",
    )

    checkpoint = parser.add_argument_group("Checkpointing")
    checkpoint.add_argument(
        "--from",
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint",
        help="Checkpoint or Hugging Face model used to initialize SFT/PEFT.",
    )
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
    runtime.add_argument("--dry-run", "--dryrun", action="store_true", dest="dryrun")
    runtime.add_argument(
        "--save-config",
        "--save-config-filepath",
        "--save_config_filepath",
        dest="save_config_filepath",
    )
    runtime.add_argument("--deterministic", action="store_true")
    runtime.add_argument("--dump-env", "--dump_env", action="store_true", dest="dump_env")
    runtime.add_argument(
        "--set",
        action="append",
        type=_parse_key_value,
        default=[],
        dest="set_overrides",
        help="Additional ConfigContainer override in KEY=VALUE form. May be repeated.",
    )

    return parser


def _collect_overrides(parser: argparse.ArgumentParser, args: argparse.Namespace, unknown: list[str]) -> list[str]:
    """Collect --set and trailing Hydra-style overrides."""
    overrides = list(args.set_overrides)
    for value in unknown:
        if value.startswith("-"):
            parser.error(f"Unknown option: {value}")
        if "=" not in value:
            parser.error(f"Expected override in KEY=VALUE form, got {value!r}")
        overrides.append(value)

    if args.dataset_preset is not None:
        overrides.append(f"dataset.hf_dataset.dataset_name={args.dataset_preset}")
    return overrides


def _resolve_dataset(args: argparse.Namespace) -> _DatasetSelection | None:
    """Resolve a public dataset name or compatibility dataset type."""
    if args.dataset is not None:
        normalized = args.dataset.lower()
        if normalized in DATASET_PRESETS:
            return DATASET_PRESETS[normalized]
        if normalized in DATASET_TYPES:
            return _DatasetSelection(normalized, preset=args.dataset_preset)
        choices = sorted({*DATASET_PRESETS, *DATASET_TYPES})
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: {', '.join(choices)}")
    if args.data is None:
        return None
    return DATASET_PRESETS[args.data]


def _dclm_prefix_from_path(path: Path) -> Path:
    """Normalize a DCLM .bin/.idx file or prefix to the indexed-data prefix."""
    if path.suffix in {".bin", ".idx"}:
        return path.with_suffix("")
    return path


def _discover_dclm_prefixes(args: argparse.Namespace) -> list[str]:
    """Resolve and validate preprocessed DCLM prefixes without falling back to mock data."""
    raw_paths = list(args.dataset_paths)
    data_prefix = os.environ.get("DCLM_DATA_PREFIX")
    data_dir = os.environ.get("DCLM_DATA_DIR")
    if not raw_paths and data_prefix:
        raw_paths.append(data_prefix)
    if not raw_paths and data_dir:
        raw_paths.append(data_dir)

    prefixes: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            prefixes.extend(_dclm_prefix_from_path(bin_path) for bin_path in sorted(path.glob("*.bin")))
        else:
            prefixes.append(_dclm_prefix_from_path(path))

    valid_prefixes: list[str] = []
    invalid_prefixes: list[str] = []
    for prefix in dict.fromkeys(prefixes):
        bin_path = Path(f"{prefix}.bin")
        idx_path = Path(f"{prefix}.idx")
        if bin_path.is_file() and idx_path.is_file():
            valid_prefixes.append(str(prefix))
        else:
            invalid_prefixes.append(str(prefix))

    if invalid_prefixes:
        formatted = ", ".join(invalid_prefixes)
        raise ValueError(f"DCLM prefix(es) must have matching .bin and .idx files: {formatted}")
    if not valid_prefixes:
        raise ValueError(
            "Dataset 'dclm' requires preprocessed Megatron .bin/.idx files. "
            "Pass --dataset-path, set DCLM_DATA_PREFIX, or set DCLM_DATA_DIR."
        )
    return valid_prefixes


def _apply_dataset_selection(
    recipe: object,
    selection: _DatasetSelection,
    args: argparse.Namespace,
    cli_overrides: list[str],
) -> object:
    """Apply the resolved dataset selection to a recipe."""
    packed_sequence = args.packed_sequence or selection.packed_sequence
    dclm_prefixes = (
        _discover_dclm_prefixes(args) if args.dataset is not None and args.dataset.lower() == "dclm" else None
    )
    if selection.thinking_format:
        seq_length = args.seq_length or getattr(getattr(recipe, "model", None), "seq_length", 4096)
        context_parallel_size = args.context_parallel_size or getattr(
            getattr(recipe, "model", None), "context_parallel_size", 1
        )
        recipe.dataset = default_openmathinstruct2_thinking_packed_config(
            seq_length=seq_length,
            packed_sequence=True,
            pad_seq_to_mult=max(1, 2 * context_parallel_size) if context_parallel_size > 1 else 1,
        )
        return recipe

    if selection.preset is not None and args.dataset_preset is None:
        cli_overrides.append(f"dataset.hf_dataset.dataset_name={selection.preset}")

    recipe = apply_dataset_override(
        recipe,
        dataset_type=selection.dataset_type,
        packed_sequence=packed_sequence,
        seq_length=args.seq_length,
        cli_overrides=cli_overrides,
    )
    if dclm_prefixes is not None:
        recipe.dataset.data_path = dclm_prefixes
        dataset_cache = args.dataset_cache or os.environ.get("DCLM_CACHE")
        if dataset_cache is not None:
            recipe.dataset.path_to_cache = dataset_cache
    return recipe


def _load_named_recipe(args: argparse.Namespace) -> tuple[object, RecipeSource]:
    """Load a full recipe function name from the requested source."""
    source = resolve_recipe_source(args.recipe, source=args.source)
    recipe = load_recipe(
        args.recipe,
        args.peft_scheme,
        args.packed_sequence,
        args.seq_length,
        args.hf_path,
        source=source,
    )
    return recipe, source


def _require_selector_args(args: argparse.Namespace, *, include_family: bool) -> None:
    """Validate selector mode arguments."""
    required = ["model_recipe_name", "num_gpus", "gpu"]
    if include_family:
        required.insert(0, "model_family_name")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        formatted = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"Missing selector arguments: {formatted}. Pass them or use --recipe.")


def _load_selected_recipe(args: argparse.Namespace) -> tuple[object, RecipeSource]:
    """Load a recipe through the shorthand selector path."""
    task = _normalize_task(args.mode or args.task)
    source = args.source
    if source == "auto":
        source = "recipes" if args.model_family_name is not None else "perf_recipes"

    if source == "recipes":
        _require_selector_args(args, include_family=True)
        return (
            load_library_recipe_by_family(
                model_family_name=args.model_family_name,
                model_recipe_name=args.model_recipe_name,
                train_task=task,
                num_gpus=args.num_gpus,
                gpu=args.gpu,
                precision=args.compute_dtype,
                config_variant=args.config_variant,
                wandb_experiment_name=args.wandb_name,
                peft_scheme=args.peft_scheme,
            ),
            "recipes",
        )

    _require_selector_args(args, include_family=False)
    return (
        load_perf_recipe_by_name(
            model_recipe_name=args.model_recipe_name,
            task=task,
            num_gpus=args.num_gpus,
            gpu=args.gpu,
            precision=args.compute_dtype,
            config_variant=args.config_variant,
        ),
        "perf_recipes",
    )


def _infer_mode(args: argparse.Namespace, dataset: _DatasetSelection | None) -> TrainMode:
    """Infer whether to call pretrain() or finetune()."""
    requested_task = args.mode or args.task
    if requested_task is not None:
        normalized_task = _normalize_task(requested_task)
        requested_mode = _mode_from_task(normalized_task)
        if args.recipe is not None:
            recipe_task = _task_from_recipe_name(args.recipe)
            if recipe_task is not None and normalized_task != recipe_task:
                raise ValueError(f"Mode '{requested_task}' is incompatible with recipe '{args.recipe}'.")
            try:
                recipe_mode = infer_train_mode(args.recipe)
            except ValueError:
                recipe_mode = requested_mode
            if requested_mode != recipe_mode:
                raise ValueError(f"Mode '{requested_task}' is incompatible with recipe '{args.recipe}'.")
        if dataset is not None and requested_mode != infer_mode_from_dataset(dataset.dataset_type):
            raise ValueError(f"Mode '{requested_task}' is incompatible with dataset '{args.dataset or args.data}'.")
        return requested_mode
    if dataset is not None:
        return infer_mode_from_dataset(dataset.dataset_type)
    if args.recipe is not None:
        try:
            return infer_train_mode(args.recipe)
        except ValueError:
            if requested_task is None:
                raise
    return _mode_from_task(_normalize_task(requested_task))


def _default_step_name(args: argparse.Namespace) -> str:
    """Pick a forward step key from explicit CLI or domain."""
    if args.step_func is not None:
        return args.step_func
    if args.domain == "diffusion" and args.recipe is not None and args.recipe.startswith("flux"):
        return "flux_step"
    if args.domain == "diffusion" and args.model_family_name == "flux":
        return "flux_step"
    return STEP_BY_DOMAIN[args.domain]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher arguments and trailing ConfigContainer overrides."""
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.recipe is not None and args.model_recipe_name is not None:
        parser.error("--recipe already identifies the model; do not also pass --model")
    if args.mode is not None and args.task is not None and _normalize_task(args.mode) != _normalize_task(args.task):
        parser.error("--mode and --task select different training modes")
    requested_task = args.mode or args.task
    if requested_task in {"lora", "dora"}:
        if args.peft_scheme is not None and args.peft_scheme != requested_task:
            parser.error(f"--mode {requested_task} conflicts with --peft-scheme {args.peft_scheme}")
        args.peft_scheme = requested_task
    if args.use_recipes:
        if args.source == "perf_recipes":
            parser.error("--use_recipes cannot be combined with --source perf_recipes")
        args.source = "recipes"
    return args, _collect_overrides(parser, args, unknown)


def main(argv: list[str] | None = None) -> None:
    """Load a recipe, apply overrides, and run or dry-run training."""
    logging.basicConfig(level=logging.INFO)
    args, cli_overrides = parse_args(argv)

    recipe, recipe_source = _load_named_recipe(args) if args.recipe is not None else _load_selected_recipe(args)
    dataset = _resolve_dataset(args)
    mode = _infer_mode(args, dataset)

    if dataset is not None:
        recipe = _apply_dataset_selection(recipe, dataset, args, cli_overrides)

    recipe = apply_launcher_overrides(recipe, args, recipe_source=recipe_source)
    recipe = apply_determinism(recipe, deterministic=args.deterministic)
    recipe = apply_cli_overrides(recipe, cli_overrides)
    recipe = sync_model_dataset_sequence_length(recipe)

    forward_step = load_forward_step(_default_step_name(args), mode=mode)
    run_config(
        config=recipe,
        mode=mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        save_config_filepath=args.save_config_filepath,
        barrier_before_destroy=recipe_source == "perf_recipes",
        dryrun_num_gpus=args.num_gpus,
        dump_environment=args.dump_env,
    )


if __name__ == "__main__":
    main()
