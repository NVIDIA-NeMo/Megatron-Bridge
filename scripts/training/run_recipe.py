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
import sys
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


TrainMode = Literal["pretrain", "finetune"]

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
    if lowered == "lora":
        return "peft"
    return lowered


def _mode_from_task(task: str) -> TrainMode:
    """Map a normalized task name to the Bridge training loop mode."""
    return "pretrain" if task == "pretrain" else "finetune"


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
        choices=["pretrain", "sft", "finetune", "peft", "lora"],
        default=None,
        help="Training task for selector mode. Full --recipe names infer this when omitted.",
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
        choices=["pretrain", "finetune"],
        help="Override training loop selection when it cannot be inferred.",
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
        choices=DATASET_TYPES,
        help="Replace the recipe dataset with a standard Bridge dataset preset.",
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
    checkpoint.add_argument("--pretrained-checkpoint", "--pretrained_checkpoint", dest="pretrained_checkpoint")
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
        overrides.append(f"dataset.dataset_name={args.dataset_preset}")
    return overrides


def _resolve_dataset(args: argparse.Namespace) -> str | None:
    """Resolve canonical dataset type from --dataset or --data."""
    if args.dataset is not None:
        return args.dataset
    if args.data is None:
        return None
    if args.data == "squad_packed":
        args.packed_sequence = True
    return DATASET_ALIASES[args.data]


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
    task = _normalize_task(args.task)
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


def _infer_mode(args: argparse.Namespace, dataset: str | None) -> TrainMode:
    """Infer whether to call pretrain() or finetune()."""
    if args.mode is not None:
        return args.mode
    if dataset is not None:
        return infer_mode_from_dataset(dataset)
    if args.recipe is not None:
        try:
            return infer_train_mode(args.recipe)
        except ValueError:
            pass
    return _mode_from_task(_normalize_task(args.task))


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
        recipe = apply_dataset_override(
            recipe,
            dataset_type=dataset,
            packed_sequence=args.packed_sequence,
            seq_length=args.seq_length,
            cli_overrides=cli_overrides,
        )

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
