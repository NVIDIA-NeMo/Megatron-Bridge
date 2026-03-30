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

"""
Generic Training Script for GPT-based Models

This script works with any model family that uses GPT-style training
(Llama, Gemma, Qwen, GPT, etc.). It dynamically loads recipes and supports
CLI overrides.

Usage:
    Pretrain (single-GPU):
        uv run python run_recipe.py \
            --recipe llama32_1b_pretrain_config

    Pretrain (multi-GPU):
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_pretrain_config

    SFT (full finetuning):
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_sft_config

    PEFT (LoRA/DoRA):
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_peft_config

    With CLI overrides:
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_pretrain_config \
            train.train_iters=5000 \
            optimizer.lr=0.0003

    With VLM step function:
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe qwen25_vl_finetune_config \
            --step_func vlm_step

    With packed sequences and custom sequence length:
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_pretrain_config \
            --packed_sequence \
            --seq_length 2048

Recipe Arguments:
    Generic scripts call recipes with no arguments: recipe().

    If you need to pass arguments to the recipe constructor
    (e.g., custom parallelism at build time), create a custom script.
"""

import argparse
import inspect
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


STEP_FUNCTION_SPECS: dict[str, tuple[str, str]] = {
    "gpt_step": ("megatron.bridge.training.gpt_step", "forward_step"),
    "vlm_step": ("megatron.bridge.training.vlm_step", "forward_step"),
    "qwen3_omni_step": ("megatron.bridge.models.qwen_omni.qwen3_omni_step", "forward_step"),
    "qwen3_vl_step": ("megatron.bridge.models.qwen_vl.qwen3_vl_step", "forward_step"),
    "llava_step": ("megatron.bridge.training.llava_step", "forward_step"),
}

TRAIN_MODE_SPECS: dict[str, tuple[str, str]] = {
    "pretrain": ("megatron.bridge.training.pretrain", "pretrain"),
    "finetune": ("megatron.bridge.training.finetune", "finetune"),
}

# Error message constants
ERR_UNKNOWN_STEP = "Unknown step type: {step_type}. Choose from: {choices}"
ERR_INFER_MODE_FAILED = (
    "Unable to infer training mode from recipe name. "
    "Please include 'pretrain', 'sft', 'peft', or 'finetune' in the recipe name or pass --mode explicitly."
)

RECIPE_MODULE_HINTS: list[tuple[str, str]] = [
    ("qwen3_omni_", "megatron.bridge.recipes.qwen_omni.qwen3_omni"),
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generic training script for GPT-based models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=sorted(TRAIN_MODE_SPECS.keys()),
        help="Training mode (optional). If omitted, inferred from recipe name.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        required=True,
        help="Recipe function name (e.g., llama32_1b_pretrain_config, gemma3_1b_sft_config, gemma3_1b_peft_config)",
    )
    parser.add_argument(
        "--step_func",
        type=str,
        default="gpt_step",
        choices=sorted(STEP_FUNCTION_SPECS.keys()),
        help=(
            "Step function: gpt_step (text-only), vlm_step (vision-language), "
            "qwen3_omni_step (Qwen3-Omni thinker), qwen3_vl_step (Qwen3-VL), "
            "or llava_step (LLaVA models)"
        ),
    )
    parser.add_argument(
        "--peft_scheme",
        type=str,
        default=None,
        help="PEFT scheme to use: 'lora', 'dora', or None.",
    )
    parser.add_argument(
        "--packed_sequence",
        action="store_true",
        default=False,
        help="Enable packed sequence training (default: False)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path to model directory. "
        "Use a local path for more stable multinode training.",
    )
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def load_recipe(
    recipe_name: str,
    peft_scheme: str | None,
    packed_sequence: bool = False,
    seq_length: int | None = None,
    hf_path: str | None = None,
) -> "ConfigContainer":
    """
    Load recipe by name from megatron.bridge.recipes.

    Args:
        recipe_name: Full recipe function name (e.g., 'llama32_1b_pretrain_config')
        peft_scheme: PEFT scheme to use ('lora', 'dora', or None)
        packed_sequence: Enable packed sequence training (default: False)
        seq_length: Sequence length for training (optional)
        hf_path: HuggingFace model ID or local path to model directory (optional)

    Returns:
        ConfigContainer from calling the recipe

    Raises:
        AttributeError: If recipe not found
    """
    recipes_module = resolve_recipe_module(recipe_name)

    if not hasattr(recipes_module, recipe_name):
        raise AttributeError(
            f"Recipe '{recipe_name}' not found in {recipes_module.__name__}.\n"
            f"Make sure the recipe name is correct and the recipe is exported in its family __init__.py.\n"
            f"Example recipe names: llama32_1b_pretrain_config, gemma3_1b_pretrain_config, qwen3_8b_pretrain_config"
        )

    config_builder = getattr(recipes_module, recipe_name)

    # Inspect the recipe's signature to determine which arguments it accepts
    try:
        sig = inspect.signature(config_builder)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        accepts_peft = "peft" in params or has_var_keyword
        accepts_packed_sequence = "packed_sequence" in params or has_var_keyword
        accepts_seq_length = "seq_length" in params or has_var_keyword
        accepts_hf_path = "hf_path" in params or has_var_keyword
    except (ValueError, TypeError):
        # If signature inspection fails, fallback conservatively
        accepts_peft = True  # peft is widely supported, try passing it
        accepts_packed_sequence = False  # new parameter, don't pass if unsure
        accepts_seq_length = False  # new parameter, don't pass if unsure
        accepts_hf_path = False  # model-specific, don't pass if unsure

    # Build kwargs dynamically based on what the recipe accepts
    kwargs = {}
    if accepts_peft:
        kwargs["peft"] = peft_scheme
    if accepts_packed_sequence and packed_sequence:
        kwargs["packed_sequence"] = packed_sequence
    if accepts_seq_length and seq_length is not None:
        kwargs["seq_length"] = seq_length
    if accepts_hf_path and hf_path is not None:
        kwargs["hf_path"] = hf_path

    try:
        return config_builder(**kwargs)
    except TypeError:
        # Fallback if the kwargs are not accepted despite signature inspection
        return config_builder()


def resolve_recipe_module(recipe_name: str) -> ModuleType:
    """Resolve the narrowest recipe module for the requested recipe name."""
    for prefix, module_name in RECIPE_MODULE_HINTS:
        if recipe_name.startswith(prefix):
            return import_module(module_name)
    return import_module("megatron.bridge.recipes")


def load_forward_step(step_type: str) -> Callable:
    """Load forward_step function based on the requested step type."""
    step_key = step_type.lower()
    if step_key not in STEP_FUNCTION_SPECS:
        raise ValueError(ERR_UNKNOWN_STEP.format(step_type=step_type, choices=", ".join(STEP_FUNCTION_SPECS)))
    module_name, attr_name = STEP_FUNCTION_SPECS[step_key]
    return getattr(import_module(module_name), attr_name)


def load_train_mode(mode: str) -> Callable[[Any, Callable], None]:
    """Load the train entrypoint lazily for the requested mode."""
    if mode not in TRAIN_MODE_SPECS:
        raise ValueError(f"Unknown train mode: {mode}. Choose from: {', '.join(TRAIN_MODE_SPECS)}")
    module_name, attr_name = TRAIN_MODE_SPECS[mode]
    return getattr(import_module(module_name), attr_name)


def infer_train_mode(recipe_name: str) -> str:
    """Infer training mode from the recipe name."""
    lowered = recipe_name.lower()
    has_pretrain = "pretrain" in lowered
    has_finetune = "finetune" in lowered or "sft" in lowered or "peft" in lowered
    if has_pretrain ^ has_finetune:
        return "pretrain" if has_pretrain else "finetune"
    raise ValueError(ERR_INFER_MODE_FAILED)


def main() -> None:
    """Run GPT training (pretrain or finetune)."""
    args, cli_overrides = parse_args()

    config = load_recipe(
        args.recipe,
        args.peft_scheme,
        args.packed_sequence,
        args.seq_length,
        args.hf_path,
    )

    config = process_config_with_overrides(
        config,
        cli_overrides=cli_overrides or None,
    )

    mode = args.mode or infer_train_mode(args.recipe)

    forward_step = load_forward_step(args.step_func)
    train_func = load_train_mode(mode)
    train_func(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
