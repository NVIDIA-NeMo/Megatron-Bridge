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
    Pretrain:
        torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_pretrain_config

    Finetune:
        torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_finetune_config

    With CLI overrides:
        torchrun --nproc_per_node=8 run_recipe.py \
            --recipe llama32_1b_pretrain_config \
            train.train_iters=5000 \
            optimizer.lr=0.0003

    With VLM step function:
        torchrun --nproc_per_node=8 run_recipe.py \
            --recipe qwen25_vl_finetune_config \
            --step_func vlm_step

    With packed sequences and custom sequence length:
        torchrun --nproc_per_node=8 run_recipe.py \
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
from typing import Callable

import megatron.bridge.recipes as recipes
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step as gpt_forward_step
from megatron.bridge.training.llava_step import forward_step as llava_forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step


STEP_FUNCTIONS: dict[str, Callable] = {
    "gpt_step": gpt_forward_step,
    "vlm_step": vlm_forward_step,
    "qwen3_vl_step": qwen3_vl_forward_step,
    "llava_step": llava_forward_step,
}

TRAIN_MODES = {
    "pretrain": pretrain,
    "finetune": finetune,
}

# Error message constants
ERR_UNKNOWN_STEP = "Unknown step type: {step_type}. Choose from: {choices}"
ERR_INFER_MODE_FAILED = (
    "Unable to infer training mode from recipe name. "
    "Please include 'pretrain' or 'finetune' (or 'sft'/'peft') in the recipe name, "
    "or pass --mode explicitly."
)


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
        choices=sorted(TRAIN_MODES.keys()),
        help="Training mode (optional). If omitted, inferred from recipe name.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        required=True,
        help="Recipe function name (e.g., llama32_1b_pretrain_config, gemma3_1b_finetune_config)",
    )
    parser.add_argument(
        "--step_func",
        type=str,
        default="gpt_step",
        choices=sorted(STEP_FUNCTIONS.keys()),
        help="Step function: gpt_step (text-only), vlm_step (vision-language), or llava_step (LLaVA models)",
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
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def load_recipe(
    recipe_name: str,
    peft_scheme: str | None,
    packed_sequence: bool = False,
    seq_length: int | None = None,
) -> ConfigContainer:
    """
    Load recipe by name from megatron.bridge.recipes.

    Args:
        recipe_name: Full recipe function name (e.g., 'llama32_1b_pretrain_config')
        peft_scheme: PEFT scheme to use ('lora', 'dora', or None)
        packed_sequence: Enable packed sequence training (default: False)
        seq_length: Sequence length for training (optional)

    Returns:
        ConfigContainer from calling the recipe

    Raises:
        AttributeError: If recipe not found
    """
    if not hasattr(recipes, recipe_name):
        raise AttributeError(
            f"Recipe '{recipe_name}' not found in megatron.bridge.recipes.\n"
            f"Make sure the recipe name is correct and the recipe is exported in its family __init__.py.\n"
            f"Example recipe names: llama32_1b_pretrain_config, gemma3_1b_pretrain_config, qwen3_8b_pretrain_config"
        )

    config_builder = getattr(recipes, recipe_name)

    # Inspect the recipe's signature to determine which arguments it accepts
    try:
        sig = inspect.signature(config_builder)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        accepts_peft = "peft" in params or has_var_keyword
        accepts_packed_sequence = "packed_sequence" in params or has_var_keyword
        accepts_seq_length = "seq_length" in params or has_var_keyword
    except (ValueError, TypeError):
        # If signature inspection fails, fallback conservatively
        accepts_peft = True  # peft is widely supported, try passing it
        accepts_packed_sequence = False  # new parameter, don't pass if unsure
        accepts_seq_length = False  # new parameter, don't pass if unsure

    # Build kwargs dynamically based on what the recipe accepts
    kwargs = {}
    if accepts_peft:
        kwargs["peft"] = peft_scheme
    if accepts_packed_sequence and packed_sequence:
        kwargs["packed_sequence"] = packed_sequence
    if accepts_seq_length and seq_length is not None:
        kwargs["seq_length"] = seq_length

    try:
        return config_builder(**kwargs)
    except TypeError:
        # Fallback if the kwargs are not accepted despite signature inspection
        return config_builder()


def load_forward_step(step_type: str) -> Callable:
    """Load forward_step function based on the requested step type."""
    step_key = step_type.lower()
    if step_key not in STEP_FUNCTIONS:
        raise ValueError(ERR_UNKNOWN_STEP.format(step_type=step_type, choices=", ".join(STEP_FUNCTIONS)))
    return STEP_FUNCTIONS[step_key]


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

    config: ConfigContainer = load_recipe(
        args.recipe,
        args.peft_scheme,
        args.packed_sequence,
        args.seq_length,
    )

    config = process_config_with_overrides(
        config,
        cli_overrides=cli_overrides or None,
    )

    mode = args.mode or infer_train_mode(args.recipe)

    forward_step = load_forward_step(args.step_func)
    train_func = TRAIN_MODES[mode]
    train_func(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
