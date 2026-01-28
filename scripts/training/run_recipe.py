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

Recipe Arguments:
    Generic scripts call recipes with no arguments: recipe().

    If you need to pass arguments to the recipe constructor
    (e.g., custom parallelism at build time), create a custom script.
"""

import argparse
import importlib

import megatron.bridge.recipes as recipes
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


STEP_MODULES = {
    "gpt_step": "megatron.bridge.training.gpt_step",
    "vlm_step": "megatron.bridge.training.vlm_step",
    "llava_step": "megatron.bridge.training.llava_step",
}

TRAIN_MODES = {
    "pretrain": pretrain,
    "finetune": finetune,
}


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
        choices=sorted(STEP_MODULES.keys()),
        help="Step function: gpt_step (text-only), vlm_step (vision-language), or llava_step (LLaVA models)",
    )
    parser.add_argument(
        "--peft_scheme",
        type=str,
        default=None,
        help="PEFT scheme to use: 'lora', 'dora', or None.",
    )
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def load_recipe(recipe_name: str, peft_scheme: str | None) -> ConfigContainer:
    """
    Load recipe by name from megatron.bridge.recipes.

    Args:
        recipe_name: Full recipe function name (e.g., 'llama32_1b_pretrain_config')
        peft_scheme: PEFT scheme to use ('lora', 'dora', or None)

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
    return config_builder(peft=peft_scheme)


def load_forward_step(step_type: str):
    """Load forward_step function based on the requested step type."""
    step_key = step_type.lower()
    if step_key not in STEP_MODULES:
        raise ValueError(f"Unknown step type: {step_type}. Choose from: {', '.join(STEP_MODULES)}")
    module = importlib.import_module(STEP_MODULES[step_key])
    if not hasattr(module, "forward_step"):
        raise AttributeError(f"{STEP_MODULES[step_key]} does not define forward_step")
    return module.forward_step


def infer_train_mode(recipe_name: str) -> str:
    """Infer training mode from the recipe name."""
    lowered = recipe_name.lower()
    has_pretrain = "pretrain" in lowered
    has_finetune = "finetune" in lowered
    if has_pretrain ^ has_finetune:
        return "pretrain" if has_pretrain else "finetune"
    raise ValueError(
        "Unable to infer training mode from recipe name. "
        "Please include 'pretrain' or 'finetune' in the recipe name or pass --mode explicitly."
    )


def main() -> None:
    """Run GPT training (pretrain or finetune)."""
    args, cli_overrides = parse_args()

    config: ConfigContainer = load_recipe(args.recipe, args.peft_scheme)

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
