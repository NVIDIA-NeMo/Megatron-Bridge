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
Generic Pretrain Script for GPT-based Models

This script works with any model family that uses GPT-style training
(Llama, Gemma, Qwen, GPT, etc.). It dynamically loads recipes and supports
YAML configuration files and CLI overrides.

Usage:
    Basic usage:
        torchrun --nproc_per_node=8 pretrain_gpt.py \
            --recipe llama32_1b_pretrain_config

    With YAML config:
        torchrun --nproc_per_node=8 pretrain_gpt.py \
            --recipe llama3_8b_pretrain_config \
            --config-file conf/my_config.yaml

    With CLI overrides:
        torchrun --nproc_per_node=8 pretrain_gpt.py \
            --recipe llama32_1b_pretrain_config \
            train.train_iters=5000 \
            optimizer.lr=0.0003

    Combined:
        torchrun --nproc_per_node=8 pretrain_gpt.py \
            --recipe gemma3_1b_pretrain_config \
            --config-file conf/my_config.yaml \
            train.train_iters=10000

Recipe Arguments:
    Generic scripts call recipes with no arguments: recipe().

    If you need to pass arguments to the recipe constructor
    (e.g., custom parallelism at build time), create a custom script.
"""

import argparse
import logging
import sys

import megatron.bridge.recipes as recipes
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generic pretrain script for GPT-based models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--recipe",
        type=str,
        required=True,
        help="Recipe function name (e.g., llama32_1b_pretrain_config, gemma3_1b_pretrain_config)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML config file for overrides",
    )

    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def load_recipe(recipe_name: str) -> ConfigContainer:
    """
    Load recipe by name from megatron.bridge.recipes.

    Args:
        recipe_name: Full recipe function name (e.g., 'llama32_1b_pretrain_config')

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
    return config_builder()


def main() -> None:
    """Run GPT pretraining."""
    args, cli_overrides = parse_args()

    config: ConfigContainer = load_recipe(args.recipe)

    try:
        config = process_config_with_overrides(
            config,
            config_file=args.config_file,
            cli_overrides=cli_overrides or None,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
