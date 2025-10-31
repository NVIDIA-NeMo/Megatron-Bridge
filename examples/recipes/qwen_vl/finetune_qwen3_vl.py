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
uv run  python -m torch.distributed.run --nproc_per_node=8 examples/recipes/qwen_vl/finetune_qwen3_vl.py
# for moe model
uv run  python -m torch.distributed.run --nproc_per_node=8 --log-dir logs/   --redirects 3     --tee "0:3" examples/recipes/qwen_vl/finetune_qwen3_vl.py --recipe qwen3_vl_3b_active_30b_moe_finetune_config

Qwen3-VL Finetuning Script with YAML and CLI Configuration Overrides.

This script supports both dense and MoE Qwen3-VL models.
You can pick a specific recipe via `--recipe`.

Available recipes:
    - qwen3_vl_8b_finetune_config: Dense 8B model (Qwen/Qwen3-VL-8B-Instruct)
    - qwen3_vl_3b_active_30b_moe_finetune_config: MoE 30B model (Qwen/Qwen3-VL-30B-A3B-Instruct)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen_vl import qwen3vl as qwen3_vl_recipes
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.training.vlm_step import forward_step
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


SCRIPT_DIR: Path = Path(__file__).parent.resolve()


def get_default_config_file(recipe_name: str) -> Path:
    """Get the default config file path based on recipe name."""
    if "moe" in recipe_name.lower():
        config_filename = "qwen3_moe_vl_pretrain_override_example.yaml"
    else:
        config_filename = "qwen3_vl_pretrain_override_example.yaml"
    return SCRIPT_DIR / "conf" / config_filename


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse known script args and return remaining as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Finetune Qwen3-VL with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=(
            "Path to the YAML OmegaConf override file. "
            "If not specified, automatically selects based on recipe: "
            "qwen3_vl_pretrain_override_example.yaml for dense models, "
            "qwen3_moe_vl_pretrain_override_example.yaml for MoE models."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to JSON/JSONL dataset (preloaded conversation or legacy messages format).",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="Optional root for resolving relative image/video paths in dataset records.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["mock", "preloaded", "hf"],
        default=None,
        help=(
            "Dataset type to use: 'mock', 'preloaded', or 'hf'. "
            "If not set, auto-detects based on --data-path/--use-preloaded."
        ),
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="qwen3_vl_8b_finetune_config",
        help=(
            "Name of the recipe function in megatron.bridge.recipes.qwen_vl.qwen3vl to use:\n"
            "  - qwen3_vl_8b_finetune_config: Dense 8B model (default)\n"
            "  - qwen3_vl_3b_active_30b_moe_finetune_config: MoE 30B model with expert parallelism"
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to imported Megatron checkpoint directory to load before finetuning. "
            "Generate it with scripts/import_hf_ckpt.py."
        ),
    )
    parser.add_argument(
        "--use-preloaded",
        action="store_true",
        help="Use preloaded dataset provider (enabled automatically when --data-path is set).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Load the base VLM recipe config, apply YAML/CLI overrides, and start pretraining.
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Qwen3-VL Finetuning Script with YAML & CLI Overrides")
    logger.info("--------------------------------------------------------------------")

    # Resolve the recipe function from the provided name
    recipe_name = getattr(args, "recipe", "qwen3_vl_8b_finetune_config")
    available_recipes = [name for name in dir(qwen3_vl_recipes) if name.endswith("_finetune_config")]
    if not hasattr(qwen3_vl_recipes, recipe_name):
        logger.error(
            "Unknown recipe '%s'. Available recipes: %s",
            recipe_name,
            ", ".join(sorted(available_recipes)),
        )
        sys.exit(2)
    pretrain_config = getattr(qwen3_vl_recipes, recipe_name)

    # Determine dataset type based on CLI flag (overrides) or fall back to auto-detect
    use_preloaded_flag = bool(args.data_path) or bool(getattr(args, "use_preloaded", False))
    dataset_type = args.dataset_type or ("preloaded" if use_preloaded_flag else "mock")

    cfg: ConfigContainer = pretrain_config(
        dataset_type=dataset_type,
        train_data_path=args.data_path,
        valid_data_path=None,
        test_data_path=None,
        image_folder=args.image_folder,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    logger.info("Loaded base configuration")

    if get_rank_safe() == 0:
        cfg.print_yaml()

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Determine which config file to use
    config_file = args.config_file
    if config_file is None:
        # Auto-select config file based on recipe
        default_config_path = get_default_config_file(recipe_name)
        if default_config_path.exists():
            config_file = str(default_config_path)
            logger.debug(f"Auto-selected config file: {config_file}")

    if config_file and os.path.exists(config_file):
        logger.debug(f"Loading YAML overrides from: {config_file}")
        yaml_overrides_omega = OmegaConf.load(config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
    elif config_file:
        logger.warning(f"Config file specified but not found: {config_file}")

    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()

