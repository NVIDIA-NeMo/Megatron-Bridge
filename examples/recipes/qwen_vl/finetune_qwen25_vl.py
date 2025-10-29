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
Qwen2.5-VL Finetuning Script with YAML and CLI Configuration Overrides.

This mirrors the Llama example flow and uses the Qwen-VL recipe helpers.
You can pick a specific recipe via `--recipe`, e.g., `qwen25_vl_3b_finetune_config`,
`qwen25_vl_7b_finetune_config`, etc.

Examples:
    Loading pretrained weights (recommended for finetune):
        1) Import HF checkpoint to Megatron format:
           $ python examples/conversion/convert_checkpoints.py import \
               --hf-model Qwen/Qwen2.5-VL-3B-Instruct \
               --megatron-path /path/to/megatron_ckpt

        2) Run finetune using the imported checkpoint:
           $ torchrun --nproc_per_node=8 examples/recipes/qwen_vl/finetune_qwen25_vl.py \
               --pretrained-checkpoint /path/to/megatron_ckpt

    Using a custom YAML config file:
        $ torchrun --nproc_per_node=8 finetune_qwen25_vl.py --config-file conf/qwen25_vl_pretrain_override_example.yaml

    CLI overrides:
        $ torchrun --nproc_per_node=8 finetune_qwen25_vl.py model.tensor_model_parallel_size=4 train.train_iters=100000

    Selecting a specific recipe:
        $ torchrun --nproc_per_node=8 finetune_qwen25_vl.py --recipe qwen25_vl_7b_finetune_config
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen_vl import qwen25_vl as qwen_vl_recipes
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
DEFAULT_CONFIG_FILENAME: str = "qwen25_vl_pretrain_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse known script args and return remaining as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Finetune Qwen2.5-VL with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/qwen25_vl_pretrain_override_example.yaml",
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
        default="qwen25_vl_3b_finetune_config",
        help=(
            "Name of the recipe function in megatron.bridge.recipes.qwen_vl.qwen25_vl to use, "
            "e.g., qwen25_vl_3b_finetune_config, qwen25_vl_7b_finetune_config."
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

    logger.info("Megatron-Bridge Qwen2.5-VL Finetuning Script with YAML & CLI Overrides")
    logger.info("-----------------------------------------------------------------------")

    # Resolve the recipe function from the provided name
    recipe_name = getattr(args, "recipe", "qwen25_vl_3b_finetune_config")
    available_recipes = [name for name in dir(qwen_vl_recipes) if name.endswith("_finetune_config")]
    if not hasattr(qwen_vl_recipes, recipe_name):
        logger.error(
            "Unknown recipe '%s'. Available recipes: %s",
            recipe_name,
            ", ".join(sorted(available_recipes)),
        )
        sys.exit(2)
    pretrain_config = getattr(qwen_vl_recipes, recipe_name)

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

    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

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
