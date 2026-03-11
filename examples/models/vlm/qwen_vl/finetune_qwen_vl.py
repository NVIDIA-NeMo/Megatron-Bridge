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
Unified Qwen-VL Finetuning Script with YAML and CLI Configuration Overrides.

Supports both Qwen2.5-VL and Qwen3-VL models (dense and MoE variants).
You can pick a specific recipe via `--recipe`.

Examples:
    Convert HF checkpoint to Megatron format:
        For Qwen2.5-VL:
            $  uv run python -m torch.distributed.run --nproc_per_node=1 examples/conversion/convert_checkpoints.py import \\
                --hf-model Qwen/Qwen2.5-VL-3B-Instruct \\
                --megatron-path ./logs/checkpoints/qwen25vl3b

        For Qwen3-VL (dense):
            $  uv run python -m torch.distributed.run --nproc_per_node=1 examples/conversion/convert_checkpoints.py import \\
                --hf-model Qwen/Qwen3-VL-8B-Instruct \\
                --megatron-path ./logs/checkpoints/qwen3vl8b

        For Qwen3-VL 30B (MoE):
            $  uv run python -m torch.distributed.run --nproc_per_node=1 examples/conversion/convert_checkpoints.py import \\
                --hf-model Qwen/Qwen3-VL-30B-A3B-Instruct \\
                --megatron-path ./logs/checkpoints/qwen3vl30b_moe

        For Qwen3-VL 235B (MoE):
            $  uv run python -m torch.distributed.run --nproc_per_node=1 examples/conversion/convert_checkpoints.py import \\
                --hf-model Qwen/Qwen3-VL-235B-A22B-Instruct \\
                --megatron-path ./logs/checkpoints/qwen3vl235b_moe

    Finetune using the imported checkpoint:
        Qwen2.5-VL 3B:
            $ uv run python -m torch.distributed.run --nproc_per_node=8 examples/models/vlm/qwen_vl/finetune_qwen_vl.py \\
                --recipe qwen25_vl_3b_finetune_config \\
                --pretrained-checkpoint ./logs/checkpoints/qwen25vl3b

        Qwen2.5-VL 7B:
            $  uv run python -m torch.distributed.run --nproc_per_node=8 examples/models/vlm/qwen_vl/finetune_qwen_vl.py \\
                --recipe qwen25_vl_7b_finetune_config \\
                --pretrained-checkpoint ./logs/checkpoints/qwen25_vl_7b

        Qwen3-VL 8B (dense):
            $ uv run python -m torch.distributed.run --nproc_per_node=8 examples/models/vlm/qwen_vl/finetune_qwen_vl.py \\
                --recipe qwen3_vl_8b_finetune_config \\
                --pretrained-checkpoint ./logs/checkpoints/qwen3_vl_8b

        Qwen3-VL 30B (MoE):
            $  uv run python -m torch.distributed.run --nproc_per_node=8 examples/models/vlm/qwen_vl/finetune_qwen_vl.py \\
                --recipe qwen3_vl_30b_a3b_finetune_config \\
                --pretrained-checkpoint ./logs/checkpoints/qwen3_vl_30b_a3b

        Qwen3-VL 235B (MoE):
            $  uv run python -m torch.distributed.run --nproc_per_node=8 examples/models/vlm/qwen_vl/finetune_qwen_vl.py \\
                --recipe qwen3_vl_235b_a22b_finetune_config \\
                --pretrained-checkpoint ./logs/checkpoints/qwen3_vl_235b_a22b

    Using a custom YAML config file:
        $  uv run python -m torch.distributed.run --nproc_per_node=8 finetune_qwen_vl.py \\
            --config-file conf/qwen25_vl_pretrain_override_example.yaml

    CLI overrides:
        $  uv run python -m torch.distributed.run --nproc_per_node=8 finetune_qwen_vl.py \\
            model.tensor_model_parallel_size=4 train.train_iters=100000

Available Recipes:
    Qwen2.5-VL:
        - qwen25_vl_3b_finetune_config: 3B model
        - qwen25_vl_7b_finetune_config: 7B model

    Qwen3-VL:
        - qwen3_vl_8b_finetune_config: Dense 8B model
        - qwen3_vl_30b_a3b_finetune_config: MoE 30B model with expert parallelism
        - qwen3_vl_235b_a22b_finetune_config: MoE 235B model with expert parallelism
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.recipes.qwen_vl import qwen3_vl as qwen3_vl_recipes
from megatron.bridge.recipes.qwen_vl import qwen25_vl as qwen25_vl_recipes
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "qwen3_vl_pretrain_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse known script args and return remaining as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Finetune Qwen-VL models (Qwen2.5-VL and Qwen3-VL) with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help=(
            "Path to the YAML OmegaConf override file. "
            "If not specified, automatically selects based on recipe:\n"
            "  - qwen25_vl_pretrain_override_example.yaml for Qwen2.5-VL models\n"
            "  - qwen3_vl_pretrain_override_example.yaml for Qwen3-VL dense models\n"
            "  - qwen3_moe_vl_pretrain_override_example.yaml for Qwen3-VL MoE models"
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
        choices=["mock", "preloaded", "hf", "energon"],
        default=None,
        help=(
            "Dataset type to use: 'mock', 'preloaded', 'hf', or 'energon'. "
            "If not set, auto-detects based on --data-path/--use-preloaded."
        ),
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="qwen25_vl_3b_finetune_config",
        help=(
            "Name of the recipe function to use:\n"
            "Qwen2.5-VL recipes:\n"
            "  - qwen25_vl_3b_finetune_config: 3B model (default)\n"
            "  - qwen25_vl_7b_finetune_config: 7B model\n"
            "Qwen3-VL recipes:\n"
            "  - qwen3_vl_8b_finetune_config: Dense 8B model\n"
            "  - qwen3_vl_30b_a3b_finetune_config: MoE 30B model\n"
            "  - qwen3_vl_235b_a22b_finetune_config: MoE 235B model"
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to imported Megatron checkpoint directory to load before finetuning. "
            "Generate it with examples/conversion/convert_checkpoints.py."
        ),
    )
    parser.add_argument(
        "--use-preloaded",
        action="store_true",
        help="Use preloaded dataset provider (enabled automatically when --data-path is set).",
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
        choices=["lora", "dora", "none"],
        help="Type of PEFT to use: 'lora', 'dora', or 'none' (full SFT). If not set, uses full SFT.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Load the base VLM recipe config, apply YAML/CLI overrides, and start pretraining.
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Qwen-VL Finetuning Script with YAML & CLI Overrides")
    logger.info("---------------------------------------------------------------------")

    recipe_name = getattr(args, "recipe", "qwen25_vl_3b_finetune_config")

    if recipe_name.startswith("qwen3"):
        recipe_module = qwen3_vl_recipes
        model_family = "Qwen3-VL"
        forward_step_func = qwen3_vl_forward_step
    elif recipe_name.startswith("qwen25"):  # qwen25
        recipe_module = qwen25_vl_recipes
        model_family = "Qwen2.5-VL"
        forward_step_func = vlm_forward_step
    else:
        raise ValueError(f"Unknown recipe name: {recipe_name}")

    pretrain_config = getattr(recipe_module, recipe_name)
    logger.info(f"Using {model_family} recipe: {recipe_name}")

    # Determine dataset type based on CLI flag (overrides) or fall back to auto-detect
    use_preloaded_flag = bool(args.data_path) or bool(getattr(args, "use_preloaded", False))
    dataset_type = args.dataset_type or ("preloaded" if use_preloaded_flag else "mock")

    # Build recipe kwargs
    recipe_kwargs = {
        "dataset_type": dataset_type,
        "train_data_path": args.data_path,
        "valid_data_path": None,
        "test_data_path": None,
        "image_folder": args.image_folder,
        "pretrained_checkpoint": args.pretrained_checkpoint,
    }

    # Add peft parameter if specified via --peft flag
    if args.peft is not None:
        recipe_kwargs["peft"] = args.peft

    cfg: ConfigContainer = pretrain_config(**recipe_kwargs)
    logger.info("Loaded base configuration")

    if get_rank_safe() == 0:
        cfg.print_yaml()

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Determine which config file to use
    config_file = args.config_file

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

    # check micro_batch_size and global_batch_size value consistency
    if dataset_type == "energon":
        assert cfg.train.micro_batch_size == cfg.dataset.micro_batch_size, (
            "value of cfg.dataset.micro_batch_size should be the same as cfg.train.micro_batch_size"
        )
        assert cfg.train.global_batch_size == cfg.dataset.global_batch_size, (
            "value of cfg.dataset.global_batch_size should be the same as cfg.train.global_batch_size"
        )

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    pretrain(config=cfg, forward_step_func=forward_step_func)


if __name__ == "__main__":
    main()
