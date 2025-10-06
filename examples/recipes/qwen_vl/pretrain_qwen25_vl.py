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
Qwen2.5-VL Pretraining Script with YAML and CLI Configuration Overrides.

This mirrors the Llama example flow and uses the Qwen-VL recipe's pretrain_config().

Examples:
    Basic usage with default configuration:
        $ torchrun --nproc_per_node=8 pretrain_qwen25_vl.py
        $ torchrun --nproc_per_node=8 examples/recipes/qwen_vl/pretrain_qwen25_vl.py


    Using a custom YAML config file:
        $ torchrun --nproc_per_node=8 pretrain_qwen25_vl.py --config-file conf/qwen25_vl_pretrain_override_example.yaml

    CLI overrides:
        $ torchrun --nproc_per_node=8 pretrain_qwen25_vl.py model.tensor_model_parallel_size=4 train.train_iters=100000
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen_vl.qwen25_vl import pretrain_config
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
    parser = argparse.ArgumentParser(
        description="Pretrain Qwen2.5-VL with YAML and CLI overrides",
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
        "--use-preloaded",
        action="store_true",
        help="Use preloaded dataset provider (enabled automatically when --data-path is set).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Qwen2.5-VL Pretraining Script with YAML & CLI Overrides")
    logger.info("-----------------------------------------------------------------------")

    # Determine dataset type based on CLI flag (overrides) or fall back to auto-detect
    use_preloaded_flag = bool(args.data_path) or bool(getattr(args, "use_preloaded", False))
    dataset_type = args.dataset_type or ("preloaded" if use_preloaded_flag else "mock")

    cfg: ConfigContainer = pretrain_config(
        dataset_type=dataset_type,
        train_data_path=args.data_path,
        valid_data_path=None,
        test_data_path=None,
        image_folder=args.image_folder,
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
