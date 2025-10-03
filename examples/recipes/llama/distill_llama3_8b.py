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
Llama3 8B Pretraining Script with YAML and CLI Configuration Overrides.

This script provides a flexible way to pretrain Llama3 8B models using Megatron-Bridge with support for
both YAML configuration files and command-line overrides using Hydra-style syntax.

Examples:
    Basic usage with default configuration:
        $ torchrun --nproc_per_node=8 pretrain_llama3_8b.py

    Using a custom YAML config file:
        $ torchrun --nproc_per_node=8 pretrain_llama3_8b.py --config-file my_custom_config.yaml

    Using CLI overrides only:
        $ torchrun --nproc_per_node=8 pretrain_llama3_8b.py model.tensor_model_parallel_size=4 train.train_iters=100000

    Combining YAML and CLI overrides (CLI takes precedence):
        $ torchrun --nproc_per_node=8 pretrain_llama3_8b.py --config-file conf/my_config.yaml \
        model.pipeline_dtype=torch.float16 \
        train.global_batch_size=512

Configuration Precedence:
    1. Base configuration from pretrain_config() recipe
    2. YAML overrides from --config-file (if provided)
    3. CLI overrides (highest precedence)

Supported Override Syntax:
    - Standard assignment: key=value
    - Nested assignment: section.subsection.key=value
    - Addition: +new_key=value
    - Deletion: ~key_to_remove
    - Type conversion: Automatic for basic types (int, float, bool, str)
    - Complex types: torch.dtype, enums, etc. are supported
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


# Define paths relative to this script's location
# Assumes this script (pretrain_llama3_8b.py) is in Megatron-Bridge/examples/recipes/llama/
# and the config is in a 'conf' subdirectory.
SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "llama3_8b_distill_override.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Llama3 8B model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/llama3_8b_pretrain_override_example.yaml",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Llama3 8B pretraining script.

    This function orchestrates the complete configuration workflow:
    1. Loads the base configuration from pretrain_config() recipe
    2. Applies YAML overrides from --config-file (if exists)
    3. Applies CLI overrides using Hydra-style syntax
    4. Starts Megatron pretraining with the final merged configuration

    Configuration merging preserves callable fields (like activation functions)
    and handles type conversions automatically.

    Examples of CLI usage:
        # Use default config with custom learning rate
        torchrun --nproc_per_node=8 pretrain_llama3_8b.py optimizer.lr=0.0002

        # Custom config file with additional overrides
        torchrun --nproc_per_node=8 pretrain_llama3_8b.py --config-file my_config.yaml train.train_iters=50000

        # Multiple overrides for distributed training
        torchrun --nproc_per_node=8 pretrain_llama3_8b.py \
            model.tensor_model_parallel_size=4 \
            model.pipeline_model_parallel_size=2 \
            train.global_batch_size=512
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Llama3 8B Pretraining Script with YAML & CLI Overrides")
    logger.info("------------------------------------------------------------------")

    # Load base configuration from the recipe as a Python dataclass
    cfg: ConfigContainer = pretrain_config()
    logger.info("Loaded base configuration")

    # Print configuration on rank 0
    if get_rank_safe() == 0:
        cfg.print_yaml()

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides if a config file is provided
    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("YAML overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Display final configuration
    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    # Start training
    logger.debug("Starting distillation...")
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
