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
Llama3.2 Knowledge Distillation Script with YAML and CLI Configuration Overrides.

This script provides an example of knowledge distillation using a Llama3.2-3B teacher model
to distill knowledge into a Llama3.2-1B student model using Megatron-Bridge with support for
both YAML configuration files and command-line overrides using Hydra-style syntax.

Examples:
    Basic usage with default configuration:
        $ torchrun --nproc_per_node=8 distill_llama32_3b-1b.py

    Using a custom YAML config file:
        $ torchrun --nproc_per_node=8 distill_llama32_3b-1b.py --config-file my_custom_config.yaml

    Using CLI overrides:
        $ torchrun --nproc_per_node=8 distill_llama32_3b-1b.py \
        model.tensor_model_parallel_size=4 \
        model.teacher.tensor_model_parallel_size=4 \
        train.train_iters=100000

    Combining YAML and CLI overrides (CLI takes precedence):
        $ torchrun --nproc_per_node=8 distill_llama32_3b-1b.py --config-file conf/my_config.yaml \
        model.pipeline_dtype=torch.float16 \
        model.teacher.pipeline_dtype=torch.float16 \
        train.global_batch_size=512

Configuration Precedence:
    1. Base configuration from student and teacher pretrain_config() recipes
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

from megatron.bridge import AutoBridge
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.recipes.llama import llama32_1b_pretrain_config, llama32_3b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


# Define paths relative to this script's location
# Assumes this script (distill_llama32_3b-1b.py) is in Megatron-Bridge/examples/distillation/llama/
# and the config is in a 'conf' subdirectory.
SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "llama32_3b-1b_distill_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation with Llama3.2 using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/llama32_3b-1b_distill_override_example.yaml",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entrypoint.

    This function orchestrates the complete configuration workflow:
    1. Loads the base student configuration (Llama3.2-1B) and teacher configuration (Llama3.2-3B)
    2. Wraps both in a DistillationProvider to create a unified distillation model
    3. Applies YAML overrides from --config-file (if exists)
    4. Applies CLI overrides using Hydra-style syntax
    5. Starts Megatron distillation with the final merged configuration

    The config.model structure contains student and teacher model providers:
    - config.model: The Llama3.2-1B student model configuration
    - config.model.teacher: The Llama3.2-3B teacher model configuration
    - config.model.kd_config: Knowledge distillation-specific settings

    Configuration merging preserves callable fields (like activation functions)
    and handles type conversions automatically.
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Llama3.2 3B-1B Distillation Script with YAML & CLI Overrides")
    logger.info("------------------------------------------------------------------")

    # Load base configurations as recipes and wrap provider for distillation mode.
    # The recipe functions do not accept a load_weights argument; use AutoBridge
    # directly to create providers that load HuggingFace weights.
    cfg: ConfigContainer = llama32_1b_pretrain_config()
    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B").to_megatron_provider(load_weights=True)
    teacher_cfg = llama32_3b_pretrain_config()
    teacher_cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-3B").to_megatron_provider(
        load_weights=True
    )
    kd_config = ModelOptDistillConfig()
    cfg.model = convert_to_distillation_provider(cfg.model, teacher_cfg.model, kd_config)
    logger.info("Loaded base student and teacher configurations")

    # Print configuration on rank 0
    if get_rank_safe() == 0:
        cfg.print_yaml()

    # Check if config file exists before processing
    if not os.path.exists(args.config_file):
        logger.error(f"Override YAML file not found: {args.config_file}")
        sys.exit(1)

    # Process configuration with YAML and CLI overrides
    cfg = process_config_with_overrides(
        config=cfg,
        config_filepath=args.config_file,
        cli_overrides=cli_overrides or None,
    )

    # Display final configuration
    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    # Start training
    logger.info("Starting distillation...")
    distill(config=cfg)


if __name__ == "__main__":
    main()
