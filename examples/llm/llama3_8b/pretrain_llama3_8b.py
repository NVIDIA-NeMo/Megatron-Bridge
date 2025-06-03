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

import argparse
import logging
import os
import sys
from typing import List as TypingList
from typing import Tuple

from omegaconf import DictConfig, OmegaConf

# It's expected that NeMo-LM is installed or PYTHONPATH is set correctly.
from nemo_lm.models.utils import forward_step
from nemo_lm.recipes.llm.llama3_8b import pretrain_config
from nemo_lm.training.config import ConfigContainer
from nemo_lm.training.pretrain import megatron_pretrain
from nemo_lm.utils.omegaconf_utils import (
    OverridesError,
    apply_overrides_with_preservation,
    parse_hydra_overrides,
    safe_create_omegaconf_with_preservation,
)

logger: logging.Logger = logging.getLogger(__name__)


def parse_cli_args() -> Tuple[argparse.Namespace, TypingList[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Llama3 8B model using NeMo-LM with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file", type=str, default=None, help="Path to the YAML OmegaConf override file. Optional."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """Entry point for the script."""
    args, cli_overrides = parse_cli_args()

    logger.info("Nemo-LM Llama3 8B Pretraining Script with YAML & CLI Overrides")
    logger.info("------------------------------------------------------------------")

    # Load base configuration from the recipe as a Python dataclass
    cfg: ConfigContainer = pretrain_config()

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_callables = safe_create_omegaconf_with_preservation(cfg)

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
        logger.debug("✓ Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    if isinstance(final_overrides_as_dict, dict):
        # Apply overrides while preserving excluded callable fields
        apply_overrides_with_preservation(cfg, final_overrides_as_dict, excluded_callables)

    # Display final configuration
    logger.info("--- Final Merged Configuration ---")
    cfg.to_yaml()
    logger.info("----------------------------------")

    Start training
    logger.debug("Starting Megatron pretraining...")
    megatron_pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
