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
import dataclasses
import os
import sys
import logging
from typing import Any, Dict, TypeVar

from omegaconf import OmegaConf, DictConfig

from nemo_lm.models.utils import forward_step
from nemo_lm.recipes.llm.llama3_8b import pretrain_config
from nemo_lm.training.config import ConfigContainer
from nemo_lm.training.pretrain import megatron_pretrain

logger: logging.Logger = logging.getLogger(__name__)

DataclassInstance = TypeVar('DataclassInstance')


def apply_overrides_recursively(config_obj: DataclassInstance, overrides_dict: Dict[str, Any]) -> None:
    """
    Recursively applies overrides from a dictionary to a Python object (typically a dataclass instance).
    """
    if not dataclasses.is_dataclass(config_obj):
        return

    for key, value in overrides_dict.items():
        if not hasattr(config_obj, key):
            logger.warning(
                f"Key '{key}' in overrides not found in config object {type(config_obj).__name__}. Skipping."
            )
            continue

        current_attr: Any = getattr(config_obj, key)

        if isinstance(value, dict) and dataclasses.is_dataclass(current_attr):
            apply_overrides_recursively(current_attr, value)
        else:
            try:
                setattr(config_obj, key, value)
            except Exception as e:
                logger.warning(
                    f"Could not set attribute {type(config_obj).__name__}.{key} to value {value}. Error: {e}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Llama3 8B model using NeMo-LM")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the YAML OmegaConf override file (e.g., conf/llama3_8b_pretrain_override.yaml).",
    )

    args: argparse.Namespace = parser.parse_args()

    logger.info("Nemo-LM Llama3 8B Pretraining Script")
    logger.info("------------------------------------")

    # 1. Load base configuration from the recipe - this is our primary Python object
    logger.info(f"Loading base configuration using 'pretrain_config' from nemo_lm.recipes.llm.llama3_8b...")
    final_cfg_container: ConfigContainer = pretrain_config()

    # 2. Load YAML overrides into an OmegaConf dictionary
    logger.info(f"Loading overrides from YAML file: {args.config_file}")
    if not os.path.exists(args.config_file):
        logger.error(f"Override YAML file not found: {args.config_file}")
        sys.exit(1)

    overrides_py_dict: Dict[str, Any]
    try:
        yaml_overrides_omega: DictConfig = OmegaConf.load(args.config_file)
        overrides_py_dict = OmegaConf.to_container(yaml_overrides_omega, resolve=True)
    except Exception as e:
        logger.exception(f"Failed to load or convert YAML override file '{args.config_file}': {e}")
        sys.exit(1)

    # 3. Apply overrides directly to the Python config object (final_cfg_container)
    logger.info("Applying overrides to the configuration object...")
    if isinstance(overrides_py_dict, dict):
        apply_overrides_recursively(final_cfg_container, overrides_py_dict)
    else:
        logger.warning(
            f"Overrides file '{args.config_file}' did not result in a dictionary. Skipping override application."
        )

    # For debugging the final configuration:
    logger.info("--- Final Merged Configuration (YAML output of Python object) ---")
    final_cfg_container.to_yaml()
    logger.info("-----------------------------------------------------------------")

    logger.info("Starting Megatron pretraining...")
    megatron_pretrain(config=final_cfg_container, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
