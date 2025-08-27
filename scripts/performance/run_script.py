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

import logging
import os
import sys
from torch.distributed import get_rank

from omegaconf import OmegaConf

from megatron.bridge.recipes.llama.llama3_8b import pretrain_config as llama3_8b_pretrain_config
from megatron.bridge.recipes.llama.llama3_70b import pretrain_config as llama3_70b_pretrain_config
from megatron.bridge.recipes.llama.llama31_405b import pretrain_config as llama31_405b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from .argument_parser import parse_cli_args
from .utils.helpers import get_precision_config, COMM_OVERLAP_CONFIG_MAP


logger: logging.Logger = logging.getLogger(__name__)

def main():
    """Main function to run the pretraining/finetuning script."""
    args, cli_overrides = parse_cli_args()

    precision_config = get_precision_config(args.compute_dtype, args.fp8_recipe)

    if args.model_name == "llama3" and args.model_size == "8b":
        recipe = llama3_8b_pretrain_config(mock=True, precision_config=precision_config)
    elif args.model_name == "llama3" and args.model_size == "70b":
        recipe = llama3_70b_pretrain_config(mock=True, precision_config=precision_config)
    elif args.model_name == "llama31" and args.model_size == "405b":
        recipe = llama31_405b_pretrain_config(mock=True, precision_config=precision_config)
    else:
        raise ValueError(f"Model {args.model_name} {args.model_size} not supported")

    if (
        f"{args.model_name}_{args.model_size}" in COMM_OVERLAP_CONFIG_MAP
        and args.gpu in COMM_OVERLAP_CONFIG_MAP[f"{args.model_name}_{args.model_size}"]
    ):
        ub_cfg = COMM_OVERLAP_CONFIG_MAP[f"{args.model_name}_{args.model_size}"][args.gpu][args.compute_dtype]
        recipe.comm_overlap.tp_comm_overlap_cfg = ub_cfg

    if args.compute_dtype == "bf16":
        recipe.optimizer.use_precision_aware_optimizer = True

    recipe.to_yaml()
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(recipe)
    # Load and merge YAML overrides if a config file is provided
    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("YAML overrides merged successfully.")
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(recipe, final_overrides_as_dict, excluded_fields)
    # Display final configuration

    if get_rank() == 0:
        logger.info("--- Final Merged Configuration ---")
        recipe.to_yaml()
        logger.info("----------------------------------")
        # Start training
        logger.info("Starting pretraining...")

    pretrain(config=recipe, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
