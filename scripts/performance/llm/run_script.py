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

import os
import sys
from os.path import basename, splitext

from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config as llama3_8b_pretrain_config
from megatron.bridge.recipes.llama.llama3_70b import pretrain_config as llama3_70b_pretrain_config
from megatron.bridge.recipes.llama.llama31_405b import pretrain_config as llama31_405b_pretrain_config
from megatron.bridge.training.mixed_precision import (
    bf16_mixed, 
    bf16_with_fp8_mixed, 
    bf16_with_fp8_current_scaling_mixed, 
    bf16_with_mxfp8_mixed, 
    bf16_with_fp8_subchannel_scaling_mixed,
)

import argparse
import logging

from omegaconf import OmegaConf

from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.training.comm_overlap import *

logger: logging.Logger = logging.getLogger(__name__)

def parse_cli_args():
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Llama3 8B model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        # default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/llama3_8b_pretrain_override_example.yaml",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        help="GPU to use for experiment.",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="bf16",
        help="Compute dtype to use for training. Default: bf16",
    )
    parser.add_argument(
        "--fp8_recipe",
        type=str,
        default="ds",
        help="FP8 recipe to use for training. Default: ds",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Model to use for experiment. Default: llama3",
        required=False,
        default="llama3",
    )
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        help="Model size to use for experiment. Default: 8b",
        required=False,
        default="8b",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def get_precision_config(compute_dtype: str, fp8_recipe: str):
    if compute_dtype == "fp8":
        if fp8_recipe == "ds":
            return bf16_with_fp8_mixed()
        elif fp8_recipe == "cs":
            return bf16_with_fp8_current_scaling_mixed()
        elif fp8_recipe == "mx":
            return bf16_with_mxfp8_mixed()
        elif fp8_recipe == "ss":
            return bf16_with_fp8_subchannel_scaling_mixed()
        else:
            raise ValueError(f"Invalid FP8 recipe: {fp8_recipe}")
    elif compute_dtype == "bf16":
        return bf16_mixed()
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype}")
    

comm_overlap_config_map = {
    "llama3_70b": {
        "h100": {
            "bf16": userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
        },
    },
    "llama31_405b": {
        "h100": {
            "bf16": userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
    }
}

def main():
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

    if (f"{args.model_name}_{args.model_size}" in comm_overlap_config_map and 
        args.gpu in comm_overlap_config_map[f"{args.model_name}_{args.model_size}"]):
        ub_cfg = comm_overlap_config_map[f"{args.model_name}_{args.model_size}"][args.gpu][args.compute_dtype]
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
    logger.info("--- Final Merged Configuration ---")
    recipe.to_yaml()
    logger.info("----------------------------------")

    # Start training
    logger.debug("Starting pretraining...")

    pretrain(config=recipe, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
