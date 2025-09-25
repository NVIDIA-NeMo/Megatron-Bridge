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
from typing import Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2 import pretrain_config

# from megatron.bridge.recipes.mamba.nemotron_nano_9b_v2 import pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.common_utils import print_rank_last


logger: logging.Logger = logging.getLogger(__name__)


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Llama3 8B model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML OmegaConf override file. Default: conf/llama3_8b_pretrain_override_example.yaml",
    )

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Mamba 8B pretraining script.
    """
    args, cli_overrides = parse_cli_args()

    train_samples = 122070313
    warmup_samples = 1024000
    # train_samples = 32*20
    # warmup_samples = 32*10
    global_batch_size = 1536
    train_iters = train_samples // global_batch_size
    warmup_iters = warmup_samples // global_batch_size

    # OUTPUT FOLDER/SBATCH FILE
    # LOG INTERVAL
    # TMUX NOT SHOWING OUTPUT???

    cfg: ConfigContainer = pretrain_config(
        name="nm6_test_data_parallelism", # should be the same as the name of the experiment
        train_iters=train_iters,
        global_batch_size=global_batch_size,
        micro_batch_size=2,
        lr_warmup_iters=warmup_iters,
        tensor_parallelism=4,
        pipeline_parallelism=2,
        log_interval=10,
        per_split_data_args_path="/lustre/fsw/portfolios/llmservice/users/jupinderp/data_blends/1T-phase1var-moresft-full.json",
        path_to_cache="/lustre/fs1/portfolios/coreai/users/liding/nemo/workspace/experiments/nm6_training/data_cache_8k",
    )
    # logger.info("Loaded base configuration")
    # if get_rank_safe() == 0:
    #     cfg.to_yaml()

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
    # logger.info("--- Final Merged Configuration ---")
    # if get_rank_safe() == 0:
    #     cfg.to_yaml()
    # logger.info("----------------------------------")
    
    # Start training
    logger.debug("Starting pretraining...")
    pretrain(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
