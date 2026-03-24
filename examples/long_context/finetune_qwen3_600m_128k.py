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

"""Long-context (128K) SFT example for Qwen3-600M.

Parallelism: TP=1, CP=8 (8 GPUs minimum)
Sequence length: 131072 (128K tokens)

Usage
-----
Minimal (mock data):
    torchrun --nproc_per_node=8 examples/long_contex/finetune_qwen3_600m_128k.py

With a blend dataset (JSON):
    torchrun --nproc_per_node=8 examples/long_contex/finetune_qwen3_600m_128k.py \\
        --per-split-data-args-path /path/to/data.json

With a pretrained checkpoint:
    torchrun --nproc_per_node=8 examples/long_contex/finetune_qwen3_600m_128k.py \\
        --per-split-data-args-path /path/to/data.json \\
        checkpoint.pretrained_checkpoint=/path/to/pretrained

Additional Hydra-style dot-notation overrides:
    torchrun --nproc_per_node=8 examples/long_contex/finetune_qwen3_600m_128k.py \\
        --per-split-data-args-path /path/to/data.json \\
        train.train_samples=10000 \\
        logger.wandb_project=my_project
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen.qwen3 import qwen3_600m_sft_128k_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger: logging.Logger = logging.getLogger(__name__)


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Long-context (128K) SFT for Qwen3-600M using Megatron-Bridge",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to a YAML OmegaConf override file.",
    )

    # Remaining args are treated as Hydra-style dot-notation overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """Entry point for Qwen3-600M 128K long-context SFT."""
    args, cli_overrides = parse_cli_args()

    cfg: ConfigContainer = qwen3_600m_sft_128k_config()

    # Convert to OmegaConf DictConfig for merging
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

    # Apply command-line Hydra-style overrides
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged config back to the ConfigContainer
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Start training
    logger.debug("Starting long-context SFT...")
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
