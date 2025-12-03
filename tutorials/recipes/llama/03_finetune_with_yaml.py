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
Finetune with YAML Configuration and CLI Overrides

This script demonstrates how to use YAML configuration files and command-line
overrides for finetuning with LoRA or full supervised finetuning (SFT).

Usage:
    With default config file:
        torchrun --nproc_per_node=1 03_finetune_with_yaml.py

    With custom config file:
        torchrun --nproc_per_node=2 03_finetune_with_yaml.py \
            --config-file conf/my_finetune_config.yaml

    With command-line overrides:
        torchrun --nproc_per_node=2 03_finetune_with_yaml.py \
            train.train_iters=1000 \
            optimizer.lr=5e-5

    Full finetuning instead of LoRA:
        torchrun --nproc_per_node=2 03_finetune_with_yaml.py \
            --peft none \
            train.train_iters=1000

    Combining YAML and CLI (CLI takes precedence):
        torchrun --nproc_per_node=2 03_finetune_with_yaml.py \
            --config-file conf/llama32_1b_finetune.yaml \
            peft.dim=16 \
            train.train_iters=2000

Configuration Priority (highest to lowest):
    1. Command-line overrides (highest)
    2. YAML config file
    3. Base recipe defaults (lowest)

See conf/ directory for example YAML configurations.
For a pure Python usage see 01_quickstart_finetune.py.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

from megatron.bridge.recipes.llama import llama32_1b_finetune_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


logger = logging.getLogger(__name__)

# Default config file location
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "conf" / "llama32_1b_finetune.yaml"


def parse_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune with YAML configuration and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=f"Path to YAML config file (optional). Default: {DEFAULT_CONFIG_FILE}",
    )
    parser.add_argument(
        "--peft",
        type=str,
        default="lora",
        choices=["lora", "dora", "none"],
        help="PEFT method to use. Use 'none' for full finetuning.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Separate known args from CLI overrides
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def main() -> None:
    """Run finetuning with YAML configuration and CLI overrides."""
    args, cli_overrides = parse_args()

    # Load base configuration from recipe
    peft_method = None if args.peft == "none" else args.peft
    config: ConfigContainer = llama32_1b_finetune_config(peft=peft_method)

    config = process_config_with_overrides(
        config,
        config_filepath=args.config_file,
        cli_overrides=cli_overrides or None,
    )

    # Start finetuning
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
