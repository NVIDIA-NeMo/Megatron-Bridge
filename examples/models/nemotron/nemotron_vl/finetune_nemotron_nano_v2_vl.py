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
Finetune Nemotron Nano V2 VL with YAML and CLI Configuration Overrides.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.llava_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "nemotron_nano_v2_vl_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line flags and return `(argparse.Namespace, overrides)`."""

    parser = argparse.ArgumentParser(
        description="Finetune Nemotron Nano V2 VL with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Same flags as pre-training script
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/nemotron_nano_v2_vl_override_example.yaml",
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        default="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        help="Path to the HuggingFace model to load weights from. Default: nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    )
    # Finetune-specific flags
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to a Megatron-Bridge checkpoint directory or HuggingFace model name to load weights from.",
    )
    parser.add_argument(
        "--lora-on-language-model",
        action="store_true",
        help="Use PEFT for finetuning on the language model.",
    )
    parser.add_argument(
        "--lora-on-vision-model",
        action="store_true",
        help="Use PEFT for finetuning on the vision model.",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """Main function to finetune Nemotron Nano V2 VL."""
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Nemotron Nano V2 VL Finetuning Script with YAML & CLI Overrides")
    logger.info("--------------------------------------------------------------------------")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from megatron.bridge.recipes.nemotron_vl.nemotron_nano_v2_vl import nemotron_nano_v2_vl_12b_finetune_config

    cfg: ConfigContainer = nemotron_nano_v2_vl_12b_finetune_config(
        hf_model_path=args.hf_model_path,
        pretrained_checkpoint=args.pretrained_checkpoint,
        lora_on_language_model=args.lora_on_language_model,
        lora_on_vision_model=args.lora_on_vision_model,
    )

    logger.info("Loaded base configuration for finetuning")

    if get_rank_safe() == 0:
        cfg.print_yaml()

    # Convert to OmegaConf so we can merge overrides
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
        logger.info("--- Final Merged Configuration (Finetune) ---")
        cfg.print_yaml()
        logger.info("--------------------------------------------")

    # Nemotron Nano V2 VL uses the LLaVaModel class for the model, so we use the llava_step.forward_step function.
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
