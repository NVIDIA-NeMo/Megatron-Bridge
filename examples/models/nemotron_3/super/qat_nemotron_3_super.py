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

from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_finetune_config as finetune_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger: logging.Logger = logging.getLogger(__name__)


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Quantization-Aware Training (QAT) of a Nemotron 3 Super model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--megatron-load-path",
        type=str,
        default=None,
        help="Path to the quantized Megatron checkpoint to load (must be created first using quantize.py)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML OmegaConf override file.",
    )
    parser.add_argument("--packed-sequence", action="store_true", help="Whether to use sequence packing")
    parser.add_argument("--seq-length", type=int, default=8192, help="Sequence length")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Nemotron 3 Super QAT script.
    """
    args, cli_overrides = parse_cli_args()

    # Check if the checkpoint path exists
    if not os.path.exists(args.megatron_load_path):
        print(f"[red]Error: Quantized checkpoint path {args.megatron_load_path} does not exist![/red]")
        print("[yellow]Please run the quantization process first:[/yellow]")
        print(f"[yellow]torchrun examples/models/quantize.py --megatron-save-path {args.megatron_load_path}[/yellow]")
        sys.exit(1)

    cfg: ConfigContainer = finetune_config(
        pretrained_checkpoint=args.megatron_load_path,
        seq_length=args.seq_length,
        peft=None,
        packed_sequence=args.packed_sequence,
    )
    cfg.model.seq_length = args.seq_length

    # Check for ModelOpt state
    checkpoint_to_check = cfg.checkpoint.pretrained_checkpoint or cfg.checkpoint.load
    if checkpoint_to_check and has_modelopt_state(checkpoint_to_check):
        cfg.model.restore_modelopt_state = True
        # ModelOpt quantization does not support gradient accumulation fusion
        cfg.model.gradient_accumulation_fusion = False
    else:
        raise logger.error(
            "No ModelOpt state found in checkpoint! Cannot start QAT. \
        Please create a quantized checkpoint using quantize.py first."
        )
        sys.exit(1)

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

    # Start QAT
    logger.debug("Starting QAT...")
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
