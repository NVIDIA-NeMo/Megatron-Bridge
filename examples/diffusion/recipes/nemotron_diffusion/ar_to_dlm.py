#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
NemotronDiffusion diffusion LM pretraining.

Uses the sbd_block_diff diffusion paradigm via DGPTStep.
Use --model-size to select 3b, 8b, or 14b (default: 3b).
Use --hf-path to override the HuggingFace model ID or local model path.

Examples:
    3B model, first job from AR checkpoint (finetune=true skips optimizer state):
        $ torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/ar_to_dlm.py \
            --model-size 3b \
            --hf-path mistralai/Ministral-3-3B-Base-2512 \
            --data-paths /path/to/dclm/merged_tokenized_text_document \
            checkpoint.pretrained_checkpoint=/path/to/hf_to_mb_3b \
            checkpoint.finetune=true

    3B model, subsequent jobs (resume from DLM checkpoint):
        $ torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/ar_to_dlm.py \
            --model-size 3b \
            --hf-path mistralai/Ministral-3-3B-Base-2512 \
            --data-paths /path/to/dclm/merged_tokenized_text_document

    8B model with TP=4:
        $ torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/ar_to_dlm.py \
            --model-size 8b \
            --hf-path mistralai/Ministral-3-8B-Base-2512 \
            --data-paths /path/to/dclm/merged_tokenized_text_document \
            checkpoint.pretrained_checkpoint=/path/to/hf_to_mb_8b \
            checkpoint.finetune=true

    14B model with TP=8:
        $ torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/ar_to_dlm.py \
            --model-size 14b \
            --hf-path mistralai/Ministral-3-14B-Base-2512 \
            --data-paths /path/to/dclm/merged_tokenized_text_document \
            checkpoint.pretrained_checkpoint=/path/to/hf_to_mb_14b \
            checkpoint.finetune=true
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf

# Register NemotronDiffusionBridge, overriding the base Ministral3Bridge so that
# AutoBridge returns NemotronDiffusionModelProvider (with NemotronDiffusionAttention).
import megatron.bridge.diffusion.conversion.nemotron_diffusion.nemotron_diffusion_bridge  # noqa: F401
from megatron.bridge.diffusion.models.common.dgpt_step import DGPTStep
from megatron.bridge.diffusion.recipes.nemotron_diffusion.ar_to_dlm import (
    nemotron_diffusion3_3b_pretrain_config,
    nemotron_diffusion3_8b_pretrain_config,
    nemotron_diffusion3_14b_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)

PRETRAIN_CONFIGS = {
    "3b": nemotron_diffusion3_3b_pretrain_config,
    "8b": nemotron_diffusion3_8b_pretrain_config,
    "14b": nemotron_diffusion3_14b_pretrain_config,
}


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments for the AR-to-DLM conversion script."""
    parser = argparse.ArgumentParser(
        description="NemotronDiffusion diffusion LM pretraining (no distillation)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(PRETRAIN_CONFIGS.keys()),
        default="3b",
        help="Model size to train (default: 3b).",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path to model weights. Overrides the default for the selected model size.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML override file.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="*",
        default=None,
        help="List of dataset file paths (space or comma-separated).",
    )
    parser.add_argument(
        "--data-args-path",
        type=str,
        default=None,
        help="Path to file containing data arguments.",
    )

    args, cli_dotlist_overrides = parser.parse_known_args()

    if args.data_paths:
        flattened_paths = []
        for path in args.data_paths:
            if "," in path:
                flattened_paths.extend(path.split(","))
            else:
                flattened_paths.append(path)
        args.data_paths = [p.strip() for p in flattened_paths if p.strip()]

    return args, cli_dotlist_overrides


def main() -> None:
    """Entry point for AR-to-DLM conversion and continued pretraining."""
    args, cli_overrides = parse_cli_args()

    pretrain_config = PRETRAIN_CONFIGS[args.model_size]
    cfg: ConfigContainer = pretrain_config(
        data_paths=args.data_paths,
        data_args_path=args.data_args_path,
        hf_path=args.hf_path,
    )

    if get_rank_safe() == 0:
        cfg.print_yaml()

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file is not None:
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    pretrain(config=cfg, forward_step_func=DGPTStep())

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
