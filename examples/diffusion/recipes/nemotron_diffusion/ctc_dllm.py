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
CTC-based dLLM pretraining.

Uses CTCStep instead of DGPTStep. The model predicts next blocks using CTC loss
with 2x output expansion (block_size=128, target=64).

Examples:
    3B model:
        $ torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/ctc_dllm.py \
            --model-size 3b \
            --hf-path mistralai/Ministral-3-3B-Base-2512 \
            --data-paths /path/to/data \
            checkpoint.pretrained_checkpoint=/path/to/checkpoint \
            checkpoint.finetune=true
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import torch
from omegaconf import OmegaConf

import megatron.bridge.diffusion.conversion.nemotron_diffusion.nemotron_diffusion_bridge  # noqa: F401
from megatron.bridge.diffusion.models.common.ctc_step import CTCStep
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


logger = logging.getLogger(__name__)

PRETRAIN_CONFIGS = {
    "3b": nemotron_diffusion3_3b_pretrain_config,
    "8b": nemotron_diffusion3_8b_pretrain_config,
    "14b": nemotron_diffusion3_14b_pretrain_config,
}


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="CTC-based dLLM pretraining",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model-size", type=str, choices=list(PRETRAIN_CONFIGS.keys()), default="3b")
    parser.add_argument("--hf-path", type=str, default=None)
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-paths", type=str, nargs="*", default=None)
    parser.add_argument("--data-args-path", type=str, default=None)

    args, cli_overrides = parser.parse_known_args()

    if args.data_paths:
        flattened = []
        for p in args.data_paths:
            if "," in p:
                flattened.extend(p.split(","))
            else:
                flattened.append(p)
        args.data_paths = [p.strip() for p in flattened if p.strip()]

    return args, cli_overrides


def main() -> None:
    args, cli_overrides = parse_cli_args()

    pretrain_config = PRETRAIN_CONFIGS[args.model_size]
    cfg: ConfigContainer = pretrain_config(
        data_paths=args.data_paths,
        data_args_path=args.data_args_path,
        hf_path=args.hf_path,
    )

    # Override model config for CTC paradigm
    cfg.model.dlm_paradigm = "ctc"
    cfg.model.block_size = 128
    cfg.model.ctc_target_block_size = 64

    if get_rank_safe() == 0:
        cfg.print_yaml()

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file is not None:
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides)

    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides, excluded_fields)

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration (CTC dLLM) ---")
        cfg.print_yaml()
        logger.info("----------------------------------------------")

    pretrain(config=cfg, forward_step_func=CTCStep())

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
