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

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors import process_squad_example
from megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2 import (
    nemotron_next_3b_v2_pretrain_config as pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig
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
        description="Finetune Llama3 8B model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML OmegaConf override file. Default: conf/llama3_8b_pretrain_override_example.yaml", # TODO: update yaml file name
    )

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Mamba 8B finetuning script.
    """ 
    args, cli_overrides = parse_cli_args()

    cfg: ConfigContainer = pretrain_config(micro_batch_size=1,enable_deepep=False)


    # TODO: hardcode dataset config w/ packed sequences here for now
    seq_length = 8192 # NOTE: need to change model seq length as well
    packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_cu_seqlens=False)
    #

    dataset_config = HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=seq_length,
        seed=1234,
        dataloader_type="single",
        num_workers=1,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=False,
        delete_raw=False,
        dataset_kwargs = {"pad_to_max_length": True},
        do_validation=False,
        do_test=False,
    )
    cfg.dataset = dataset_config
    cfg.model.seq_length = seq_length

    # tokenizer_model = "/lustre/fs1/portfolios/coreai/users/bobchen/ckpt/nm6-hybrid-3b-new-666000"
    tokenizer_model = "nvidia/Nemotron-H-8B-Reasoning-128K" 
    # "/lustre/fsw/portfolios/llmservice/users/soumyes/nano-v3/nano-v3-sft-tokenizer"
    cfg.tokenizer=TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=tokenizer_model)

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

    # Start training
    logger.debug("Starting finetuning...")
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
