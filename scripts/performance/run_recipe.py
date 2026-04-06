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
Training script for Megatron-Bridge recipes.
This script runs inside the container and handles the actual training execution.
"""

import logging
import os
import sys

import torch
from argument_parser import parse_cli_args
from utils.datasets import (
    create_mock_dataset_config,
    create_rp2_dataset_config,
    create_squad_dataset_config,
)
from utils.utils import get_library_recipe

from megatron.bridge.utils.common_utils import get_rank_safe

# Diffusion model families manage their own dataset configs and require
# a dedicated forward step function rather than the standard GPT step.
DIFFUSION_FAMILIES = frozenset({"flux", "wan"})


def _get_diffusion_step(model_family_name: str):
    """Return the appropriate forward step instance for a diffusion model family."""
    if model_family_name == "flux":
        from megatron.bridge.diffusion.models.flux.flux_step import FluxForwardStep

        return FluxForwardStep()
    elif model_family_name == "wan":
        from megatron.bridge.diffusion.models.wan.wan_step import WanForwardStep

        return WanForwardStep()
    raise ValueError(f"Unknown diffusion model family: {model_family_name!r}")


def set_user_overrides(config, args):
    """Apply CLI arguments to ConfigContainer fields."""
    is_diffusion = args.model_family_name in DIFFUSION_FAMILIES

    # Training configuration
    if args.max_steps:
        config.train.train_iters = args.max_steps
    if args.global_batch_size:
        config.train.global_batch_size = args.global_batch_size
    if args.micro_batch_size:
        config.train.micro_batch_size = args.micro_batch_size

    # Optimizer configuration
    if args.lr:
        config.optimizer.lr = args.lr
    if args.min_lr:
        config.optimizer.min_lr = args.min_lr

    # Scheduler configuration
    if args.warmup_iters:
        config.scheduler.lr_warmup_iters = args.warmup_iters

    # Checkpoint configuration
    if args.pretrained_checkpoint:
        config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    if args.save_dir:
        config.checkpoint.save = args.save_dir
    if args.load_dir:
        config.checkpoint.load = args.load_dir
    if args.save_interval:
        config.checkpoint.save_interval = args.save_interval
    if args.most_recent_k:
        config.checkpoint.most_recent_k = args.most_recent_k

    # Dataset configuration
    # Diffusion models (FLUX, WAN) configure their own dataset inside the recipe
    # (data_paths=None → mock/synthetic data by default). Replacing config.dataset
    # with a GPT-style mock config would break them, so skip this block entirely.
    if not is_diffusion:
        logging.info(f"Configuring dataset: type={args.data}")

        cp_size = getattr(config.model, "context_parallel_size", 1) or 1
        pad_seq_to_mult = cp_size * 2 if cp_size > 1 else 1

        # Create dataset configuration based on type
        if args.data == "mock":
            config.dataset = create_mock_dataset_config(seq_length=args.seq_length or 8192)
        elif args.data == "rp2":
            if not args.dataset_paths or not args.index_mapping_dir:
                raise ValueError("--dataset-paths and --index-mapping-dir are required for rp2 dataset")
            config.dataset = create_rp2_dataset_config(
                dataset_paths=args.dataset_paths,
                seq_length=args.seq_length or 8192,
                index_mapping_dir=args.index_mapping_dir,
            )
        elif args.data == "squad":
            if not args.dataset_root:
                raise ValueError("--dataset-root is required for squad dataset")
            config.dataset = create_squad_dataset_config(
                dataset_root=args.dataset_root,
                seq_length=args.seq_length or 8192,
                packed=False,
                pad_seq_to_mult=pad_seq_to_mult,
            )
        elif args.data == "squad_packed":
            if not args.dataset_root:
                raise ValueError("--dataset-root is required for squad_packed dataset")
            config.dataset = create_squad_dataset_config(
                dataset_root=args.dataset_root,
                seq_length=args.seq_length or 8192,
                packed=True,
                pad_seq_to_mult=pad_seq_to_mult,
            )
        else:
            raise ValueError(f"Unknown dataset type: {args.data}")

        # Tokenizer configuration
        from megatron.bridge.training.config import TokenizerConfig

        if args.tokenizer_type == "NullTokenizer":
            config.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=args.vocab_size)
        elif args.tokenizer_type == "HuggingFaceTokenizer":
            if not args.tokenizer_model:
                raise ValueError("--tokenizer-model is required when using HuggingFaceTokenizer")
            tokenizer_model = args.tokenizer_model
            config.tokenizer = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=tokenizer_model)
        elif args.tokenizer_type == "SentencePieceTokenizer":
            if not args.tokenizer_model:
                raise ValueError("--tokenizer-model is required for SentencePieceTokenizer")
            config.tokenizer = TokenizerConfig(
                tokenizer_type="SentencePieceTokenizer", tokenizer_model=args.tokenizer_model
            )

    # Model configuration
    # Diffusion models use fixed image/latent dimensions; seq_length is not applicable.
    if args.seq_length and not is_diffusion:
        config.model.seq_length = args.seq_length
    if args.tensor_model_parallel_size:
        config.model.tensor_model_parallel_size = args.tensor_model_parallel_size
    if args.pipeline_model_parallel_size:
        config.model.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    if args.context_parallel_size:
        config.model.context_parallel_size = args.context_parallel_size
    if args.virtual_pipeline_model_parallel_size != -1:
        config.model.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    if args.expert_model_parallel_size:
        config.model.expert_model_parallel_size = args.expert_model_parallel_size
    if args.expert_tensor_parallel_size:
        config.model.expert_tensor_model_parallel_size = args.expert_tensor_parallel_size

    # Logging configuration
    config.logger.log_timers_to_tensorboard = True
    if args.save_config_filepath:
        config.logger.save_config_filepath = args.save_config_filepath

    # WandB configuration
    if args.wandb_project_name:
        config.logger.wandb_project = args.wandb_project_name
    if args.wandb_entity_name:
        config.logger.wandb_entity = args.wandb_entity_name
    if args.wandb_experiment_name:
        config.logger.wandb_exp_name = args.wandb_experiment_name
    if args.wandb_save_dir:
        config.logger.wandb_save_dir = args.wandb_save_dir

    # Handle convergence mode configuration
    config.logger.log_interval = 1

    # Checkpoint configuration for convergence
    if args.max_steps <= 100:
        # Short convergence runs - save at the end
        config.checkpoint.save_interval = args.save_interval or args.max_steps
    else:
        # Long convergence runs - save every save_interval steps
        config.checkpoint.save_interval = args.save_interval or 1000

    # Validation configuration for convergence
    if args.max_steps <= 100:
        config.train.eval_interval = args.max_steps
        config.train.eval_iters = 0  # Disable evaluation for short convergence runs
    else:
        config.train.eval_interval = 800

    if args.max_steps > 100:
        config.scheduler.lr_warmup_iters = int(0.01 * args.max_steps)

    return config


def main():
    """Main entry point for the training script."""

    # Parse known args and capture unknown ones for config overrides
    parser = parse_cli_args()
    args, _ = parser.parse_known_args()

    recipe = get_library_recipe(
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        train_task=args.task,
        wandb_experiment_name=args.wandb_experiment_name,
    )

    recipe = set_user_overrides(recipe, args)

    if args.dryrun:
        save_path = args.save_config_filepath or "ConfigContainer.yaml"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        recipe.to_yaml(save_path)
        logging.info(f"ConfigContainer saved to: {os.path.abspath(save_path)}")
        recipe.print_yaml()
        sys.exit(0)

    # Log final configuration
    if get_rank_safe() == 0:
        logging.info("Final configuration:")
        recipe.print_yaml()

    if args.task == "pretrain":
        logging.info("Starting pretraining")
        from megatron.bridge.training.pretrain import pretrain

        if args.model_family_name in DIFFUSION_FAMILIES:
            forward_step = _get_diffusion_step(args.model_family_name)
        else:
            from megatron.bridge.training.gpt_step import forward_step

        pretrain(config=recipe, forward_step_func=forward_step)
    elif args.task in ["sft", "lora"]:
        logging.info("Starting finetuning")
        from megatron.bridge.training.finetune import finetune
        from megatron.bridge.training.gpt_step import forward_step

        finetune(config=recipe, forward_step_func=forward_step)
    else:
        raise ValueError("Must specify either --pretrain or --finetune")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()