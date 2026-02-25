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
FLUX Fine-tuning Script with YAML and CLI Configuration Overrides.

This script provides a flexible way to fine-tune FLUX models using Megatron-Bridge with support for
both YAML configuration files and command-line overrides using Hydra-style syntax.

The script loads a pretrained checkpoint and continues training with your custom dataset.
Fine-tuning typically uses lower learning rates and fewer training iterations compared to pretraining.

Forward Step Options:
    - Automodel FlowMatchingPipeline (default): Unified flow matching implementation
    - Original FluxForwardStep (--use-original-step): Classic implementation

Examples:
    Basic usage with checkpoint loading (uses automodel pipeline):
        $ torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/pretrained/checkpoint --mock

    Using original FluxForwardStep:
        $ torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/pretrained/checkpoint --mock --use-original-step

    Using a custom YAML config file:
        $ torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/pretrained/checkpoint \
        --config-file my_custom_config.yaml

    Using CLI overrides only:
        $ torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/pretrained/checkpoint \
        model.tensor_model_parallel_size=4 train.train_iters=5000 optimizer.lr=1e-5

    Combining YAML and CLI overrides (CLI takes precedence):
        $ torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/pretrained/checkpoint \
        --config-file conf/my_config.yaml \
        model.pipeline_dtype=torch.float16 \
        train.global_batch_size=512

    Using automodel pipeline with custom parameters (automodel is default):
        $ torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/pretrained/checkpoint --mock \
        --flow-shift=1.0 --use-loss-weighting

Configuration Precedence:
    1. Base configuration from pretrain_config() recipe
    2. YAML overrides from --config-file (if provided)
    3. CLI overrides (highest precedence)
    4. Checkpoint loading path (from --load-checkpoint)

Supported Override Syntax:
    - Standard assignment: key=value
    - Nested assignment: section.subsection.key=value
    - Addition: +new_key=value
    - Deletion: ~key_to_remove
    - Type conversion: Automatic for basic types (int, float, bool, str)
    - Complex types: torch.dtype, enums, etc. are supported
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from omegaconf import OmegaConf

from megatron.bridge.diffusion.models.flux.flux_step_with_automodel import create_flux_forward_step
from megatron.bridge.diffusion.recipes.flux.flux import pretrain_config


logger: logging.Logger = logging.getLogger(__name__)


# Define paths relative to this script's location
SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "flux_pretrain_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Fine-tune FLUX model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained checkpoint directory to load for fine-tuning.",
    )
    parser.add_argument("--mock", action="store_true", help="Whether to use mock data.")
    parser.add_argument(
        "--timestep-sampling",
        choices=["logit_normal", "uniform", "mode"],
        default="logit_normal",
        help="Timestep sampling strategy for flow matching.",
    )
    parser.add_argument(
        "--logit-mean",
        type=float,
        default=0.0,
        help="Mean for logit-normal timestep sampling.",
    )
    parser.add_argument(
        "--logit-std",
        type=float,
        default=1.0,
        help="Std for logit-normal timestep sampling.",
    )
    parser.add_argument(
        "--mode-scale",
        type=float,
        default=1.29,
        help="Scale for mode timestep sampling.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale for FLUX-dev models.",
    )
    parser.add_argument(
        "--scheduler-steps",
        type=int,
        default=1000,
        help="Number of scheduler training steps.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/flux_pretrain_override_example.yaml",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Forward step implementation choice
    parser.add_argument(
        "--use-original-step",
        action="store_true",
        help="Use original FluxForwardStep instead of automodel FlowMatchingPipeline (default)",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=1.0,
        help="Flow shift parameter (for automodel pipeline)",
    )
    parser.add_argument(
        "--use-loss-weighting",
        action="store_true",
        help="Use loss weighting (for automodel pipeline)",
    )

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the FLUX fine-tuning script.

    This function orchestrates the complete configuration workflow:
    1. Loads the base configuration from pretrain_config() recipe
    2. Applies YAML overrides from --config-file (if exists)
    3. Applies CLI overrides using Hydra-style syntax
    4. Sets checkpoint loading path for fine-tuning
    5. Starts Megatron training with the final merged configuration

    Configuration merging preserves callable fields (like activation functions)
    and handles type conversions automatically.

    Examples of CLI usage:
        # Fine-tune with default config and custom learning rate (automodel pipeline is default)
        torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/checkpoint --mock optimizer.lr=1e-5

        # Use original FluxForwardStep instead of automodel pipeline
        torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/checkpoint --mock --use-original-step

        # Custom config file with additional overrides
        torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/checkpoint \
        --config-file my_config.yaml train.train_iters=5000

        # Multiple overrides for distributed fine-tuning (uses automodel by default)
        torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/checkpoint --mock \
        model.tensor_model_parallel_size=4 \
        model.pipeline_model_parallel_size=2 \
        train.global_batch_size=512 \
        optimizer.lr=5e-6

        # Automodel pipeline with custom flow matching parameters
        torchrun --nproc_per_node=8 finetune_flux.py \
        --load-checkpoint /path/to/checkpoint --mock \
        --flow-shift=1.0 --use-loss-weighting
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge FLUX Fine-tuning Script with YAML & CLI Overrides")
    logger.info("------------------------------------------------------------------")

    # Validate and normalize checkpoint path
    checkpoint_path = args.load_checkpoint
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    # Check if the path points to a specific iteration directory (iter_XXXXXXX)
    # If so, extract the base directory and iteration number
    import re

    checkpoint_dir_name = os.path.basename(checkpoint_path.rstrip("/"))
    iter_match = re.match(r"iter_(\d+)", checkpoint_dir_name)

    if iter_match:
        # User provided a specific iteration directory
        iteration_num = int(iter_match.group(1))
        base_dir = os.path.dirname(checkpoint_path.rstrip("/"))
        if not base_dir:
            base_dir = "."
        logger.info(f"Detected iteration directory: {checkpoint_dir_name}")
        logger.info(f"Extracted base directory: {base_dir}, iteration: {iteration_num}")
        checkpoint_path = base_dir
        # Set ckpt_step to load from this specific iteration
        ckpt_step_override = iteration_num
    else:
        # User provided base directory - will load latest checkpoint
        ckpt_step_override = None
        logger.info(f"Using checkpoint base directory: {checkpoint_path}")

    # Load base configuration from the recipe as a Python dataclass
    cfg: ConfigContainer = pretrain_config(mock=args.mock)
    logger.info("Loaded base configuration")

    # Set checkpoint configuration for fine-tuning
    # If ckpt_step is specified, we need to set load (validation requirement)
    # Otherwise, use pretrained_checkpoint (preferred for fine-tuning)
    if ckpt_step_override is not None:
        # When loading a specific iteration, set load (required by validation when ckpt_step is set)
        cfg.checkpoint.load = checkpoint_path
        cfg.checkpoint.ckpt_step = ckpt_step_override
        cfg.checkpoint.pretrained_checkpoint = None
        logger.info(f"Will load from specific iteration: {ckpt_step_override}")
        logger.info(f"Checkpoint load path set to: {checkpoint_path}")
    else:
        # When loading latest checkpoint, use pretrained_checkpoint (preferred for fine-tuning)
        cfg.checkpoint.pretrained_checkpoint = checkpoint_path
        cfg.checkpoint.load = None  # Clear load to ensure pretrained_checkpoint takes precedence
        logger.info(f"Pretrained checkpoint path set to: {checkpoint_path}")
    # Explicitly set finetune flag to True - this ensures:
    # - Model weights are loaded
    # - Iteration is reset to 0
    # - Optimizer state is NOT loaded (fresh optimizer)
    # - RNG state is NOT loaded (fresh RNG)
    cfg.checkpoint.finetune = True
    logger.info("Fine-tuning mode enabled (checkpoint.finetune=True)")

    # Print configuration on rank 0
    if get_rank_safe() == 0:
        cfg.print_yaml()

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

    # Ensure checkpoint configuration is set correctly (override any YAML/CLI that might have changed it)
    if ckpt_step_override is not None:
        # When loading a specific iteration, set load (required by validation when ckpt_step is set)
        merged_omega_conf.checkpoint.load = checkpoint_path
        merged_omega_conf.checkpoint.ckpt_step = ckpt_step_override
        merged_omega_conf.checkpoint.pretrained_checkpoint = None
    else:
        # When loading latest checkpoint, use pretrained_checkpoint (preferred for fine-tuning)
        merged_omega_conf.checkpoint.pretrained_checkpoint = checkpoint_path
        merged_omega_conf.checkpoint.load = None  # Clear load to ensure pretrained_checkpoint takes precedence
    merged_omega_conf.checkpoint.finetune = True

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Ensure checkpoint configuration is set correctly in the final config
    if ckpt_step_override is not None:
        # When loading a specific iteration, set load (required by validation when ckpt_step is set)
        cfg.checkpoint.load = checkpoint_path
        cfg.checkpoint.ckpt_step = ckpt_step_override
        cfg.checkpoint.pretrained_checkpoint = None
    else:
        # When loading latest checkpoint, use pretrained_checkpoint (preferred for fine-tuning)
        cfg.checkpoint.pretrained_checkpoint = checkpoint_path
        cfg.checkpoint.load = None  # Clear load to ensure pretrained_checkpoint takes precedence
    cfg.checkpoint.finetune = True

    # Create forward step (configurable: original or automodel pipeline)
    # Default is automodel pipeline unless --use-original-step is specified
    if not args.use_original_step:
        # Use automodel FlowMatchingPipeline
        flux_forward_step = create_flux_forward_step(
            use_automodel_pipeline=True,
            timestep_sampling=args.timestep_sampling,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            flow_shift=args.flow_shift,
            scheduler_steps=args.scheduler_steps,
            guidance_scale=args.guidance_scale,
            use_loss_weighting=args.use_loss_weighting,
        )
        if get_rank_safe() == 0:
            logger.info("=" * 70)
            logger.info("✅ Using AUTOMODEL FlowMatchingPipeline")
            logger.info(f"   Timestep Sampling: {args.timestep_sampling}")
            logger.info(f"   Flow Shift: {args.flow_shift}")
            logger.info(f"   Loss Weighting: {args.use_loss_weighting}")
            logger.info("=" * 70)
    else:
        # Use original FluxForwardStep
        flux_forward_step = create_flux_forward_step(
            use_automodel_pipeline=False,
            timestep_sampling=args.timestep_sampling,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
            scheduler_steps=args.scheduler_steps,
            guidance_scale=args.guidance_scale,
        )
        if get_rank_safe() == 0:
            logger.info("=" * 70)
            logger.info("✅ Using ORIGINAL FluxForwardStep")
            logger.info(f"   Timestep Sampling: {args.timestep_sampling}")
            logger.info("=" * 70)

    # Display final configuration
    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")
        logger.info("Fine-tuning Configuration:")
        if cfg.checkpoint.pretrained_checkpoint is not None:
            logger.info(f"  Pretrained Checkpoint Path: {cfg.checkpoint.pretrained_checkpoint}")
        if cfg.checkpoint.load is not None:
            logger.info(f"  Checkpoint Load Path: {cfg.checkpoint.load}")
        if cfg.checkpoint.ckpt_step is not None:
            logger.info(f"  Checkpoint Step: {cfg.checkpoint.ckpt_step}")
        logger.info("FluxForwardStep config:")
        logger.info(f"  timestep_sampling: {args.timestep_sampling}")
        logger.info(f"  logit_mean: {args.logit_mean}")
        logger.info(f"  logit_std: {args.logit_std}")
        logger.info(f"  mode_scale: {args.mode_scale}")
        logger.info(f"  scheduler_steps: {args.scheduler_steps}")
        logger.info(f"  guidance_scale: {args.guidance_scale}")
        if not args.use_original_step:
            logger.info(f"  flow_shift: {args.flow_shift}")
            logger.info(f"  use_loss_weighting: {args.use_loss_weighting}")

    # Start training (fine-tuning)
    logger.debug("Starting fine-tuning...")
    pretrain(config=cfg, forward_step_func=flux_forward_step)


if __name__ == "__main__":
    main()
