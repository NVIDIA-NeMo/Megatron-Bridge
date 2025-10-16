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

import argparse
import importlib
import logging

import torch

from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


def parse_plugin_config_overrides(unknown_args: list[str]) -> list[str]:
    """Parse unknown arguments as config overrides from plugins.

    Args:
        unknown_args: List of unknown command line arguments

    Returns:
        List of config override strings in format "section.field=value"
    """
    config_overrides = []
    for arg in unknown_args:
        if "=" in arg:
            # Handle dotted config format: section.field=value
            config_overrides.append(arg)
        else:
            logging.warning(f"Unknown argument ignored (expected format section.field=value): {arg}")

    if config_overrides:
        logging.info(f"Found {len(config_overrides)} config overrides from plugins: {config_overrides}")

    return config_overrides


def create_mock_dataset_config(seq_length):
    """Create mock dataset configuration for Megatron-Bridge."""
    from megatron.bridge.training.config import MockGPTDatasetConfig

    # Create mock dataset using MockGPTDatasetConfig which enforces blend=None, blend_per_split=None
    return MockGPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        sequence_length=seq_length,
        num_dataset_builder_threads=1,
        split="99990,8,2",  # Standard train/val/test split
        # Dataloader config parameters
        data_sharding=True,
        dataloader_type="single",
        num_workers=1,
    )


def create_rp2_dataset_config(dataset_paths, seq_length, index_mapping_dir=None):
    """Create RedPajama2 dataset configuration for Megatron-Bridge."""
    from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
    from megatron.bridge.training.config import GPTDatasetConfig

    # Get blend configuration for rp2 data paths
    blend, blend_per_split, split = get_blend_fields_from_data_paths(data_paths=dataset_paths, mock=False)

    return GPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        sequence_length=seq_length,
        num_dataset_builder_threads=1,
        blend=blend,
        blend_per_split=blend_per_split,
        split=split or "99990,8,2",
        path_to_cache=index_mapping_dir,
        # Dataloader config parameters
        data_sharding=True,
        dataloader_type="single",
        num_workers=1,
        persistent_workers=True,
    )


def create_squad_dataset_config(dataset_root, seq_length, packed=False):
    """Create SQuAD dataset configuration for Megatron-Bridge using HF dataset."""
    from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
    from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
    from megatron.bridge.data.hf_processors import process_squad_example

    # Create packed sequence specs if needed
    packed_sequence_specs = None
    if packed:
        packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length)

    return HFDatasetConfig(
        dataset_name="squad",  # Hugging Face dataset name
        process_example_fn=process_squad_example,  # Processing function
        dataset_root=dataset_root,  # Local cache/processed files location
        seq_length=seq_length,
        seed=1234,
        memmap_workers=1,
        # Dataloader config parameters
        dataloader_type="single",
        num_workers=2,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=False,  # Rewrite existing processed files
        delete_raw=False,  # Keep raw HF dataset cache
    )


def apply_args_to_config(config, args):
    """Apply CLI arguments to ConfigContainer fields."""

    # Training configuration
    if args.max_steps:
        config.train.train_iters = args.max_steps
    if args.gbs:
        config.train.global_batch_size = args.gbs
    if args.mbs:
        config.train.micro_batch_size = args.mbs

    # Optimizer configuration
    if args.lr:
        config.optimizer.lr = args.lr
    if args.min_lr:
        config.optimizer.min_lr = args.min_lr

    # Scheduler configuration
    if args.warmup_iters:
        config.scheduler.lr_warmup_iters = args.warmup_iters

    # PEFT configuration - only override if explicitly provided
    if args.finetune and args.peft_scheme:
        if args.peft_scheme == "lora":
            from megatron.bridge.peft.lora import LoRA

            config.peft = LoRA()
        elif args.peft_scheme == "dora":
            from megatron.bridge.peft.dora import DoRA

            config.peft = DoRA()
        else:
            raise ValueError(f"Unknown PEFT scheme: {args.peft_scheme}")

    # Checkpoint configuration
    if args.pretrained_checkpoint:
        config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    if args.save_dir:
        config.checkpoint.save = args.save_dir
    if args.load_dir:
        config.checkpoint.load = args.load_dir
    if args.save_interval:
        config.checkpoint.save_interval = args.save_interval
    if args.async_save:
        config.checkpoint.async_save = args.async_save

    # Dataset configuration
    logging.info(f"Configuring dataset: type={args.data}")

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
            dataset_root=args.dataset_root, seq_length=args.seq_length or 8192, packed=False
        )
    elif args.data == "squad_packed":
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for squad_packed dataset")
        config.dataset = create_squad_dataset_config(
            dataset_root=args.dataset_root, seq_length=args.seq_length or 8192, packed=True
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
    if args.seq_length:
        config.model.seq_length = args.seq_length
    if args.tensor_parallel_size:
        config.model.tensor_model_parallel_size = args.tensor_parallel_size
    if args.pipeline_parallel_size:
        config.model.pipeline_model_parallel_size = args.pipeline_parallel_size
    if args.context_parallel_size:
        config.model.context_parallel_size = args.context_parallel_size
    if args.virtual_pipeline_size:
        config.model.virtual_pipeline_model_parallel_size = args.virtual_pipeline_size
    if args.expert_parallel_size:
        config.model.expert_model_parallel_size = args.expert_parallel_size
    if args.expert_tensor_parallel_size:
        config.model.expert_tensor_parallel_size = args.expert_tensor_parallel_size

    # Logging configuration
    config.logger.log_timers_to_tensorboard = args.tensorboard is True

    # WandB configuration
    if args.wandb_project:
        config.logger.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.logger.wandb_entity = args.wandb_entity
    if args.wandb_exp_name:
        config.logger.wandb_exp_name = args.wandb_exp_name
    if args.wandb_save_dir:
        config.logger.wandb_save_dir = args.wandb_save_dir

    # Handle convergence mode configuration
    if args.convergence:
        config.logger.log_interval = 1

        # Checkpoint configuration for convergence
        if args.max_steps <= 100:
            # Short convergence runs - save at the end
            config.checkpoint.save_interval = args.max_steps
        else:
            # Long convergence runs - save every 1000 steps
            config.checkpoint.save_interval = 1000

        # Validation configuration for convergence
        if args.max_steps <= 100:
            config.train.eval_interval = args.max_steps
            config.train.eval_iters = 0  # Disable evaluation for short convergence runs
        else:
            config.train.eval_interval = 800

        if args.max_steps > 100:
            config.scheduler.lr_warmup_iters = int(0.01 * args.max_steps)

    if args.precision_config_name:
        config.mixed_precision = get_mixed_precision_config(args.precision_config_name)

    # Profiling configuration
    if args.nsys or args.mem:
        from megatron.bridge.training.config import ProfilingConfig

        config.profiling = ProfilingConfig(
            use_nsys_profiler=args.nsys,
            record_memory_history=args.mem,
            profile_step_start=5,
            profile_step_end=min(6, args.max_steps),
        )

    return config


def setup_argument_parser():
    """Set up and return the argument parser for the training script."""
    parser = argparse.ArgumentParser(description="Megatron-Bridge Recipe Training Script")

    # Model specification
    parser.add_argument("--model-family", required=True, help="Model family (e.g., llama)")
    parser.add_argument("--recipe-name", required=True, help="Recipe name (e.g., llama3_8b_pretrain_config)")
    parser.add_argument("--exp-name", required=True, help="Experiment name for logging and checkpoints")

    # Training modes
    parser.add_argument("--pretrain", action="store_true", help="Run pretraining")
    parser.add_argument("--finetune", action="store_true", help="Run finetuning")
    parser.add_argument(
        "--config-name", type=str, default=None, help="Config name (defaults to pretrain_config and finetune_config"
    )

    # Training configuration
    parser.add_argument("--max-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--gbs", type=int, default=8, help="Global batch size")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    parser.add_argument("--seq-length", type=int, help="Sequence length")
    parser.add_argument(
        "--precision-config-name", type=str, default=None, help="Precision config name in mixed_precision.py"
    )

    # PEFT configuration
    parser.add_argument("--peft-scheme", type=str, default=None, help="PEFT scheme")

    # Parallelism
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", type=int, default=None, help="Pipeline parallel size")
    parser.add_argument("--context-parallel-size", type=int, default=None, help="Context parallel size")
    parser.add_argument("--virtual-pipeline-size", type=int, default=None, help="Virtual pipeline size")
    parser.add_argument("--expert-parallel-size", type=int, default=None, help="Expert parallel size")
    parser.add_argument("--expert-tensor-parallel-size", type=int, default=None, help="Expert tensor parallel size")

    # Optimization
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--min-lr", type=float, help="Minimum learning rate")
    parser.add_argument("--warmup-iters", type=int, help="Warmup iterations")

    # Checkpointing
    parser.add_argument("--pretrained-checkpoint", type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--save-dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--load-dir", type=str, help="Directory to load checkpoints")
    parser.add_argument("--save-interval", type=int, help="Number of iterations between checkpoint saves")
    parser.add_argument("--async-save", action="store_true", help="Enable async checkpoint saving", default=False)

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="mock",
        choices=["mock", "rp2", "squad", "squad_packed"],
        help="Dataset type to use",
    )
    parser.add_argument("--dataset-paths", nargs="*", help="Dataset paths (for rp2 dataset)")
    parser.add_argument("--dataset-root", type=str, help="Dataset root directory (for squad datasets)")
    parser.add_argument("--index-mapping-dir", type=str, help="Index mapping directory (for rp2 dataset)")
    parser.add_argument("--dataset-name", type=str, help="Dataset name (deprecated)")
    parser.add_argument("--packed-sequence", action="store_true", help="Use packed sequences")
    parser.add_argument("--head-only", action="store_true", help="Use only head data (for rp2 dataset)")

    # Tokenizer configuration
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="SentencePieceTokenizer",
        choices=["NullTokenizer", "HuggingFaceTokenizer", "SentencePieceTokenizer"],
        help="Type of tokenizer to use",
    )
    parser.add_argument(
        "--tokenizer-model", type=str, help="Path to tokenizer model (automatically provided by launcher)"
    )
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size for NullTokenizer")

    # Debugging and profiling
    parser.add_argument("--convergence", action="store_true", help="Enable convergence run", default=False)
    parser.add_argument("--nsys", action="store_true", help="Enable nsys profiling", default=False)
    parser.add_argument("--mem", action="store_true", help="Enable torch memory profiling", default=False)
    parser.add_argument(
        "--tensorboard", action="store_true", dest="tensorboard", help="Enable tensorboard logging", default=True
    )
    parser.add_argument(
        "--no-tensorboard", action="store_false", dest="tensorboard", help="Disable tensorboard logging"
    )

    # WandB configuration
    parser.add_argument("--wandb-project", type=str, help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, help="WandB entity name")
    parser.add_argument("--wandb-exp-name", type=str, help="WandB experiment name")
    parser.add_argument("--wandb-save-dir", type=str, help="Directory to save WandB logs locally")

    return parser


def main():
    """Main entry point for the training script."""
    # Set up argument parser
    parser = setup_argument_parser()

    # Parse known args and capture unknown ones for config overrides
    args, unknown_args = parser.parse_known_args()

    # Parse plugin config overrides from unknown arguments
    plugin_config_overrides = parse_plugin_config_overrides(unknown_args)

    # Import recipe dynamically using merged naming convention with legacy fallback.
    #
    # Supported cases (in order):
    # 1) New merged-name API (preferred):
    #    - Path:  megatron.bridge.recipes.<family>.<merged_name>
    #    - Args:  --model-family llama --recipe-name llama3_8b_pretrain_config --pretrain
    #    - Example resolved symbol: megatron.bridge.recipes.llama.llama3_8b_pretrain_config
    #
    # 2) Legacy module API (single module exposes config function):
    #    - Path:  megatron.bridge.recipes.<family>.<module>.<pretrain_config|finetune_config>
    #    - Args:  --model-family llama --recipe-name llama3 --pretrain
    #    - Example resolved symbol: megatron.bridge.recipes.llama.llama3.pretrain_config
    #
    # 3) Oldest attribute API (family __init__ exposes suffixed names):
    #    - Path:  megatron.bridge.recipes.<family>.<recipe_name>_<pretrain_config|finetune_config>
    #    - Args:  --model-family llama --recipe-name llama3_8b --pretrain
    #    - Example resolved symbol: megatron.bridge.recipes.llama.llama3_8b_pretrain_config
    #
    # The resolver below tries (1) then (2) then (3), raising a clear error if none match.
    merged_attr = args.recipe_name
    family_pkg_path = f"megatron.bridge.recipes.{args.model_family}"
    logging.info(f"Attempting merged-name import: {family_pkg_path}.{merged_attr}")

    try:
        family_pkg = importlib.import_module(family_pkg_path)
        if not hasattr(family_pkg, merged_attr):
            raise AttributeError
        config_builder = getattr(family_pkg, merged_attr)
        logging.info(f"Using merged recipe API: {family_pkg_path}.{merged_attr}")
    except Exception:
        # Legacy fallback paths
        # 1) args.recipe_name is a module under the family exposing pretrain_config/finetune_config
        legacy_module_path = f"{family_pkg_path}.{args.recipe_name}"
        logging.info(f"Merged import failed; trying legacy module path: {legacy_module_path}")

        # Determine function name by mode
        if args.pretrain:
            config_name = args.config_name or "pretrain_config"
        elif args.finetune:
            config_name = args.config_name or "finetune_config"
        else:
            raise ValueError("Must specify either --pretrain or --finetune")

        try:
            recipe_module = importlib.import_module(legacy_module_path)
            if not hasattr(recipe_module, config_name):
                raise AttributeError
            config_builder = getattr(recipe_module, config_name)
            logging.info(f"Using legacy module API: {legacy_module_path}.{config_name}")
        except Exception:
            # 2) Oldest style: attribute on family package named <recipe_name>_<config_name>
            # Avoid double suffixing if user already passed a merged name
            if merged_attr.endswith("_pretrain_config") or merged_attr.endswith("_finetune_config"):
                legacy_attr = merged_attr
            else:
                legacy_attr = f"{args.recipe_name}_{config_name}"
            logging.info(f"Trying oldest legacy attribute: {family_pkg_path}.{legacy_attr}")
            family_pkg = importlib.import_module(family_pkg_path)
            if not hasattr(family_pkg, legacy_attr):
                raise ValueError(
                    "Unable to resolve recipe. Tried: "
                    f"(1) {family_pkg_path}.{merged_attr}, "
                    f"(2) {legacy_module_path}.{config_name}, "
                    f"(3) {family_pkg_path}.{legacy_attr}"
                )
            config_builder = getattr(family_pkg, legacy_attr)
            logging.info(f"Using oldest legacy API: {family_pkg_path}.{legacy_attr}")

    base_config = config_builder(dir="/nemo_run/", name=args.exp_name)

    # Apply plugin config overrides first (lower priority)
    if plugin_config_overrides:
        omega_conf, excluded_fields = create_omegaconf_dict_config(base_config)
        updated_conf = parse_hydra_overrides(omega_conf, plugin_config_overrides)
        apply_overrides(base_config, updated_conf, excluded_fields)

    # Apply CLI arguments to config (higher priority - overrides plugin settings)
    final_config = apply_args_to_config(base_config, args)

    # Log final configuration
    if get_rank_safe() == 0:
        logging.info("Final configuration:")
        final_config.print_yaml()

    if args.pretrain:
        logging.info("Starting pretraining")
        from megatron.bridge.training.gpt_step import forward_step
        from megatron.bridge.training.pretrain import pretrain

        pretrain(config=final_config, forward_step_func=forward_step)
    elif args.finetune:
        logging.info("Starting finetuning")
        from megatron.bridge.training.finetune import finetune
        from megatron.bridge.training.gpt_step import forward_step

        finetune(config=final_config, forward_step_func=forward_step)
    else:
        raise ValueError("Must specify either --pretrain or --finetune")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
