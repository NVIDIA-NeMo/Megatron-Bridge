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
Quickstart: Finetune Llama 3.2 1B with Megatron Bridge

Usage:
    Single GPU with LoRA:
        torchrun --nproc_per_node=1 01_quickstart_finetune.py \
            --pretrained-checkpoint /path/to/megatron/checkpoint

    Multiple GPUs (automatic data parallelism):
        torchrun --nproc_per_node=8 01_quickstart_finetune.py \
            --pretrained-checkpoint /path/to/megatron/checkpoint

Prerequisites:
    You need a checkpoint in Megatron format. You can either:
    1. Convert HF checkpoint to Megatron format:
       python examples/conversion/convert_checkpoints.py import \
           --hf-model meta-llama/Llama-3.2-1B \
           --megatron-path ./checkpoints/llama32_1b
    2. Use a checkpoint from pretraining (see 00_quickstart_pretrain.py)

The script uses SQuAD dataset by default. See inline comments for:
- Using your own dataset
- Adjusting LoRA hyperparameters
- Switching to full supervised finetuning

For YAML configuration, see 03_finetune_with_yaml.py
For multi-node training, see launch_with_sbatch.sh or 04_launch_slurm_with_nemo_run.py
"""

import argparse

from megatron.bridge.recipes.llama import llama32_1b_finetune_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune Llama 3.2 1B with LoRA",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint in Megatron format",
    )
    return parser.parse_args()


def main() -> None:
    """Run Llama 3.2 1B finetuning with LoRA."""
    args = parse_args()

    # Load the base finetune configuration
    # Uses LoRA for efficient finetuning on a single GPU
    config = llama32_1b_finetune_config()

    # Load from the pretrained checkpoint
    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    # === Quick test run ===
    config.train.train_iters = 10
    config.scheduler.lr_warmup_iters = 2

    # ===== OPTIONAL CUSTOMIZATIONS =====
    # Uncomment and modify as needed:

    # === Use your own dataset ===
    # Replace SQuAD with your custom dataset
    # Option 1: Simple path override
    # config.dataset.dataset_root = "/path/to/your/dataset"

    # Or replace the dataset with FinetuningDatasetConfig for JSONL data
    # from megatron.bridge.training.config import FinetuningDatasetConfig
    # config.dataset = FinetuningDatasetConfig(
    #     dataset_root="/path/to/your/dataset_dir",  # expects training/validation/test jsonl files
    #     seq_length=config.model.seq_length,
    # )

    # === Adjust learning rate ===
    # config.optimizer.lr = 5e-5

    # === Change checkpoint save frequency ===
    # config.train.save_interval = 100

    # === Adjust LoRA hyperparameters ===
    # Higher rank = more trainable parameters, potentially better quality but slower
    # config.peft.dim = 16  # LoRA rank
    # config.peft.alpha = 32  # LoRA alpha scaling

    # === Full supervised finetuning (no LoRA) ===
    # For full finetuning, reload config with peft=None:
    # config = llama32_1b_finetune_config(peft=None)
    # config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    # Note: Full finetuning uses more memory than LoRA
    # The recipe automatically adjusts parallelism for full SFT

    # Start finetuning
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
