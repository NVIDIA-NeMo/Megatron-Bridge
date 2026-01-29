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
Basic Fault Tolerance Example

This example demonstrates how to enable fault tolerance during training
using the nvidia-resiliency-ext package. Fault tolerance monitors training
progress through sections (setup, step, checkpointing) and enables automatic
restart on hang detection.

IMPORTANT: This script must be run with ft_launcher, not torch.distributed.run.

Usage:
    uv run ft_launcher \\
        --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \\
        --nnodes=1 --nproc-per-node=2 \\
        --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \\
        --ft-param-rank_out_of_section_timeout=300 \\
        examples/resiliency/fault_tolerance/basic_fault_tolerance.py

    # Or use the launch script:
    ./examples/resiliency/fault_tolerance/run_fault_tolerance.sh

Documentation:
    - Megatron-Bridge: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/resiliency.html
    - NVRx Fault Tolerance: https://nvidia.github.io/nvidia-resiliency-ext/
"""

import argparse
import logging
import os
from dataclasses import dataclass

import torch

from megatron.bridge.models.llama import Llama3ModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    FaultToleranceConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


@dataclass
class TinyLlama3Config(Llama3ModelProvider):
    """Tiny Llama3 model (~145M params) for fast example execution."""

    rotary_base: int = 500_000
    num_layers: int = 4
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int | None = None


def create_config(
    checkpoint_dir: str,
    train_iters: int = 50,
    calc_timeouts: bool = True,
) -> ConfigContainer:
    """Create training configuration with fault tolerance enabled.

    Args:
        checkpoint_dir: Directory for checkpoints (required for FT state persistence).
        train_iters: Number of training iterations.
        calc_timeouts: Whether to calculate and update FT timeouts based on observed times.
    """
    seq_length = 2048

    model_config = TinyLlama3Config(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        attention_softmax_in_fp32=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        seq_length=seq_length,
        make_vocab_size_divisible_by=128,
    )

    train_config = TrainingConfig(
        train_iters=train_iters,
        micro_batch_size=4,
        global_batch_size=8,
        eval_interval=train_iters + 1,  # Disable evaluation
        eval_iters=0,
        exit_signal_handler=True,
    )

    dataset_config = MockGPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=seq_length,
        num_dataset_builder_threads=1,
        data_sharding=True,
        dataloader_type="single",
        num_workers=1,
    )

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        lr=1e-4,
        weight_decay=0.01,
        min_lr=1e-6,
    )

    scheduler_config = SchedulerConfig(
        start_weight_decay=0.01,
        end_weight_decay=0.01,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_warmup_iters=2,
        lr_warmup_init=0.0,
        lr_decay_iters=train_iters,
        override_opt_param_scheduler=True,
    )

    ddp_config = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
        use_distributed_optimizer=True,
    )

    # Checkpoint configuration (required for fault tolerance)
    checkpoint_config = CheckpointConfig(
        save=checkpoint_dir,
        load=checkpoint_dir,
        save_interval=25,  # Save every 25 iterations
        ckpt_format="torch_dist",
        async_save=True,  # Async checkpoints for better performance
    )

    # Fault Tolerance Configuration
    # See: https://nvidia.github.io/nvidia-resiliency-ext/
    ft_config = FaultToleranceConfig(
        enable_ft_package=True,
        calc_ft_timeouts=calc_timeouts,  # Learn optimal timeouts from observed intervals
    )

    return ConfigContainer(
        train=train_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        dataset=dataset_config,
        logger=LoggerConfig(log_interval=10, tensorboard_dir=None),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=10000),
        checkpoint=checkpoint_config,
        rng=RNGConfig(seed=1234),
        ddp=ddp_config,
        ft=ft_config,
    )


def main() -> None:
    """Run fault tolerance example with configurable parameters."""
    parser = argparse.ArgumentParser(description="Fault Tolerance Example")
    parser.add_argument("--train-iters", type=int, default=50, help="Number of training iterations")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/tmp/megatron_bridge_ft_example",
        help="Checkpoint directory (must be shared across all ranks)",
    )
    parser.add_argument(
        "--no-calc-timeouts",
        action="store_true",
        help="Disable automatic timeout calculation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Ensure checkpoint directory exists (all ranks use the same path)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    config = create_config(
        checkpoint_dir=args.checkpoint_dir,
        train_iters=args.train_iters,
        calc_timeouts=not args.no_calc_timeouts,
    )
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
