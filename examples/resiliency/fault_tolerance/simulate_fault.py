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
Fault Tolerance with Simulated Fault Injection

This example demonstrates fault tolerance with simulated rank failures.
A specified rank will be killed or hung after a delay, triggering the
fault tolerance recovery mechanism.

IMPORTANT - Timing Requirements:
- This script must be run with ft_launcher, not torch.distributed.run.
- The following timing relationship must be satisfied for successful recovery:

    checkpoint_time < fault_delay < total_training_time

  Where:
    - checkpoint_time: Wall-clock time to reach and finalize the first checkpoint
    - fault_delay: Seconds before fault injection (--fault-delay)
    - total_training_time: Wall-clock time for all training iterations

  If fault_delay < checkpoint_time: Job restarts from iteration 0 indefinitely
  If fault_delay > total_training_time: Training completes before fault triggers

Default Configuration (designed to work out-of-box):
- train_iters=2000: ~90 seconds of training with tiny model
- save_interval=200: First checkpoint at iteration 200 (~9 seconds)
- fault_delay=60: Fault triggers at 60 seconds (after checkpoint, before completion)

Usage:
    uv run ft_launcher \\
        --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \\
        --nnodes=1 --nproc-per-node=2 \\
        --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \\
        --ft-param-rank_out_of_section_timeout=300 \\
        --max-restarts=3 \\
        examples/resiliency/fault_tolerance/simulate_fault.py

    # Or use the launch script:
    ./examples/resiliency/fault_tolerance/run_fault_tolerance.sh --simulate-fault

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
    train_iters: int = 100,
    fault_type: str = "rank_killed",
    fault_rank: int = 1,
    fault_delay: float = 30.0,
) -> ConfigContainer:
    """Create training configuration with fault simulation enabled.

    Args:
        checkpoint_dir: Directory for checkpoints (required for recovery).
        train_iters: Number of training iterations.
        fault_type: Type of fault to simulate ("rank_killed", "rank_hung", "random").
        fault_rank: Which rank to fail (use -1 for random selection).
        fault_delay: Seconds to wait before injecting the fault.
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

    # Checkpoint configuration (required for fault recovery)
    checkpoint_config = CheckpointConfig(
        save=checkpoint_dir,
        load=checkpoint_dir,
        save_interval=200,  # Save checkpoints for recovery (less frequent for longer runs)
        ckpt_format="torch_dist",
        async_save=True,
    )

    # Fault Tolerance Configuration with Fault Simulation
    # See: https://nvidia.github.io/nvidia-resiliency-ext/
    # Note: calc_ft_timeouts is disabled for fault simulation since we want to
    # demonstrate recovery behavior, not timeout learning. Timeout learning
    # requires enough training iterations before each checkpoint.
    ft_config = FaultToleranceConfig(
        enable_ft_package=True,
        calc_ft_timeouts=False,
        # Fault simulation settings
        simulate_fault=True,
        simulated_fault_type=fault_type,  # "rank_killed", "rank_hung", or "random"
        simulated_fault_rank=fault_rank if fault_rank >= 0 else None,  # None = random
        simulated_fault_base_delay=fault_delay,  # Seconds before fault injection
    )

    return ConfigContainer(
        train=train_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        dataset=dataset_config,
        logger=LoggerConfig(log_interval=5, tensorboard_dir=None),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=10000),
        checkpoint=checkpoint_config,
        rng=RNGConfig(seed=1234),
        ddp=ddp_config,
        ft=ft_config,
    )


def main() -> None:
    """Run fault tolerance example with simulated fault injection."""
    parser = argparse.ArgumentParser(description="Fault Tolerance with Simulated Fault")
    parser.add_argument("--train-iters", type=int, default=2000, help="Number of training iterations")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/tmp/megatron_bridge_ft_fault_example",
        help="Checkpoint directory (must be shared across all ranks)",
    )
    parser.add_argument(
        "--fault-type",
        type=str,
        default="rank_killed",
        choices=["rank_killed", "rank_hung", "random"],
        help="Type of fault to simulate",
    )
    parser.add_argument(
        "--fault-rank",
        type=int,
        default=1,
        help="Rank to fail (-1 for random)",
    )
    parser.add_argument(
        "--fault-delay",
        type=float,
        default=60.0,
        help="Seconds before fault injection (must be after first checkpoint is saved)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Ensure checkpoint directory exists (all ranks use the same path)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    config = create_config(
        checkpoint_dir=args.checkpoint_dir,
        train_iters=args.train_iters,
        fault_type=args.fault_type,
        fault_rank=args.fault_rank,
        fault_delay=args.fault_delay,
    )
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
