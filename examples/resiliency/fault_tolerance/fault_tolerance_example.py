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
Fault Tolerance Example

Demonstrates fault tolerance during training using nvidia-resiliency-ext.
Fault tolerance monitors training progress through sections (setup, step,
checkpointing) and enables automatic restart on hang detection.

Prerequisites:
    - HuggingFace token with access to Llama models (set HF_TOKEN env var)
    - Accept Llama license at https://huggingface.co/meta-llama/Llama-3.2-1B

IMPORTANT: This script must be run with ft_launcher, not torch.distributed.run.

Fault Simulation Mode (--simulate-fault):
    Demonstrates fault recovery by killing a rank after a delay.

    Timing requirements for successful recovery:
        checkpoint_time < fault_delay < total_training_time

    Where:
        - checkpoint_time: Wall-clock time to reach and finalize the first checkpoint
        - fault_delay: Seconds before fault injection (--fault-delay)
        - total_training_time: Wall-clock time for all training iterations

    If fault_delay < checkpoint_time: Job restarts from iteration 0 indefinitely
    If fault_delay > total_training_time: Training completes before fault triggers

Usage:
    uv run ft_launcher \\
        --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \\
        --nnodes=1 --nproc-per-node=2 \\
        --ft-rank_section_timeouts=setup:600,step:180,checkpointing:420 \\
        --ft-rank_out_of_section_timeout=300 \\
        examples/resiliency/fault_tolerance/fault_tolerance_example.py

    # With fault simulation:
    uv run ft_launcher ... --max-restarts=3 \\
        examples/resiliency/fault_tolerance/fault_tolerance_example.py --simulate-fault

    # Or use the launch script:
    ./examples/resiliency/fault_tolerance/run_fault_tolerance.sh
    ./examples/resiliency/fault_tolerance/run_fault_tolerance.sh --simulate-fault

Documentation:
    - Megatron-Bridge: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/resiliency.html
    - NVRx Fault Tolerance: https://nvidia.github.io/nvidia-resiliency-ext/
"""

import argparse
import logging
import os

import torch

from megatron.bridge.models import AutoBridge
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


# Default model - smallest Llama 3.2 for fast examples
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"


def create_config(
    checkpoint_dir: str,
    model_id: str = DEFAULT_MODEL,
    train_iters: int = 50,
    save_interval: int = 25,
    simulate_fault: bool = False,
    fault_type: str = "rank_killed",
    fault_rank: int = 1,
    fault_delay: float = 60.0,
) -> ConfigContainer:
    """Create training configuration with fault tolerance enabled.

    Args:
        checkpoint_dir: Directory for checkpoints (required for FT state persistence).
        model_id: HuggingFace model ID to load.
        train_iters: Number of training iterations.
        save_interval: Checkpoint save interval.
        simulate_fault: Whether to simulate a fault for testing recovery.
        fault_type: Type of fault to simulate ("rank_killed", "rank_hung", "random").
        fault_rank: Which rank to fail (use -1 for random selection).
        fault_delay: Seconds to wait before injecting the fault.
    """
    seq_length = 512  # Short sequence for fast examples

    # Load model configuration from HuggingFace
    bridge = AutoBridge.from_hf_pretrained(model_id, torch_dtype=torch.bfloat16)
    model_config = bridge.to_megatron_provider()
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1
    model_config.context_parallel_size = 1
    model_config.sequence_parallel = False
    model_config.bf16 = True
    model_config.seq_length = seq_length

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
        adam_eps=1e-8,
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

    checkpoint_config = CheckpointConfig(
        save=checkpoint_dir,
        load=checkpoint_dir,
        save_interval=save_interval,
        ckpt_format="torch_dist",
        async_save=True,
    )

    # Fault Tolerance Configuration
    # See: https://nvidia.github.io/nvidia-resiliency-ext/
    # When simulating faults, disable timeout calculation since we want to
    # demonstrate recovery behavior, not timeout learning.
    ft_config = FaultToleranceConfig(
        enable_ft_package=True,
        calc_ft_timeouts=not simulate_fault,
        # Fault simulation settings (only used when simulate_fault=True)
        simulate_fault=simulate_fault,
        simulated_fault_type=fault_type,
        simulated_fault_rank=fault_rank if fault_rank >= 0 else None,
        simulated_fault_base_delay=fault_delay,
    )

    return ConfigContainer(
        train=train_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        dataset=dataset_config,
        logger=LoggerConfig(log_interval=10, tensorboard_dir=None),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=model_config.padded_vocab_size),
        checkpoint=checkpoint_config,
        rng=RNGConfig(seed=1234),
        ddp=ddp_config,
        ft=ft_config,
    )


def main() -> None:
    """Run fault tolerance example with configurable parameters."""
    parser = argparse.ArgumentParser(description="Fault Tolerance Example")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--train-iters", type=int, default=None, help="Number of training iterations")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/tmp/megatron_bridge_ft_example",
        help="Checkpoint directory (must be shared across all ranks)",
    )

    # Fault simulation options
    parser.add_argument(
        "--simulate-fault",
        action="store_true",
        help="Enable fault simulation to test recovery",
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
        help="Seconds before fault injection (must be after first checkpoint)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Set defaults based on mode
    if args.train_iters is None:
        # Fault simulation needs more iterations so training outlasts the fault delay
        args.train_iters = 2000 if args.simulate_fault else 50

    # Checkpoint less frequently for longer runs
    save_interval = 200 if args.simulate_fault else 25

    # Ensure checkpoint directory exists (all ranks use the same path)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    config = create_config(
        checkpoint_dir=args.checkpoint_dir,
        model_id=args.model,
        train_iters=args.train_iters,
        save_interval=save_interval,
        simulate_fault=args.simulate_fault,
        fault_type=args.fault_type,
        fault_rank=args.fault_rank,
        fault_delay=args.fault_delay,
    )
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
