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
NVIDIA Resiliency Extension In-Process Restart Example for Megatron-Bridge

This example demonstrates how to use NVIDIA Resiliency Extension (NVRx) in-process restart
with Megatron-Bridge for fault-tolerant distributed training.

Key Features:
- In-process restart capability via NVRx
- Async checkpointing with proper cleanup during restart
- Fault injection for testing (via Megatron-Bridge fault tolerance)
- Compatible with both ft_launcher execution

Usage Examples:

1. With fault injection (requires ft_launcher for coordination):
   ```bash
   ft_launcher \\
     --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \\
     --nnodes=1 --nproc-per-node=2 \\
     --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \\
     --ft-param-rank_out_of_section_timeout=300 \\
     --monitor-interval=5 --max-restarts=3 \\
     --ft-restart-policy=min-healthy \\
     examples/resiliency/inprocess_restart.py \\
     --train-iters 80 --inject-fault-after 2.0
   ```

Environment Variables:
- TORCH_CPP_LOG_LEVEL=error (recommended to reduce noise)
- MASTER_ADDR, MASTER_PORT (set by launcher or default to localhost:29500)

For more information on NVRx in-process restart, see:
https://nvidia.github.io/nvidia-resiliency-ext/inprocess/
"""

import argparse
import os
import tempfile

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    FaultToleranceConfig,
    InProcessRestartConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def build_config(
    save_dir: str,
    train_iters: int,
    seq_length: int,
    async_save: bool,
    save_interval: int,
    fault_delay: float | None = None,
) -> ConfigContainer:
    """Build training configuration with in-process restart enabled.

    Args:
        save_dir: Directory to save checkpoints (must be accessible by all ranks)
        train_iters: Number of training iterations
        seq_length: Sequence length for the model
        async_save: Whether to enable async checkpointing
        fault_delay: If set, inject a fault after this many seconds (requires ft_launcher)

    Returns:
        Complete configuration for training with in-process restart
    """
    model_cfg = Llama32ModelProvider1B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        attention_softmax_in_fp32=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        seq_length=seq_length,
        vocab_size=None,
    )

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=train_iters + 1,  # Disable evaluation for simplicity
            eval_iters=0,
            global_batch_size=8,
            micro_batch_size=1,
            exit_signal_handler=True,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=2,
            lr_warmup_init=0.0,
            lr_decay_iters=train_iters,
            override_opt_param_scheduler=True,
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-5,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            lr=3e-3,
            weight_decay=0.01,
            min_lr=1e-6,
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=MockGPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(
            log_interval=5,
            tensorboard_dir=None,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=10000,
        ),
        checkpoint=CheckpointConfig(
            save=save_dir,
            load=save_dir,
            save_interval=save_interval,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=async_save,
        ),
        rng=RNGConfig(seed=1234),
        inprocess_restart=InProcessRestartConfig(
            enabled=True,
            granularity="rank",
            active_world_size=int(os.getenv("WORLD_SIZE", "2")),
            empty_cuda_cache=True,
        ),
        ft=FaultToleranceConfig(
            enable_ft_package=fault_delay is not None,
            simulate_fault=fault_delay is not None,
            simulated_fault_type="rank_killed",
            simulated_fault_rank=1,
            simulated_fault_base_delay=fault_delay if fault_delay else 0,
        )
        if fault_delay is not None
        else None,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NVRx In-Process Restart Example for Megatron-Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With fault injection (requires ft_launcher)
  ft_launcher --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \\
    --nnodes=1 --nproc-per-node=2 \\
    --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \\
    --ft-param-rank_out_of_section_timeout=300 \\
    --monitor-interval=5 --max-restarts=3 --ft-restart-policy=min-healthy \\
    examples/resiliency/inprocess_restart.py \\
    --train-iters 100 --inject-fault-after 30.0 --save-interval 40

Environment:
  Set TORCH_CPP_LOG_LEVEL=error to reduce NVRx logging noise.
        """,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints - must be accessible by all ranks (default: temp directory)",
    )
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length for the model (default: 512)")
    parser.add_argument("--train-iters", type=int, default=50, help="Number of training iterations (default: 50)")
    parser.add_argument(
        "--inject-fault-after",
        type=float,
        default=None,
        help="Inject fault after this many seconds (requires ft_launcher with fault tolerance config)",
    )
    parser.add_argument("--async-save", action="store_true", help="Enable async checkpointing (default: disabled)")
    parser.add_argument(
        "--save-interval", type=int, default=50, help="Save checkpoint every N iterations (default: 50)"
    )
    return parser.parse_args()


def main():
    """Main function demonstrating NVRx in-process restart with Megatron-Bridge."""
    args = parse_args()

    # Create checkpoint directory
    save_dir = args.save_dir
    if save_dir is None:
        tmpbase = tempfile.gettempdir()
        save_dir = os.path.join(tmpbase, "mbridge_inproc_restart_example")

    # Only rank 0 creates directory (use environment RANK, not torch.distributed)
    if int(os.getenv("RANK", "0")) == 0:
        os.makedirs(save_dir, exist_ok=True)

    # Build configuration with in-process restart enabled
    config = build_config(
        save_dir=save_dir,
        train_iters=args.train_iters,
        seq_length=args.seq_length,
        async_save=args.async_save,
        save_interval=args.save_interval,
        fault_delay=args.inject_fault_after,
    )

    print("Starting training with in-process restart enabled...")
    if args.inject_fault_after:
        print(f"Fault will be injected after {args.inject_fault_after} seconds on rank 1")

    # Run training - pretrain() handles in-process restart internally
    pretrain(config, forward_step)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
