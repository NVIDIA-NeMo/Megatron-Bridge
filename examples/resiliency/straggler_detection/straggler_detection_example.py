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
NVRx Straggler Detection Example

Demonstrates straggler detection during training using nvidia-resiliency-ext.
Straggler detection monitors GPU performance across ranks and identifies
underperforming GPUs.

Prerequisites:
    - HuggingFace token with access to Llama models (set HF_TOKEN env var)
    - Accept Llama license at https://huggingface.co/meta-llama/Llama-3.2-1B

Usage:
    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/resiliency/straggler_detection/straggler_detection_example.py

    # With more GPUs
    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        examples/resiliency/straggler_detection/straggler_detection_example.py

    # Customize training iterations
    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/resiliency/straggler_detection/straggler_detection_example.py --train-iters 200

Documentation:
    - Megatron-Bridge: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/resiliency.html
    - NVRx Straggler Detection: https://nvidia.github.io/nvidia-resiliency-ext/
"""

import argparse
import logging

import torch

from megatron.bridge.models import AutoBridge
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    NVRxStragglerDetectionConfig,
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
    model_id: str = DEFAULT_MODEL,
    train_iters: int = 100,
    report_interval: float = 5.0,
) -> ConfigContainer:
    """Create training configuration with straggler detection enabled.

    Args:
        model_id: HuggingFace model ID to load.
        train_iters: Number of training iterations.
        report_interval: How often (in seconds) to generate straggler reports.
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

    # NVRx Straggler Detection Configuration
    # See: https://nvidia.github.io/nvidia-resiliency-ext/
    nvrx_config = NVRxStragglerDetectionConfig(
        enabled=True,
        report_time_interval=report_interval,  # Generate reports at this interval
        calc_relative_gpu_perf=True,  # Compare GPU performance across ranks
        calc_individual_gpu_perf=True,  # Track individual GPU performance over time
        num_gpu_perf_scores_to_print=5,  # Number of best/worst performers to print
        gpu_relative_perf_threshold=0.7,  # Flag GPUs below 70% of average
        gpu_individual_perf_threshold=0.7,  # Flag GPUs with 30%+ performance drop
        stop_if_detected=False,  # Don't stop training on straggler detection
        enable_logging=True,
        profiling_interval=1,  # Profile every iteration
    )

    return ConfigContainer(
        train=train_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        dataset=dataset_config,
        logger=LoggerConfig(log_interval=10, tensorboard_dir=None),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=model_config.padded_vocab_size),
        checkpoint=CheckpointConfig(save=None, load=None, save_interval=None),
        rng=RNGConfig(seed=1234),
        ddp=ddp_config,
        nvrx_straggler=nvrx_config,
    )


def main() -> None:
    """Run straggler detection example with configurable parameters."""
    parser = argparse.ArgumentParser(description="NVRx Straggler Detection Example")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--train-iters", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--report-interval", type=float, default=5.0, help="Straggler report interval in seconds")
    args = parser.parse_args()

    # Configure logging to show straggler detection output
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    config = create_config(
        model_id=args.model,
        train_iters=args.train_iters,
        report_interval=args.report_interval,
    )
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
