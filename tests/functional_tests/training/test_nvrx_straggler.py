# Copyright (c) 2025, NVIDIA CORPORATION.
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

#!/usr/bin/env python3
"""
End-to-end functional test for NVRx straggler detection with megatron/hub training.

This test runs the actual pretrain function with NVRx straggler detection
enabled, using mock data and a tiny model configuration for fast testing.
"""

import argparse
import logging
import os
import tempfile
import time

import torch

from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
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
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


def create_functional_test_config(enable_nvrx: bool = True) -> ConfigContainer:
    """Create a complete minimal configuration for functional testing, based on test_pretrain.py."""

    seq_length = 512
    train_config = TrainingConfig(
        train_iters=10,
        micro_batch_size=1,
        global_batch_size=2,
        eval_interval=10,
        eval_iters=0,
    )

    model_config = Llama32ModelProvider1B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        attention_softmax_in_fp32=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        seq_length=seq_length,
        make_vocab_size_divisible_by=128,
        vocab_size=None,
    )

    dataset_config = MockGPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        sequence_length=seq_length,
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
        lr=3e-3,
        weight_decay=0.01,
        min_lr=1e-6,
    )

    scheduler_config = SchedulerConfig(
        start_weight_decay=0.033,
        end_weight_decay=0.033,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_warmup_iters=2,
        lr_warmup_init=0.0,
        lr_decay_iters=train_config.train_iters,
        override_opt_param_scheduler=True,
    )

    tokenizer_config = TokenizerConfig(
        tokenizer_type="NullTokenizer",
        vocab_size=10000,
    )

    logger_config = LoggerConfig(
        log_interval=5,
        tensorboard_dir=None,
    )

    checkpoint_config = CheckpointConfig(
        save=None,
        load=None,
        save_interval=None,
    )

    rng_config = RNGConfig(seed=1234)

    dist_config = DistributedInitConfig()

    ddp_config = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
        use_distributed_optimizer=True,
    )

    nvrx_config = None
    if enable_nvrx:
        nvrx_config = NVRxStragglerDetectionConfig(
            enabled=True,
            report_time_interval=2.0,
            calc_relative_gpu_perf=True,
            calc_individual_gpu_perf=True,
            num_gpu_perf_scores_to_print=4,
            gpu_relative_perf_threshold=0.7,
            gpu_individual_perf_threshold=0.7,
            stop_if_detected=False,
            enable_logging=True,
            profiling_interval=1,
            logger_name="nvrx_functional_test",
        )

    return ConfigContainer(
        train=train_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        dataset=dataset_config,
        logger=logger_config,
        tokenizer=tokenizer_config,
        checkpoint=checkpoint_config,
        rng=rng_config,
        dist=dist_config,
        ddp=ddp_config,
        nvrx_straggler=nvrx_config,
    )


def create_timed_forward_step_func(sleep_time: float = 1.0):
    """Create a forward step function that sleeps before calling the real forward_step.

    This simulates work being done and allows NVRx to measure performance differences.
    Only rank 1 will be slow to simulate a straggler scenario.

    Args:
        sleep_time: Time to sleep in seconds before each forward step (only for rank 1)

    Returns:
        A forward step function compatible with megatron training
    """

    def timed_forward_step_func(state: GlobalState, data_iterator, model, return_schedule_plan: bool = False):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            time.sleep(sleep_time)
            print(f"Rank {torch.distributed.get_rank()}: Simulated slow forward step (slept {sleep_time}s)")

        return forward_step(state, data_iterator, model, return_schedule_plan=return_schedule_plan)

    return timed_forward_step_func


def setup_test_logging(log_file: str, rank: int):
    """Setup logging to capture all output including NVRx logs to a file."""

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler for the test log
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(f"%(asctime)s [Rank {rank}] %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Ensure NVRx logger also writes to our file
    nvrx_logger = logging.getLogger("nvrx_functional_test")
    nvrx_logger.setLevel(logging.INFO)
    nvrx_logger.addHandler(file_handler)

    # Also capture any other potential NVRx loggers
    for logger_name in ("nvidia_resiliency", "straggler", "nvrx"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return file_handler


def test_nvrx_straggler_detection_end_to_end(sleep_time: float = 1.0):
    """
    End-to-end functional test that runs actual megatron training with NVRx.

    This test:
    1. Sets up a complete megatron training configuration
    2. Uses mock data and small model for fast execution
    3. Runs the actual pretrain function
    4. Verifies NVRx straggler detection is working by checking logs
    """
    rank = get_rank_safe()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Use rank-specific log files to avoid conflicts
        log_file = os.path.join(temp_dir, f"training_rank_{rank}.log")

        # Setup logging to capture everything
        file_handler = setup_test_logging(log_file, rank)

        try:
            # Create complete configuration
            config = create_functional_test_config(enable_nvrx=True)
            forward_step_func = create_timed_forward_step_func(sleep_time=sleep_time)

            try:
                pretrain(
                    config=config,
                    forward_step_func=forward_step_func,
                )
                training_success = True

            except Exception:
                training_success = False
                if rank == 0:
                    import traceback

                    traceback.print_exc()

            assert training_success, "Training must complete successfully"

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            # Allow time for file system to sync
            time.sleep(3.0)

            if rank == 0:
                log_content = ""
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        log_content = f.read()

                # Check for NVRx straggler detection activity
                has_gpu_perf_logs = "GPU relative performance" in log_content
                has_rank_scores = any(keyword in log_content for keyword in ["Rank=", "Score="])
                has_straggler_detection = "straggler" in log_content.lower()
                has_nvidia_resiliency = "nvidia_resiliency" in log_content.lower()

                # Assert that NVRx is actually working
                assert has_gpu_perf_logs or has_rank_scores or has_straggler_detection or has_nvidia_resiliency, (
                    f"Expected NVRx straggler detection logs not found. "
                    f"GPU perf logs: {has_gpu_perf_logs}, "
                    f"Rank scores: {has_rank_scores}, "
                    f"Straggler detection: {has_straggler_detection}, "
                    f"Nvidia resiliency: {has_nvidia_resiliency}. "
                    f"This suggests nvidia-resiliency-ext is not properly installed or integrated."
                )

                # If we detect actual straggler reports, verify they contain rank information
                if has_gpu_perf_logs or has_rank_scores:
                    assert "Rank=" in log_content, "Expected rank information in straggler detection logs"
                elif has_straggler_detection or has_nvidia_resiliency:
                    print_rank_0("NVRx straggler detection is present in logs")

        except Exception:
            raise

        finally:
            # Cleanup logging handlers
            if file_handler:
                file_handler.close()
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)


def main():
    """Main function for running the functional test directly."""
    parser = argparse.ArgumentParser(description="NVRx Straggler Detection End-to-End Functional Test")
    parser.add_argument(
        "--sleep-time", type=float, default=1.0, help="Sleep time per forward step to simulate work on rank 1"
    )

    args = parser.parse_args()
    test_nvrx_straggler_detection_end_to_end(args.sleep_time)


if __name__ == "__main__":
    main()
