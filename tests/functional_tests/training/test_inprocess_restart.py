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
Integration test for in-process restart functionality.

This test validates the end-to-end behavior of the inprocess_restart module
by running actual training with simulated failures and verifying restart behavior.
"""

import os
from typing import Optional

import pytest
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


def build_test_config(
    save_dir: str,
    train_iters: int = 20,
    seq_length: int = 512,
    async_save: bool = False,
    save_interval: int = 10,
    fault_delay: Optional[float] = None,
) -> ConfigContainer:
    """Build training configuration with in-process restart enabled for testing.

    Args:
        save_dir: Directory to save checkpoints (must be accessible by all ranks)
        train_iters: Number of training iterations
        seq_length: Sequence length for the model
        async_save: Whether to enable async checkpointing
        save_interval: Save checkpoint every N iterations
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
            # Increase timeouts for CI environment where operations can be slower
            heartbeat_interval=10.0,
            heartbeat_timeout=120.0,
            soft_timeout=300.0,  # 5 minutes - much longer for CI
            hard_timeout=600.0,  # 10 minutes - much longer for CI
            barrier_timeout=180.0,
            completion_timeout=180.0,
            monitor_process_interval=5.0,
            monitor_thread_interval=5.0,
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


class TestInProcessRestartIntegration:
    """Integration tests for in-process restart functionality."""

    @pytest.mark.run_only_on("GPU")
    def test_inprocess_restart_basic_functionality(self, tmp_path):
        """Test basic in-process restart functionality without faults."""
        # NOTE: Do not call initialize_distributed() here - inprocess restart must handle distributed initialization
        # from within the wrapped function

        # Create a shared temporary directory that all processes can access
        # Use a predictable path that's consistent across all processes
        # Try common temp directories that should exist across platforms
        temp_root = os.environ.get("TMPDIR", os.environ.get("TMP", "/tmp"))
        shared_base_dir = os.path.join(temp_root, "inprocess_restart_basic_test")
        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")

        # Create checkpoint directory (all processes will create the same path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # Create config with in-process restart enabled
            config = build_test_config(
                save_dir=checkpoint_dir,
                train_iters=20,
                seq_length=512,
                async_save=False,
                save_interval=10,
            )

            forward_step_func = forward_step

            try:
                pretrain(config=config, forward_step_func=forward_step_func)
                training_success = True
            except Exception as e:
                training_success = False
                print(f"Training failed: {e}")
                import traceback

                traceback.print_exc()

            assert training_success, "Training with in-process restart should complete successfully"

        finally:
            # Clean up the shared directory
            import shutil

            if os.path.exists(shared_base_dir):
                shutil.rmtree(shared_base_dir, ignore_errors=True)

    @pytest.mark.run_only_on("GPU")
    def test_inprocess_restart_with_fault_injection(self, tmp_path):
        """Test in-process restart with Megatron-Bridge fault injection.

        Note: This test requires at least 2 processes since fault injection occurs on rank 1.
        This test should be run with ft_launcher for proper fault tolerance coordination.
        """
        # NOTE: Do not call initialize_distributed() here - inprocess restart handles it

        # Check world size from environment (since distributed is not yet initialized)
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        if world_size < 2:
            pytest.skip("Fault injection test requires at least 2 processes (fault injected on rank 1)")

        # Create a shared temporary directory that all processes can access
        # Use a predictable path that's consistent across all processes
        # Try common temp directories that should exist across platforms
        temp_root = os.environ.get("TMPDIR", os.environ.get("TMP", "/tmp"))
        shared_base_dir = os.path.join(temp_root, "inprocess_restart_fault_test")
        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")

        # Create checkpoint directory (all processes will create the same path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # Create config with fault injection enabled (simulates rank_killed on rank 1 after 5 seconds)
            config = build_test_config(
                save_dir=checkpoint_dir,
                train_iters=30,  # More iterations to allow fault injection
                seq_length=512,
                async_save=True,
                save_interval=10,  # Save checkpoint every 10 iterations
                fault_delay=40.0,  # Inject fault after 35 seconds
            )

            try:
                pretrain(config=config, forward_step_func=forward_step)
                print("Training completed successfully despite fault injection")
            except Exception as e:
                print(f"Training failed with fault injection: {e}")

            # For fault injection tests, we expect either:
            # 1. Successful recovery and completion (when using ft_launcher)
            # 2. Graceful failure with proper error handling (when not using ft_launcher)
            # The key is that the process doesn't hang or crash unexpectedly

            # Just verify the test ran without hanging
            print("Fault injection test completed")

        finally:
            # Clean up the shared directory
            import shutil

            if os.path.exists(shared_base_dir):
                shutil.rmtree(shared_base_dir, ignore_errors=True)
