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

import os
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from megatron.bridge.models.llama import Llama3ModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
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
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


@dataclass
class Llama3ModelProviderFSDP145M(Llama3ModelProvider):
    """Small Llama3 model configuration for FSDP testing."""

    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int | None = None
    # Disable gradient accumulation fusion for FSDP
    gradient_accumulation_fusion: bool = False


def create_fsdp_model_config(seq_length: int, bf16: bool = True, **kwargs) -> Llama3ModelProviderFSDP145M:
    """Create a standardized FSDP model configuration."""
    base_config = {
        "seq_length": seq_length,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "attention_softmax_in_fp32": True,
        "make_vocab_size_divisible_by": 128,
        "vocab_size": None,
    }
    if bf16:
        base_config.update(
            {
                "bf16": True,
                "pipeline_dtype": torch.bfloat16,
            }
        )
    base_config.update(kwargs)
    return Llama3ModelProviderFSDP145M(**base_config)


def create_base_training_config(
    train_iters: int, global_batch_size: int = 8, micro_batch_size: int = 1, **kwargs
) -> TrainingConfig:
    """Create a standardized training configuration."""
    base_config = {
        "train_iters": train_iters,
        "eval_interval": train_iters + 1,  # Disable evaluation to avoid hanging
        "eval_iters": 0,  # No evaluation iterations
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "exit_signal_handler": True,
    }
    base_config.update(kwargs)
    return TrainingConfig(**base_config)


def create_base_optimizer_config(**kwargs) -> OptimizerConfig:
    """Create a standardized optimizer configuration."""
    base_config = {
        "optimizer": "adam",
        "bf16": True,
        "fp16": False,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-5,
        "use_distributed_optimizer": True,
        "clip_grad": 1.0,
        "lr": 3e-3,
        "weight_decay": 0.01,
        "min_lr": 1e-6,
    }
    base_config.update(kwargs)
    return OptimizerConfig(**base_config)


def create_base_scheduler_config(total_iters: int, **kwargs) -> SchedulerConfig:
    """Create a standardized scheduler configuration."""
    base_config = {
        "start_weight_decay": 0.033,
        "end_weight_decay": 0.033,
        "weight_decay_incr_style": "constant",
        "lr_decay_style": "cosine",
        "lr_warmup_iters": 2,
        "lr_warmup_init": 0.0,
        "lr_decay_iters": total_iters,
        "override_opt_param_scheduler": True,
    }
    base_config.update(kwargs)
    return SchedulerConfig(**base_config)


def create_base_ddp_config(overlap_param_gather: bool = True, **kwargs) -> DistributedDataParallelConfig:
    """Create a standardized DDP configuration for FSDP."""
    base_config = {
        "check_for_nan_in_grad": True,
        "grad_reduce_in_fp32": True,
        "overlap_grad_reduce": True,
        "overlap_param_gather": overlap_param_gather,
        "average_in_collective": False,  # Required for FSDP
        "data_parallel_sharding_strategy": "optim_grads_params",  # For Megatron FSDP only
        "use_distributed_optimizer": True,
        "use_megatron_fsdp": True,  # Enable FSDP in DDP config too
    }
    base_config.update(kwargs)
    return DistributedDataParallelConfig(**base_config)


def create_base_dataset_config(seq_length: int, **kwargs) -> MockGPTDatasetConfig:
    """Create a standardized dataset configuration."""
    base_config = {
        "random_seed": 1234,
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "sequence_length": seq_length,
        "num_dataset_builder_threads": 1,
        "data_sharding": True,
        "dataloader_type": "single",
        "num_workers": 1,
    }
    base_config.update(kwargs)
    return MockGPTDatasetConfig(**base_config)


def create_base_logger_config(tensorboard_dir: Optional[str] = None, log_interval: int = 5, **kwargs) -> LoggerConfig:
    """Create a standardized logger configuration."""
    base_config = {
        "log_interval": log_interval,
        "log_params_norm": True,
    }
    if tensorboard_dir:
        base_config["tensorboard_dir"] = tensorboard_dir
    base_config.update(kwargs)
    return LoggerConfig(**base_config)


def create_base_tokenizer_config(**kwargs) -> TokenizerConfig:
    """Create a standardized tokenizer configuration."""
    base_config = {
        "tokenizer_type": "NullTokenizer",
        "vocab_size": 10000,
    }
    base_config.update(kwargs)
    return TokenizerConfig(**base_config)


def create_base_checkpoint_config(
    checkpoint_dir: Optional[str] = None, load_dir: Optional[str] = None, save_interval: Optional[int] = None, **kwargs
) -> CheckpointConfig:
    """Create a standardized checkpoint configuration."""
    base_config = {
        "ckpt_format": "fsdp_dtensor",  # Use FSDP DTensor format
        "fully_parallel_save": True,
        "async_save": False,  # Disable async save for testing
    }
    if checkpoint_dir:
        base_config["save"] = checkpoint_dir
    if load_dir:
        base_config["load"] = load_dir
    if save_interval:
        base_config["save_interval"] = save_interval
    base_config.update(kwargs)
    return CheckpointConfig(**base_config)


def create_fsdp_config_container(
    seq_length: int,
    train_iters: int,
    checkpoint_dir: Optional[str] = None,
    load_dir: Optional[str] = None,
    save_interval: Optional[int] = None,
    tensorboard_dir: Optional[str] = None,
    overlap_param_gather: bool = True,
    **overrides,
) -> ConfigContainer:
    """Create a complete FSDP configuration container with common defaults."""
    return ConfigContainer(
        model=create_fsdp_model_config(seq_length, **overrides.pop("model", {})),
        dist=DistributedInitConfig(use_megatron_fsdp=True),
        train=create_base_training_config(train_iters, **overrides.pop("train", {})),
        optimizer=create_base_optimizer_config(**overrides.pop("optimizer", {})),
        scheduler=create_base_scheduler_config(train_iters, **overrides.pop("scheduler", {})),
        ddp=create_base_ddp_config(overlap_param_gather, **overrides.pop("ddp", {})),
        dataset=create_base_dataset_config(seq_length, **overrides.pop("dataset", {})),
        logger=create_base_logger_config(tensorboard_dir, **overrides.pop("logger", {})),
        tokenizer=create_base_tokenizer_config(**overrides.pop("tokenizer", {})),
        checkpoint=create_base_checkpoint_config(
            checkpoint_dir, load_dir, save_interval, **overrides.pop("checkpoint", {})
        ),
        rng=RNGConfig(seed=1234, **overrides.pop("rng", {})),
    )


class TestMegatronFSDP:
    """
    Test end to end training with Megatron FSDP and fsdp_dtensor checkpoint functionality.
    """

    @pytest.mark.run_only_on("GPU")
    def test_fsdp_pretrain_basic(self, tmp_path):
        """
        Test basic FSDP training without checkpointing.
        """
        initialize_distributed()

        torch.distributed.barrier()

        try:
            seq_length = 512
            total_iters = 10

            cfg = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=total_iters,
                overlap_param_gather=False,
            )

            # Run training
            pretrain(cfg, forward_step)

            torch.distributed.barrier()

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_fsdp_pretrain_with_checkpoint(self, tmp_path):
        """
        Test FSDP training with checkpoint saving using fsdp_dtensor format.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            seq_length = 512
            total_iters = 10

            # Create config with checkpointing enabled
            cfg = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=total_iters,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                save_interval=10,
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify FSDP DTensor checkpoint files
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters, ckpt_format=cfg.checkpoint.ckpt_format)

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_fsdp_pretrain_save_resume(self, tmp_path):
        """
        Test FSDP training with checkpoint saving and resuming using fsdp_dtensor format.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            seq_length = 512
            total_iters = 10
            checkpoint_iters = 5

            # First training run - train for 10 iterations and save checkpoint
            cfg_first = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=checkpoint_iters,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                save_interval=checkpoint_iters,
                scheduler={"lr_decay_iters": total_iters},  # Override scheduler for total iterations
            )

            # Run first training job
            pretrain(cfg_first, forward_step)

            torch.distributed.barrier()

            # Verify FSDP DTensor checkpoint files from first run
            verify_checkpoint_files(checkpoint_dir, checkpoint_iters, ckpt_format=cfg_first.checkpoint.ckpt_format)

            torch.distributed.barrier()

            # Second training run - resume from checkpoint and train remaining iterations
            cfg_second = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=total_iters,
                checkpoint_dir=checkpoint_dir,
                load_dir=checkpoint_dir,  # Resume from checkpoint
                tensorboard_dir=tensorboard_dir,
                save_interval=checkpoint_iters,
                scheduler={"lr_decay_iters": total_iters},  # Override scheduler for total iterations
            )

            # Run second training job (resume from checkpoint)
            pretrain(cfg_second, forward_step)

            torch.distributed.barrier()

            # Verify FSDP DTensor checkpoint files from second run (should be at total_iters)
            verify_checkpoint_files(checkpoint_dir, total_iters, ckpt_format=cfg_second.checkpoint.ckpt_format)

        finally:
            clear_directories(shared_base_dir)
