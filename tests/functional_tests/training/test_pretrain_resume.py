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
import shutil
from dataclasses import dataclass

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.hub.models.llama import Llama3ModelProvider
from megatron.hub.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    MockGPTDatasetConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.hub.training.gpt_step import forward_step
from megatron.hub.training.pretrain import pretrain
from tests.functional_tests.utils import broadcast_path, initialize_distributed


@dataclass
class Llama3ModelProvider145M(Llama3ModelProvider):
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16


class TestPretrainResume:
    """
    Test end to end training with checkpoint functionality.
    """

    def clear_directories(self, tmp_path):
        """Teardown method called after each test method."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                if os.path.exists(tmp_path):
                    shutil.rmtree(tmp_path)
            torch.distributed.barrier()

    def _verify_checkpoint_files(self, checkpoint_dir, total_iters):
        """Verify that checkpoint files were created correctly."""
        if torch.distributed.get_rank() == 0:
            latest_tracker_file = os.path.join(checkpoint_dir, "latest_train_state.pt")
            assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

            final_iter_dir = os.path.join(checkpoint_dir, f"iter_{total_iters:07d}")
            assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"

            metadata_file = os.path.join(final_iter_dir, ".metadata")
            assert os.path.exists(metadata_file), "Checkpoint metadata file not found"

            distcp_files = [f for f in os.listdir(final_iter_dir) if f.endswith(".distcp")]
            num_expected_files = 2 * torch.distributed.get_world_size()
            assert len(distcp_files) == num_expected_files, (
                f"Expected {num_expected_files} .distcp files, found {len(distcp_files)}: {distcp_files}"
            )

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_save_load(self, tmp_path):
        """
        Test end to end training with checkpoint saving and resuming functionality.
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
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 20
            checkpoint_iters = 10

            # First training run - train for 10 iterations and save checkpoint
            cfg_first = ConfigContainer(
                model=Llama3ModelProvider145M(),
                train=TrainingConfig(
                    train_iters=checkpoint_iters,
                    eval_interval=5,
                    eval_iters=2,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
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
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=2,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
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
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=checkpoint_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run first training job
            pretrain(cfg_first, forward_step)

            torch.distributed.barrier()

            # Verify checkpoint files from first run
            self._verify_checkpoint_files(checkpoint_dir, checkpoint_iters)

            torch.distributed.barrier()

            # Second training run - resume from checkpoint and train for remaining 10 iterations
            cfg_second = ConfigContainer(
                model=Llama3ModelProvider145M(),
                train=TrainingConfig(
                    train_iters=total_iters,
                    eval_interval=5,
                    eval_iters=2,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
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
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=2,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
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
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=checkpoint_iters,
                    save=checkpoint_dir,
                    load=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run second training job (resume from checkpoint)
            pretrain(cfg_second, forward_step)

            torch.distributed.barrier()

            # Verify checkpoint files from second run
            self._verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            self.clear_directories(shared_base_dir)
