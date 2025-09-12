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

import pytest
import torch

from megatron.bridge.models.llama import Llama3ModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


@dataclass
class Llama3ModelProvider145M(Llama3ModelProvider):
    rotary_base: int = 500_000
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int | None = None


class TestSupervisedFinetuning:
    """
    Test end to end supervised finetuning: pretrain -> save checkpoint -> finetune using pretrained checkpoint.
    """

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_then_finetune(self, tmp_path):
        """Test end to end supervised finetuning: pretrain -> save checkpoint -> finetune using pretrained checkpoint."""
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        pretrain_checkpoint_dir, pretrain_tensorboard_dir, finetune_checkpoint_dir, finetune_tensorboard_dir = (
            self._setup_directories(shared_base_dir)
        )

        torch.distributed.barrier()

        try:
            seq_length = 512
            pretrain_iters = 10
            finetune_iters = 5

            # Create pretrain config and run
            pretrain_cfg = self._create_config(
                pretrain_iters, pretrain_checkpoint_dir, pretrain_tensorboard_dir, seq_length
            )
            pretrain(pretrain_cfg, forward_step)
            verify_checkpoint_files(pretrain_checkpoint_dir, pretrain_iters)

            # Create finetune config and run (lower LR, different seed, use pretrained checkpoint)
            finetune_cfg = self._create_config(
                finetune_iters,
                finetune_checkpoint_dir,
                finetune_tensorboard_dir,
                seq_length,
                lr=1e-4,
                seed=5678,
                pretrained_checkpoint=pretrain_checkpoint_dir,
            )
            finetune(finetune_cfg, forward_step)
            verify_checkpoint_files(finetune_checkpoint_dir, finetune_iters)

        finally:
            clear_directories(shared_base_dir)

    def _create_config(
        self,
        train_iters,
        checkpoint_dir,
        tensorboard_dir,
        seq_length=512,
        lr=3e-3,
        seed=1234,
        pretrained_checkpoint=None,
    ):
        """Create training configuration with customizable parameters."""
        return ConfigContainer(
            model=Llama3ModelProvider145M(seq_length=seq_length),
            train=TrainingConfig(
                train_iters=train_iters,
                eval_interval=5,
                eval_iters=0,
                global_batch_size=8,
                micro_batch_size=1,
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
                lr=lr,
                weight_decay=0.01,
                min_lr=1e-6 if lr > 1e-4 else 1e-7,
            ),
            scheduler=SchedulerConfig(
                start_weight_decay=0.033,
                end_weight_decay=0.033,
                weight_decay_incr_style="constant",
                lr_decay_style="cosine",
                lr_warmup_iters=2 if train_iters >= 10 else 1,
                lr_warmup_init=0.0,
                lr_decay_iters=train_iters,
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
                random_seed=seed,
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
                save_interval=train_iters,
                save=checkpoint_dir,
                pretrained_checkpoint=pretrained_checkpoint,
                ckpt_format="torch_dist",
                fully_parallel_save=True,
                async_save=True,
            ),
            rng=RNGConfig(seed=seed),
        )

    def _setup_directories(self, base_dir):
        """Setup test directories."""
        pretrain_checkpoint_dir = os.path.join(base_dir, "pretrain_checkpoints")
        pretrain_tensorboard_dir = os.path.join(base_dir, "pretrain_tensorboard")
        finetune_checkpoint_dir = os.path.join(base_dir, "finetune_checkpoints")
        finetune_tensorboard_dir = os.path.join(base_dir, "finetune_tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
            os.makedirs(finetune_checkpoint_dir, exist_ok=True)
            os.makedirs(pretrain_tensorboard_dir, exist_ok=True)
            os.makedirs(finetune_tensorboard_dir, exist_ok=True)

        return pretrain_checkpoint_dir, pretrain_tensorboard_dir, finetune_checkpoint_dir, finetune_tensorboard_dir
