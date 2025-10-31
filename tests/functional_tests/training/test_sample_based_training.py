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

"""Functional tests for sample-based training that run on 2 GPUs with torchrun."""

import logging

import torch

from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing_samples
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    RerunStateMachineConfig,
    RNGConfig,
    TrainingConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.utils.common_utils import get_rank_safe


_logger: logging.Logger = logging.getLogger(__name__)


class TestSampleBasedTrainingFunctional:
    """Functional tests for sample-based training on 2 GPUs."""

    def test_sample_based_training_mini_run(self):
        """Mini end-to-end test that runs a few training steps with sample-based training."""
        # Use the new sample-based optimizer utility
        optimizer_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing_samples(
            precision="bf16-mixed",
            lr_warmup_samples=8,  # Very small for quick test
            lr_decay_samples=24,
            max_lr=1e-3,
            min_lr=1e-4,
        )

        cfg = ConfigContainer(
            train=TrainingConfig(
                micro_batch_size=1,
                global_batch_size=4,  # 2 GPUs * 2 data_parallel_size
                train_samples=32,  # Sample-based training (8 iterations)
                eval_iters=2,
                eval_interval=4,
                skip_train=False,
            ),
            model=Llama32ModelProvider1B(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                num_layers=2,
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=256,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
            ),
            optimizer=optimizer_cfg,
            scheduler=scheduler_cfg,
            dataset=MockGPTDatasetConfig(
                random_seed=1234,
                sequence_length=256,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                num_dataset_builder_threads=1,
                data_sharding=True,
                dataloader_type="single",
                num_workers=1,
            ),
            logger=LoggerConfig(
                log_interval=2,
                tensorboard_dir=None,  # Disable tensorboard for testing
            ),
            tokenizer=TokenizerConfig(
                tokenizer_type="NullTokenizer",
                vocab_size=10000,
            ),
            checkpoint=CheckpointConfig(),
            dist=DistributedInitConfig(),
            ddp=DistributedDataParallelConfig(use_distributed_optimizer=True),
            rng=RNGConfig(seed=42),
            rerun_state_machine=RerunStateMachineConfig(),
        )

        assert cfg.train.train_samples == 32
        assert cfg.scheduler.lr_decay_samples == 24
        assert cfg.scheduler.lr_warmup_samples == 8

        pretrain(config=cfg, forward_step_func=forward_step)

        if get_rank_safe() == 0:
            _logger.debug(f"Trained for {cfg.train.train_samples} samples over {cfg.train.train_iters} iterations")
            _logger.debug(f"Used sample-based scheduler with {cfg.scheduler.lr_warmup_samples} warmup samples")
