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

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.hub.models.llama import Llama32ModelProvider1B
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
from megatron.hub.training.step import forward_step


class TestPretrain:
    """
    Test end to end training with checkpoint functionality.
    """

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_checkpoint(self, tmp_path):
        """
        Test end to end training with checkpoint functionality.
        """
        checkpoint_dir = str(tmp_path / "checkpoints")
        tensorboard_dir = str(tmp_path / "tensorboard")

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 100

            model_cfg = Llama32ModelProvider1B(
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

            # Config Container
            cfg = ConfigContainer(
                model=model_cfg,
                train=TrainingConfig(
                    train_iters=total_iters,
                    eval_interval=50,
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
                    save_interval=40,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Check for the latest checkpoint tracker file
            if torch.distributed.get_rank() == 0:
                latest_tracker_file = os.path.join(checkpoint_dir, "latest_train_state.pt")
                assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

                # Check for the final checkpoint directory (should be iter_0000020)
                final_iter_dir = os.path.join(checkpoint_dir, f"iter_{total_iters:07d}")
                assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"

                # For distributed checkpoints, check for the metadata file
                metadata_file = os.path.join(final_iter_dir, ".metadata")
                assert os.path.exists(metadata_file), "Checkpoint metadata file not found"

        finally:
            # pytest's tmp_path fixture doesn't clean up immediately.
            # Clean up manually.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    if os.path.exists(checkpoint_dir):
                        shutil.rmtree(checkpoint_dir)
                    if os.path.exists(tensorboard_dir):
                        shutil.rmtree(tensorboard_dir)
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
