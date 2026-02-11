# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Functional tests for local (non-persistent) checkpointing."""

import os
from dataclasses import dataclass

import pytest
import torch

from megatron.bridge.models.llama import Llama3ModelProvider
from megatron.bridge.training.callbacks import Callback, CallbackContext
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
    ValidationConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
)


@dataclass
class Llama3ModelProvider145M(Llama3ModelProvider):
    """Minimal Llama-3 config for fast functional tests (single GPU)."""

    rotary_base: int = 500_000
    seq_length: int = 1024
    num_layers: int = 1
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int | None = None


class TrainStateAssertCallback(Callback):
    """Callback that records training state at start and end of training.

    Used to verify that checkpoint resume restores the correct iteration,
    training counters, and that the right number of steps are executed.
    """

    def __init__(self):
        self.start_step: int | None = None
        self.end_step: int | None = None
        self.start_consumed_samples: int | None = None
        self.end_consumed_samples: int | None = None
        self.steps_executed: int = 0

    def on_train_start(self, context: CallbackContext) -> None:
        self.start_step = context.state.train_state.step
        self.start_consumed_samples = context.state.train_state.consumed_train_samples

    def on_train_step_end(self, context: CallbackContext) -> None:
        self.steps_executed += 1

    def on_train_end(self, context: CallbackContext) -> None:
        self.end_step = context.state.train_state.step
        self.end_consumed_samples = context.state.train_state.consumed_train_samples


def _make_config(
    *,
    checkpoint_dir: str,
    local_ckpt_dir: str,
    tensorboard_dir: str,
    train_iters: int,
    save_interval: int | None = None,
    non_persistent_save_interval: int | None = None,
    most_recent_k: int = -1,
    load_dir: str | None = None,
) -> ConfigContainer:
    """Build a ConfigContainer for local-checkpoint testing."""
    seq_length = 512

    return ConfigContainer(
        model=Llama3ModelProvider145M(seq_length=seq_length),
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=8,
            micro_batch_size=1,
            exit_signal_handler=True,
        ),
        validation=ValidationConfig(eval_interval=1000, eval_iters=2),
        optimizer=OptimizerConfig(
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
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=False,
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
            seq_length=seq_length,
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
            save=checkpoint_dir,
            load=load_dir,
            save_interval=save_interval,
            non_persistent_save_interval=non_persistent_save_interval,
            non_persistent_ckpt_type="local",
            non_persistent_local_ckpt_dir=local_ckpt_dir,
            non_persistent_local_ckpt_algo="atomic",
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
            most_recent_k=most_recent_k,
        ),
        rng=RNGConfig(seed=1234),
    )


class TestLocalCheckpointing:
    """Functional tests for local (non-persistent) checkpointing."""

    @pytest.mark.run_only_on("GPU")
    def test_local_checkpoint_save_and_resume(self, tmp_path):
        """Verify that training can save a local checkpoint and then
        resume from it, restoring the correct iteration and training
        counters.  No global checkpoint is involved.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "ckpt")
        local_ckpt_dir = os.path.join(shared_base_dir, "local_ckpt")
        tensorboard_dir = os.path.join(shared_base_dir, "tb")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(local_ckpt_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            # Run 1: train for 5 iters, save a local checkpoint at iter 5
            cfg_run1 = _make_config(
                checkpoint_dir=checkpoint_dir,
                local_ckpt_dir=local_ckpt_dir,
                tensorboard_dir=tensorboard_dir,
                train_iters=5,
                non_persistent_save_interval=5,
            )

            pretrain(cfg_run1, forward_step)
            torch.distributed.barrier()

            # Run 2: resume from the local checkpoint and train to iter 10
            cb = TrainStateAssertCallback()
            cfg_run2 = _make_config(
                checkpoint_dir=checkpoint_dir,
                local_ckpt_dir=local_ckpt_dir,
                tensorboard_dir=tensorboard_dir,
                train_iters=10,
                non_persistent_save_interval=5,
                load_dir=checkpoint_dir,
            )

            pretrain(cfg_run2, forward_step, callbacks=[cb])
            torch.distributed.barrier()

            # Verify: resumed from iteration 5, not 0
            assert cb.start_step == 5, f"Expected resume from step 5, got {cb.start_step}"

            # Verify: finished at iteration 10
            assert cb.end_step == 10, f"Expected end at step 10, got {cb.end_step}"

            # Verify: only ran 5 new steps (not 10 from scratch)
            assert cb.steps_executed == 5, f"Expected 5 steps, got {cb.steps_executed}"

            # Verify: consumed_train_samples was restored (not reset to 0)
            assert cb.start_consumed_samples > 0, (
                f"Expected consumed_train_samples > 0 at resume, got {cb.start_consumed_samples}"
            )

        finally:
            clear_directories(shared_base_dir)

    @pytest.mark.run_only_on("GPU")
    def test_local_checkpoint_save_resume_with_most_recent_k(self, tmp_path):
        """Verify local checkpoint save and resume with most_recent_k
        enabled end-to-end.  Exercises save-side cleanup skipping and
        load-side handling of CkptID tuples together.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "ckpt")
        local_ckpt_dir = os.path.join(shared_base_dir, "local_ckpt")
        tensorboard_dir = os.path.join(shared_base_dir, "tb")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(local_ckpt_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            # Run 1: train for 5 iters with local ckpt + most_recent_k
            cfg_run1 = _make_config(
                checkpoint_dir=checkpoint_dir,
                local_ckpt_dir=local_ckpt_dir,
                tensorboard_dir=tensorboard_dir,
                train_iters=5,
                non_persistent_save_interval=5,
                most_recent_k=1,
            )

            pretrain(cfg_run1, forward_step)
            torch.distributed.barrier()

            # Run 2: resume and train to iter 10 with most_recent_k
            cb = TrainStateAssertCallback()
            cfg_run2 = _make_config(
                checkpoint_dir=checkpoint_dir,
                local_ckpt_dir=local_ckpt_dir,
                tensorboard_dir=tensorboard_dir,
                train_iters=10,
                non_persistent_save_interval=5,
                most_recent_k=1,
                load_dir=checkpoint_dir,
            )

            pretrain(cfg_run2, forward_step, callbacks=[cb])
            torch.distributed.barrier()

            # Verify: resumed from iteration 5, finished at 10, ran 5 steps
            assert cb.start_step == 5, f"Expected resume from step 5, got {cb.start_step}"
            assert cb.end_step == 10, f"Expected end at step 10, got {cb.end_step}"
            assert cb.steps_executed == 5, f"Expected 5 steps, got {cb.steps_executed}"
            assert cb.start_consumed_samples > 0, (
                f"Expected consumed_train_samples > 0 at resume, got {cb.start_consumed_samples}"
            )

        finally:
            clear_directories(shared_base_dir)
