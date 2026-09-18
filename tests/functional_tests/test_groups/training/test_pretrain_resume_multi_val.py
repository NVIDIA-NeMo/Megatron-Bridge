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

"""End-to-end save/resume for per-set validation bookkeeping on the GPT sampler path.

Exercises ``ValidationConfig.multiple_validation_sets`` with two validation datasets so
each set gets its own ``consumed_valid_samples_per_set`` offset. With a single set the
per-set counter equals the old aggregate, so this test uses two sets: a regression to the
shared aggregate would advance the counters wrong and fail the assertions.
"""

import os
from dataclasses import dataclass

import pytest
import torch

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
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
from megatron.bridge.training.utils.checkpoint_utils import (
    TRACKER_PREFIX,
    get_checkpoint_train_state_filename,
)
from tests.functional_tests.test_groups.training.test_pretrain_resume import Llama3TinyModelProvider
from tests.functional_tests.utils import broadcast_path, clear_directories, initialize_distributed


@dataclass(kw_only=True)
class MultiValMockGPTProvider(DatasetProvider):
    """Returns the mock validation dataset repeated ``num_val_sets`` times as a list.

    A list validation slot is what triggers the bridge's multi-set validation path, so
    each set resumes from its own ``consumed_valid_samples_per_set`` offset.
    """

    mock_config: MockGPTDatasetConfig
    num_val_sets: int = 2

    def build_datasets(self, context: DatasetBuildContext):
        self.mock_config.tokenizer = context.tokenizer
        # Nested configs are not finalized by ConfigContainer (only cfg.dataset itself is);
        # finalize() runs the deferred __post_init__ that switches the config to mock mode.
        self.mock_config.finalize()
        num_samples = [context.train_samples, context.valid_samples, context.test_samples]
        train_ds, valid_ds, test_ds = pretrain_train_valid_test_datasets_provider(num_samples, self.mock_config)
        return train_ds, [valid_ds] * self.num_val_sets, test_ds


def _make_config(*, train_iters, checkpoint_dir, tensorboard_dir, seq_length, gbs, mbs, total_iters, load=None):
    mock = MockGPTDatasetConfig(
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
    return ConfigContainer(
        model=Llama3TinyModelProvider(seq_length=seq_length),
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            exit_signal_handler=True,
        ),
        validation=ValidationConfig(
            eval_interval=5,
            eval_iters=2,
            eval_at_step_zero=True,
            multiple_validation_sets=True,
            validation_set_names=["val_a", "val_b"],
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
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
        dataset=MultiValMockGPTProvider(
            mock_config=mock,
            num_val_sets=2,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(log_interval=5, tensorboard_dir=tensorboard_dir),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=10000),
        checkpoint=CheckpointConfig(
            save_interval=5,
            save=checkpoint_dir,
            load=load,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            # Synchronous saves: this test reads latest_train_state.pt right after pretrain()
            # returns, and async finalization writes that file on a background thread, racing
            # the read. Async save/resume mechanics are covered by test_pretrain_resume.py.
            async_save=False,
            dist_ckpt_optim_fully_reshardable=True,
        ),
        rng=RNGConfig(seed=1234),
    )


def _read_valid_counters(checkpoint_dir: str) -> tuple[list[int], int]:
    # Raw torch.load rather than read_train_state: the latter is lru_cache'd (a second call
    # returns the first call's result even after the file changed) and broadcasts the result
    # across ranks (calling it on a subset of ranks desyncs collectives).
    state_dict = torch.load(
        get_checkpoint_train_state_filename(checkpoint_dir, prefix=TRACKER_PREFIX),
        map_location="cpu",
        weights_only=True,
    )
    return state_dict["consumed_valid_samples_per_set"].tolist(), state_dict["consumed_valid_samples"].item()


class TestPretrainResumeMultiVal:
    """Per-set validation resume on the GPT sampler path."""

    @pytest.mark.run_only_on("GPU")
    def test_per_set_consumed_valid_samples_resume(self, tmp_path, capsys):
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        seq_length = 512
        gbs, mbs = 8, 1
        checkpoint_iters, total_iters = 5, 10

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)
        torch.distributed.barrier()

        try:
            # First run: train to the checkpoint iteration; validation runs and advances
            # a per-set counter for each of the two validation sets.
            pretrain(
                _make_config(
                    train_iters=checkpoint_iters,
                    checkpoint_dir=checkpoint_dir,
                    tensorboard_dir=tensorboard_dir,
                    seq_length=seq_length,
                    gbs=gbs,
                    mbs=mbs,
                    total_iters=total_iters,
                ),
                forward_step,
            )
            torch.distributed.barrier()

            # The step-0 run emits one pre-train validation block per set.
            captured_first = capsys.readouterr().out
            if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
                assert captured_first.count("(pre-train validation)") == 2, captured_first

            # Read and assert on every rank so no rank runs ahead of the others.
            after_first, aggregate_first = _read_valid_counters(checkpoint_dir)
            # Two sets, each advanced the same amount (both evaluated every interval).
            assert len(after_first) == 2, f"expected 2 per-set counters, got {after_first}"
            assert after_first[0] == after_first[1], f"per-set counters diverged: {after_first}"
            assert after_first[0] > 0, f"per-set counter did not advance: {after_first}"
            assert aggregate_first == sum(after_first), (
                f"aggregate {aggregate_first} != sum of per-set counters {after_first}"
            )
            torch.distributed.barrier()

            # Second run: resume and train to total_iters. The counters must continue from
            # the restored per-set values, not reset or double-count the aggregate.
            pretrain(
                _make_config(
                    train_iters=total_iters,
                    checkpoint_dir=checkpoint_dir,
                    tensorboard_dir=tensorboard_dir,
                    seq_length=seq_length,
                    gbs=gbs,
                    mbs=mbs,
                    total_iters=total_iters,
                    load=checkpoint_dir,
                ),
                forward_step,
            )
            torch.distributed.barrier()

            # The resumed run starts at a nonzero step, so eval_at_step_zero must not re-run.
            captured_second = capsys.readouterr().out
            if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
                assert "(pre-train validation)" not in captured_second, captured_second

            after_second, aggregate_second = _read_valid_counters(checkpoint_dir)
            assert len(after_second) == 2, f"expected 2 per-set counters, got {after_second}"
            assert after_second[0] == after_second[1], f"per-set counters diverged: {after_second}"
            assert after_second[0] > after_first[0], (
                f"per-set counters did not continue after resume: {after_first} -> {after_second}"
            )
            assert aggregate_second == sum(after_second), (
                f"aggregate {aggregate_second} != sum of per-set counters {after_second}"
            )
            torch.distributed.barrier()
        finally:
            clear_directories(shared_base_dir)
