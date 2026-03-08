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
from typing import Any, Optional

import pytest
import torch

import megatron.bridge.training.gpt_step as gpt_step_module
from megatron.bridge.models.llama import Llama3ModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetBuildContext,
    DatasetProvider,
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
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


class _ShortSeqPretrainDataset(torch.utils.data.Dataset):
    """Emit sequences shorter than model seq_length to force PP padding in gpt_step.get_batch()."""

    def __init__(self, size: int, produced_seq_length: int, vocab_size: int = 1024, seed: int = 1234) -> None:
        self.size = int(size)
        self.produced_seq_length = int(produced_seq_length)
        self.vocab_size = int(vocab_size)
        self._g = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: ARG002
        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.produced_seq_length,),
            dtype=torch.long,
            generator=self._g,
        )
        labels = tokens.clone()
        loss_mask = torch.ones(self.produced_seq_length, dtype=torch.float)
        position_ids = torch.arange(self.produced_seq_length, dtype=torch.long)
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


@dataclass(kw_only=True)
class ShortSeqDatasetProvider(DatasetProvider):
    """DatasetProvider with a public `seq_length` matching the model, but emits shorter samples."""

    seq_length: int
    produced_seq_length: int
    vocab_size: int = 1024
    seed: int = 1234
    skip_getting_attention_mask_from_dataset: bool = True

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        train_ds = _ShortSeqPretrainDataset(
            size=context.train_samples,
            produced_seq_length=self.produced_seq_length,
            vocab_size=self.vocab_size,
            seed=self.seed,
        )
        valid_ds = _ShortSeqPretrainDataset(
            size=max(1, context.valid_samples),
            produced_seq_length=self.produced_seq_length,
            vocab_size=self.vocab_size,
            seed=self.seed + 1,
        )
        test_ds = _ShortSeqPretrainDataset(
            size=max(1, context.test_samples),
            produced_seq_length=self.produced_seq_length,
            vocab_size=self.vocab_size,
            seed=self.seed + 2,
        )
        return train_ds, valid_ds, test_ds


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
            verify_checkpoint_files(
                pretrain_checkpoint_dir,
                pretrain_iters,
                ckpt_format=pretrain_cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=pretrain_cfg.checkpoint.storage_writers_per_rank,
            )

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
            verify_checkpoint_files(
                finetune_checkpoint_dir,
                finetune_iters,
                ckpt_format=finetune_cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=finetune_cfg.checkpoint.storage_writers_per_rank,
            )

        finally:
            clear_directories(shared_base_dir)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_then_finetune_with_pp_padding(self, tmp_path, monkeypatch):
        """E2E pretrain -> finetune with PP enabled should exercise gpt_step PP padding.

        This mirrors `examples/debug.py` style (PP on) and the existing e2e flow here,
        but uses a DatasetProvider that emits shorter sequences to force the padding path.
        """
        initialize_distributed()

        if torch.distributed.get_world_size() < 2:
            pytest.skip("PP test requires WORLD_SIZE >= 2")

        shared_base_dir = broadcast_path(tmp_path)
        pretrain_checkpoint_dir, pretrain_tensorboard_dir, finetune_checkpoint_dir, finetune_tensorboard_dir = (
            self._setup_directories(shared_base_dir)
        )

        torch.distributed.barrier()

        # Instrument padding helper to ensure the <seq_length -> pad> path is hit.
        pad_calls = {"padded": 0}
        _orig_pad = gpt_step_module.pad_or_truncate_2d_to_len

        def _pad_wrapper(x, target_len, max_cap, pad_value):  # noqa: ANN001
            if x is not None and x.dim() == 2 and x.size(1) < target_len:
                pad_calls["padded"] += 1
            return _orig_pad(x, target_len, max_cap, pad_value)

        monkeypatch.setattr(gpt_step_module, "pad_or_truncate_2d_to_len", _pad_wrapper, raising=True)

        try:
            model_seq_length = 256
            produced_seq_length = 128  # shorter than model seq length to force padding

            # Pretrain (save a checkpoint to finetune from)
            pretrain_cfg = ConfigContainer(
                model=Llama3ModelProvider145M(
                    seq_length=model_seq_length,
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    num_layers=2,
                ),
                train=TrainingConfig(
                    train_iters=2,
                    eval_interval=10,
                    eval_iters=0,
                    global_batch_size=2,
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
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=2,
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
                dataset=ShortSeqDatasetProvider(
                    seq_length=model_seq_length,
                    produced_seq_length=produced_seq_length,
                    dataloader_type="single",
                    num_workers=1,
                    data_sharding=True,
                    pin_memory=True,
                ),
                logger=LoggerConfig(
                    log_interval=1,
                    tensorboard_dir=pretrain_tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=2,
                    save=pretrain_checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )
            pretrain(pretrain_cfg, forward_step)
            verify_checkpoint_files(pretrain_checkpoint_dir, 2)

            # Finetune from the pretrained checkpoint (finetune() enforces this)
            finetune_cfg = ConfigContainer(
                model=Llama3ModelProvider145M(
                    seq_length=model_seq_length,
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    num_layers=2,
                ),
                train=TrainingConfig(
                    train_iters=1,
                    eval_interval=10,
                    eval_iters=0,
                    global_batch_size=2,
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
                    lr=1e-4,
                    weight_decay=0.01,
                    min_lr=1e-7,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=2,
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
                dataset=ShortSeqDatasetProvider(
                    seq_length=model_seq_length,
                    produced_seq_length=produced_seq_length,
                    dataloader_type="single",
                    num_workers=1,
                    data_sharding=True,
                    pin_memory=True,
                ),
                logger=LoggerConfig(
                    log_interval=1,
                    tensorboard_dir=finetune_tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=1,
                    save=finetune_checkpoint_dir,
                    pretrained_checkpoint=pretrain_checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=5678),
            )
            finetune(finetune_cfg, forward_step)

            if torch.distributed.get_rank() == 0:
                assert pad_calls["padded"] > 0, "Expected PP padding (<seq_length) to be exercised during training"

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
        # Keep warmup strictly below total iterations to avoid scheduler assertion.
        warmup_iters = 2 if train_iters >= 10 else 1
        if train_iters is not None:
            warmup_iters = min(warmup_iters, max(train_iters - 1, 0))
        return ConfigContainer(
            model=Llama3ModelProvider145M(seq_length=seq_length),
            train=TrainingConfig(
                train_iters=train_iters,
                global_batch_size=8,
                micro_batch_size=1,
                exit_signal_handler=True,
            ),
            validation=ValidationConfig(
                eval_interval=5,
                eval_iters=0,
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
                lr=lr,
                weight_decay=0.01,
                min_lr=1e-6 if lr > 1e-4 else 1e-7,
            ),
            scheduler=SchedulerConfig(
                start_weight_decay=0.033,
                end_weight_decay=0.033,
                weight_decay_incr_style="constant",
                lr_decay_style="cosine",
                lr_warmup_iters=warmup_iters,
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
