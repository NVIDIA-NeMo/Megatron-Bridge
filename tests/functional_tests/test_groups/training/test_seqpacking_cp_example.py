# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.models.gpt_provider import GPTModelProvider
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
class Llama3TinyModelProvider(GPTModelProvider):
    """Small Llama-style GPT provider for packed sequence + CP smoke coverage."""

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    num_query_groups: int = 4
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-05
    rotary_percent: float = 1.0
    rotary_base: int = 500_000
    num_layers: int = 2
    hidden_size: int = 512
    ffn_hidden_size: int = 2048
    num_attention_heads: int = 8
    make_vocab_size_divisible_by: int = 128
    vocab_size: int | None = 50257


class TestPeftSftExample:
    """Run the PEFT SFT example as a functional test with packed sequences + CP."""

    @pytest.mark.run_only_on("GPU")
    def test_sft_example_runs_with_cp_and_packing(self, tmp_path):
        pytest.importorskip("transformer_engine_torch")
        initialize_distributed()

        if torch.distributed.get_world_size() < 2:
            pytest.skip("requires >=2 GPUs for context_parallel_size=2")

        shared_dir = broadcast_path(tmp_path)
        pretrain_checkpoint_dir = os.path.join(shared_dir, "pretrain_checkpoints")
        pretrain_tensorboard_dir = os.path.join(shared_dir, "pretrain_tensorboard")
        sft_checkpoint_dir = os.path.join(shared_dir, "sft_checkpoints")
        sft_tensorboard_dir = os.path.join(shared_dir, "sft_tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
            os.makedirs(pretrain_tensorboard_dir, exist_ok=True)
            os.makedirs(sft_checkpoint_dir, exist_ok=True)
            os.makedirs(sft_tensorboard_dir, exist_ok=True)
        torch.distributed.barrier()

        seq_length = 256
        packed_sequence_size = 512
        pretrain_iters = 1
        sft_iters = 2
        pretrain_cfg = self._create_pretrain_config(
            pretrain_iters,
            pretrain_checkpoint_dir,
            pretrain_tensorboard_dir,
            seq_length,
        )
        cfg = self._create_sft_config(
            sft_iters,
            sft_checkpoint_dir,
            sft_tensorboard_dir,
            pretrain_checkpoint_dir,
            seq_length,
            packed_sequence_size,
        )

        try:
            pretrain(pretrain_cfg, forward_step)
            verify_checkpoint_files(
                pretrain_checkpoint_dir,
                pretrain_cfg.train.train_iters,
                ckpt_format=pretrain_cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=pretrain_cfg.checkpoint.storage_writers_per_rank,
            )

            finetune(cfg, forward_step)
            verify_checkpoint_files(
                sft_checkpoint_dir,
                cfg.train.train_iters,
                ckpt_format=cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg.checkpoint.storage_writers_per_rank,
            )
        finally:
            clear_directories(shared_dir)

    def _create_model_provider(self, seq_length):
        """Create a tiny model provider with CP enabled."""
        return Llama3TinyModelProvider(
            seq_length=seq_length,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=None,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=2,
            sequence_parallel=False,
            calculate_per_token_loss=True,
        )

    def _create_training_config(self, train_iters):
        """Create a small training config for CP=2 on two ranks."""
        return TrainingConfig(
            train_iters=train_iters,
            global_batch_size=2,
            micro_batch_size=1,
            manual_gc=True,
            manual_gc_interval=1,
        )

    def _create_validation_config(self):
        """Disable validation while preserving training loop setup."""
        return ValidationConfig(eval_interval=1, eval_iters=0)

    def _create_optimizer_config(self, lr=3e-3):
        """Create an optimizer config matching other functional smoke tests."""
        return OptimizerConfig(
            optimizer="adam",
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            lr=lr,
            weight_decay=0.01,
            min_lr=1e-6 if lr > 1e-4 else 1e-7,
        )

    def _create_scheduler_config(self, train_iters):
        """Create a scheduler config for a very short run."""
        return SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=0,
            lr_warmup_init=0.0,
            lr_decay_iters=train_iters,
        )

    def _create_ddp_config(self):
        """Create a DDP config that avoids extra overlap buffers for this smoke test."""
        return DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=False,
            use_distributed_optimizer=True,
        )

    def _create_logger_config(self, tensorboard_dir):
        """Create a logger config."""
        return LoggerConfig(
            log_interval=1,
            tensorboard_dir=tensorboard_dir,
        )

    def _create_checkpoint_config(self, save_interval, save_dir, pretrained_checkpoint=None):
        """Create a synchronous checkpoint config to keep this test's memory footprint small."""
        return CheckpointConfig(
            save_interval=save_interval,
            save=save_dir,
            load=None,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
            dist_ckpt_optim_fully_reshardable=True,
        )

    def _create_pretrain_config(self, train_iters, checkpoint_dir, tensorboard_dir, seq_length):
        """Create the cheap checkpoint producer used by the SFT run."""
        return ConfigContainer(
            model=self._create_model_provider(seq_length),
            train=self._create_training_config(train_iters),
            validation=self._create_validation_config(),
            optimizer=self._create_optimizer_config(),
            scheduler=self._create_scheduler_config(train_iters),
            ddp=self._create_ddp_config(),
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
            logger=self._create_logger_config(tensorboard_dir),
            tokenizer=TokenizerConfig(
                tokenizer_type="NullTokenizer",
                vocab_size=50257,
            ),
            checkpoint=self._create_checkpoint_config(train_iters, checkpoint_dir),
            rng=RNGConfig(seed=1234),
            mixed_precision="bf16_mixed",
        )

    def _create_sft_config(
        self,
        train_iters,
        checkpoint_dir,
        tensorboard_dir,
        pretrained_checkpoint_dir,
        seq_length,
        packed_sequence_size,
    ):
        """Create the packed SFT config under test."""
        return ConfigContainer(
            model=self._create_model_provider(seq_length),
            train=self._create_training_config(train_iters),
            validation=self._create_validation_config(),
            optimizer=self._create_optimizer_config(lr=1e-4),
            scheduler=self._create_scheduler_config(train_iters),
            ddp=self._create_ddp_config(),
            dataset=HFDatasetConfig(
                dataset_name="rajpurkar/squad",
                process_example_fn=process_squad_example,
                seq_length=seq_length,
                dataloader_type="batch",
                num_workers=1,
                do_validation=False,
                do_test=False,
                val_proportion=None,
                dataset_kwargs={"pad_to_max_length": True},
                max_train_samples=16,
                packed_sequence_specs=PackedSequenceSpecs(
                    packed_sequence_size=packed_sequence_size,
                    tokenizer_model_name="gpt2",
                    pad_seq_to_mult=2 * 2,
                ),
                rewrite=False,
            ),
            logger=self._create_logger_config(tensorboard_dir),
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model="gpt2",
            ),
            checkpoint=self._create_checkpoint_config(train_iters, checkpoint_dir, pretrained_checkpoint_dir),
            rng=RNGConfig(seed=5678),
            mixed_precision="bf16_mixed",
        )
