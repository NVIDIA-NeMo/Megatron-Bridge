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

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.models.llama import Llama3ModelProvider
from megatron.bridge.peft.lora import LoRA
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
    verify_peft_checkpoint_smaller,
)


@dataclass
class Llama3ModelProvider145M(Llama3ModelProvider):
    rotary_base: int = 500_000
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    make_vocab_size_divisible_by: int = 128
    vocab_size: int | None = 128256


class TestLoRAFinetune:
    """
    Test end to end LoRA finetuning: pretrain -> save checkpoint -> finetune with LoRA.
    """

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_then_lora_finetune(self, tmp_path):
        """Test end to end LoRA finetuning: pretrain -> save checkpoint -> finetune with LoRA."""
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        pretrain_checkpoint_dir, pretrain_tensorboard_dir, lora_checkpoint_dir, lora_tensorboard_dir = (
            self._setup_directories(shared_base_dir)
        )

        torch.distributed.barrier()

        try:
            seq_length = 512
            pretrain_iters = 10
            lora_iters = 5

            # Create pretrain config and run
            pretrain_cfg = self._create_pretrain_config(
                pretrain_iters, pretrain_checkpoint_dir, pretrain_tensorboard_dir, seq_length
            )
            pretrain(pretrain_cfg, forward_step)
            verify_checkpoint_files(pretrain_checkpoint_dir, pretrain_iters)

            # Create LoRA config and run finetuning
            lora_cfg = self._create_lora_config(
                lora_iters, lora_checkpoint_dir, lora_tensorboard_dir, pretrain_checkpoint_dir, seq_length
            )
            finetune(lora_cfg, forward_step)
            verify_checkpoint_files(lora_checkpoint_dir, lora_iters)
            verify_peft_checkpoint_smaller(pretrain_checkpoint_dir, lora_checkpoint_dir, pretrain_iters, lora_iters)

        finally:
            clear_directories(shared_base_dir)

    @pytest.mark.run_only_on("GPU")
    def test_lora_save_and_resume(self, tmp_path):
        """
        Test LoRA finetuning with save and resume functionality (simulating job interruption).
        """

        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        pretrain_checkpoint_dir, pretrain_tensorboard_dir, lora_checkpoint_dir, lora_tensorboard_dir = (
            self._setup_directories(shared_base_dir, "_resume")
        )

        torch.distributed.barrier()

        try:
            seq_length = 512
            pretrain_iters = 10
            initial_lora_iters = 6  # First phase of LoRA training
            total_lora_iters = 12  # Total LoRA training iterations

            # First run: Pretrain and save checkpoint
            pretrain_cfg = self._create_pretrain_config(
                pretrain_iters, pretrain_checkpoint_dir, pretrain_tensorboard_dir, seq_length
            )

            # Run pretrain
            pretrain(pretrain_cfg, forward_step)

            verify_checkpoint_files(pretrain_checkpoint_dir, pretrain_iters)

            # Second run: LoRA finetuning initial phase (will be "interrupted")

            # Initial LoRA training configuration (use total iters for scheduler)
            lora_initial_cfg = self._create_lora_config(
                initial_lora_iters,
                lora_checkpoint_dir,
                lora_tensorboard_dir,
                pretrain_checkpoint_dir,
                seq_length,
                scheduler_total_iters=total_lora_iters,
            )

            # Run initial LoRA finetuning (simulate job getting interrupted)
            finetune(lora_initial_cfg, forward_step)

            verify_checkpoint_files(lora_checkpoint_dir, initial_lora_iters)

            # Third run: Resume LoRA finetuning from checkpoint (adapter-only states)
            lora_resume_cfg = self._create_lora_config(
                total_lora_iters,
                lora_checkpoint_dir,
                lora_tensorboard_dir,
                pretrain_checkpoint_dir,
                seq_length,
                load_checkpoint=lora_checkpoint_dir,
                scheduler_total_iters=total_lora_iters,  # Keep total for scheduler calculation
            )
            # Override save interval for final phase and use checkpoint scheduler settings
            lora_resume_cfg.checkpoint.save_interval = total_lora_iters - initial_lora_iters
            lora_resume_cfg.scheduler.use_checkpoint_opt_param_scheduler = True  # Use scheduler state from checkpoint

            # Run resumed LoRA finetuning (should continue from iteration 6 to 12)
            finetune(lora_resume_cfg, forward_step)

            verify_checkpoint_files(lora_checkpoint_dir, total_lora_iters)
            verify_peft_checkpoint_smaller(
                pretrain_checkpoint_dir, lora_checkpoint_dir, pretrain_iters, initial_lora_iters
            )
            verify_peft_checkpoint_smaller(
                pretrain_checkpoint_dir, lora_checkpoint_dir, pretrain_iters, total_lora_iters
            )

        finally:
            clear_directories(shared_base_dir)

    def _create_model_provider(self, seq_length=512, tensor_parallel_size=1, pipeline_parallel_size=1):
        """Create a model provider with specified configuration."""
        return Llama3ModelProvider145M(
            seq_length=seq_length,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            pipeline_dtype=torch.bfloat16,
            sequence_parallel=(tensor_parallel_size > 1),
        )

    def _create_training_config(self, train_iters, global_batch_size=8, micro_batch_size=1):
        """Create a training configuration."""
        return TrainingConfig(
            train_iters=train_iters,
            eval_interval=5,
            eval_iters=0,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        )

    def _create_optimizer_config(self, lr=3e-3):
        """Create an optimizer configuration."""
        return OptimizerConfig(
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
        )

    def _create_scheduler_config(self, total_iters):
        """Create a scheduler configuration."""
        return SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=2 if total_iters >= 10 else 1,
            lr_warmup_init=0.0,
            lr_decay_iters=total_iters,
        )

    def _create_ddp_config(self):
        """Create a DDP configuration."""
        return DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        )

    def _create_mock_dataset_config(self, seq_length, seed=1234):
        """Create a mock dataset configuration."""
        return MockGPTDatasetConfig(
            random_seed=seed,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        )

    def _create_squad_dataset_config(self, seq_length, seed=5678, packed_sequences=False):
        """Create a SQuAD dataset configuration."""
        if packed_sequences:
            dataset_kwargs = {"pad_to_max_length": True}
            packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length)
        else:
            dataset_kwargs = {}
            packed_sequence_specs = None

        config = HFDatasetConfig(
            dataset_name="squad",
            process_example_fn=process_squad_example,
            seq_length=seq_length,
            seed=seed,
            dataloader_type="cyclic" if packed_sequences else "single",
            num_workers=1,
            do_validation=False,
            do_test=False,
            val_proportion=None,
            dataset_kwargs=dataset_kwargs,
            packed_sequence_specs=packed_sequence_specs,
            rewrite=False,
        )

        return config

    def _create_pretrain_tokenizer_config(self):
        """Create a tokenizer configuration for pretraining."""
        return TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=9999,
        )

    def _create_finetune_tokenizer_config(self):
        """Create a tokenizer configuration for finetuning."""
        return TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
        )

    def _create_logger_config(self, tensorboard_dir):
        """Create a logger configuration."""
        return LoggerConfig(
            log_interval=5,
            tensorboard_dir=tensorboard_dir,
        )

    def _create_checkpoint_config(self, save_interval, save_dir, pretrained_checkpoint=None, load_dir=None):
        """Create a checkpoint configuration."""
        return CheckpointConfig(
            save_interval=save_interval,
            save=save_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            load=load_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
        )

    def _create_rng_config(self, seed=1234):
        """Create an RNG configuration."""
        return RNGConfig(seed=seed)

    def _create_lora_peft(self, dim=16, alpha=32, dropout=0.1):
        """Create a LoRA PEFT configuration."""
        return LoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            dim=dim,
            alpha=alpha,
            dropout=dropout,
        )

    def _create_pretrain_config(
        self,
        train_iters,
        checkpoint_dir,
        tensorboard_dir,
        seq_length=512,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    ):
        """Create complete pretrain configuration with model."""
        model = self._create_model_provider(seq_length, tensor_parallel_size, pipeline_parallel_size)

        return ConfigContainer(
            model=model,
            train=self._create_training_config(train_iters),
            optimizer=self._create_optimizer_config(),
            scheduler=self._create_scheduler_config(train_iters),
            ddp=self._create_ddp_config(),
            dataset=self._create_mock_dataset_config(seq_length),
            logger=self._create_logger_config(tensorboard_dir),
            tokenizer=self._create_pretrain_tokenizer_config(),
            checkpoint=self._create_checkpoint_config(train_iters, checkpoint_dir),
            rng=self._create_rng_config(),
        )

    def _create_lora_config(
        self,
        train_iters,
        checkpoint_dir,
        tensorboard_dir,
        pretrained_checkpoint_dir,
        seq_length=512,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        packed_sequences=False,
        load_checkpoint=None,
        scheduler_total_iters=None,
    ):
        """Create complete LoRA finetuning configuration with model and PEFT."""
        model = self._create_model_provider(seq_length, tensor_parallel_size, pipeline_parallel_size)
        lora_peft = self._create_lora_peft()

        # Use scheduler_total_iters if provided, otherwise use train_iters
        scheduler_iters = scheduler_total_iters if scheduler_total_iters is not None else train_iters

        return ConfigContainer(
            model=model,
            train=self._create_training_config(train_iters),
            optimizer=self._create_optimizer_config(lr=1e-4),  # Lower LR for finetuning
            scheduler=self._create_scheduler_config(scheduler_iters),
            ddp=self._create_ddp_config(),
            dataset=self._create_squad_dataset_config(seq_length, packed_sequences=packed_sequences),
            logger=self._create_logger_config(tensorboard_dir),
            tokenizer=self._create_finetune_tokenizer_config(),
            checkpoint=self._create_checkpoint_config(
                train_iters, checkpoint_dir, pretrained_checkpoint_dir, load_checkpoint
            ),
            rng=self._create_rng_config(seed=5678),
            peft=lora_peft,
        )

    def _setup_directories(self, base_dir, suffix=""):
        """Setup test directories."""
        pretrain_checkpoint_dir = os.path.join(base_dir, f"pretrain_checkpoints{suffix}")
        pretrain_tensorboard_dir = os.path.join(base_dir, f"pretrain_tensorboard{suffix}")
        lora_checkpoint_dir = os.path.join(base_dir, f"lora_checkpoints{suffix}")
        lora_tensorboard_dir = os.path.join(base_dir, f"lora_tensorboard{suffix}")

        if torch.distributed.get_rank() == 0:
            os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
            os.makedirs(pretrain_tensorboard_dir, exist_ok=True)
            os.makedirs(lora_checkpoint_dir, exist_ok=True)
            os.makedirs(lora_tensorboard_dir, exist_ok=True)

        return pretrain_checkpoint_dir, pretrain_tensorboard_dir, lora_checkpoint_dir, lora_tensorboard_dir
