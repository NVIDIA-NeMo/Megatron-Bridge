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
from pathlib import Path

import pytest
import torch

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


class TestAutoBridgeFinetune:
    """
    Test finetuning using AutoBridge-imported HuggingFace checkpoints.

    This demonstrates the import-first workflow:
    1. HF model imported once to Megatron format using:
       python examples/conversion/convert_checkpoints.py import \
           --hf-model "meta-llama/Llama-3.2-1B" \
           --megatron-path "/path/to/TestData/megatron_bridge/checkpoints/llama-32-1b-hf-to-megatron-import" \
           --torch-dtype bfloat16 \
           --device-map auto
    2. Tokenizer saved locally using:
       huggingface-cli download meta-llama/Llama-3.2-1B \
           --include "tokenizer*" \
           --local-dir /path/to/TestData/megatron_bridge/tokenizers/llama-32-1b
    3. Training uses standard checkpoint configuration
    """

    # Path to the imported Llama 3.2 1B checkpoint
    IMPORTED_CHECKPOINT = "/home/TestData/megatron_bridge/checkpoints/llama-32-1b-hf-to-megatron-import"
    # Path to the saved tokenizer assets
    TOKENIZER_PATH = "/home/TestData/megatron_bridge/tokenizers/llama-32-1b"

    def setup_method(self):
        """Setup before each test method."""
        # Clear GPU memory before each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Clear GPU memory after each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @pytest.mark.run_only_on("GPU")
    def test_imported_hf_sft(self, tmp_path):
        """Test supervised finetuning using imported HuggingFace checkpoint."""
        # Skip if imported checkpoint or tokenizer doesn't exist
        if not Path(self.IMPORTED_CHECKPOINT).exists():
            pytest.skip(f"Imported checkpoint not found at {self.IMPORTED_CHECKPOINT}")
        if not Path(self.TOKENIZER_PATH).exists():
            pytest.skip(f"Tokenizer not found at {self.TOKENIZER_PATH}")

        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        finetune_checkpoint_dir, finetune_tensorboard_dir = self._setup_directories(shared_base_dir, "sft")

        torch.distributed.barrier()

        try:
            seq_length = 512
            finetune_iters = 10

            # Create SFT config using imported checkpoint
            sft_cfg = self._create_sft_config(
                finetune_iters, finetune_checkpoint_dir, finetune_tensorboard_dir, self.IMPORTED_CHECKPOINT, seq_length
            )

            # Run finetuning - no pretraining needed!
            finetune(sft_cfg, forward_step)
            verify_checkpoint_files(finetune_checkpoint_dir, finetune_iters)

        finally:
            clear_directories(shared_base_dir)

    @pytest.mark.run_only_on("GPU")
    def test_imported_hf_lora(self, tmp_path):
        """Test LoRA finetuning using imported HuggingFace checkpoint."""
        # Skip if imported checkpoint or tokenizer doesn't exist
        if not Path(self.IMPORTED_CHECKPOINT).exists():
            pytest.skip(f"Imported checkpoint not found at {self.IMPORTED_CHECKPOINT}")
        if not Path(self.TOKENIZER_PATH).exists():
            pytest.skip(f"Tokenizer not found at {self.TOKENIZER_PATH}")

        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        lora_checkpoint_dir, lora_tensorboard_dir = self._setup_directories(shared_base_dir, "lora")

        torch.distributed.barrier()

        try:
            seq_length = 512
            lora_iters = 10

            # Create LoRA config using imported checkpoint
            lora_cfg = self._create_lora_config(
                lora_iters, lora_checkpoint_dir, lora_tensorboard_dir, self.IMPORTED_CHECKPOINT, seq_length
            )

            # Run LoRA finetuning - no pretraining needed!
            finetune(lora_cfg, forward_step)
            verify_checkpoint_files(lora_checkpoint_dir, lora_iters)

        finally:
            clear_directories(shared_base_dir)

    @pytest.mark.run_only_on("GPU")
    def test_imported_hf_lora_resume(self, tmp_path):
        """Test LoRA finetuning with save and resume using imported checkpoint."""
        # Skip if imported checkpoint or tokenizer doesn't exist
        if not Path(self.IMPORTED_CHECKPOINT).exists():
            pytest.skip(f"Imported checkpoint not found at {self.IMPORTED_CHECKPOINT}")
        if not Path(self.TOKENIZER_PATH).exists():
            pytest.skip(f"Tokenizer not found at {self.TOKENIZER_PATH}")

        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        lora_checkpoint_dir, lora_tensorboard_dir = self._setup_directories(shared_base_dir, "lora_resume")

        torch.distributed.barrier()

        try:
            seq_length = 512
            initial_lora_iters = 6
            total_lora_iters = 10

            # Initial LoRA training phase
            lora_initial_cfg = self._create_lora_config(
                initial_lora_iters,
                lora_checkpoint_dir,
                lora_tensorboard_dir,
                self.IMPORTED_CHECKPOINT,
                seq_length,
                scheduler_total_iters=total_lora_iters,
            )

            # Run initial LoRA finetuning
            finetune(lora_initial_cfg, forward_step)
            verify_checkpoint_files(lora_checkpoint_dir, initial_lora_iters)

            # Resume LoRA training
            lora_resume_cfg = self._create_lora_config(
                total_lora_iters,
                lora_checkpoint_dir,
                lora_tensorboard_dir,
                self.IMPORTED_CHECKPOINT,
                seq_length,
                load_checkpoint=lora_checkpoint_dir,
                scheduler_total_iters=total_lora_iters,
            )
            lora_resume_cfg.checkpoint.save_interval = total_lora_iters - initial_lora_iters
            lora_resume_cfg.scheduler.use_checkpoint_opt_param_scheduler = True

            # Run resumed LoRA finetuning
            finetune(lora_resume_cfg, forward_step)
            verify_checkpoint_files(lora_checkpoint_dir, total_lora_iters)

        finally:
            clear_directories(shared_base_dir)

    def _create_model_provider(self, seq_length=512, tensor_parallel_size=1, pipeline_parallel_size=1):
        """Create Llama 3.2 1B model provider matching imported checkpoint."""
        return Llama32ModelProvider1B(
            seq_length=seq_length,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            pipeline_dtype=torch.bfloat16,
            sequence_parallel=(tensor_parallel_size > 1),
        )

    def _create_training_config(self, train_iters, global_batch_size=8, micro_batch_size=1):
        """Create training configuration."""
        return TrainingConfig(
            train_iters=train_iters,
            eval_interval=5,
            eval_iters=0,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        )

    def _create_optimizer_config(self, lr=1e-4):
        """Create optimizer configuration with lower LR for finetuning."""
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
            min_lr=1e-6,
        )

    def _create_scheduler_config(self, total_iters):
        """Create scheduler configuration."""
        return SchedulerConfig(
            start_weight_decay=0.1,
            end_weight_decay=0.1,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=1,
            lr_warmup_init=0.0,
            lr_decay_iters=total_iters,
        )

    def _create_ddp_config(self):
        """Create DDP configuration."""
        return DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        )

    def _create_squad_dataset_config(self, seq_length, seed=5678):
        """Create SQuAD dataset configuration for finetuning."""
        return HFDatasetConfig(
            dataset_name="squad",
            process_example_fn=process_squad_example,
            seq_length=seq_length,
            seed=seed,
            dataloader_type="single",
            num_workers=1,
            do_validation=False,
            do_test=False,
            rewrite=False,
        )

    def _create_tokenizer_config(self):
        """Create tokenizer configuration using locally saved tokenizer."""
        return TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=self.TOKENIZER_PATH,
        )

    def _create_logger_config(self, tensorboard_dir):
        """Create logger configuration."""
        return LoggerConfig(
            log_interval=1,
            tensorboard_dir=tensorboard_dir,
        )

    def _create_checkpoint_config(self, save_interval, save_dir, pretrained_checkpoint, load_dir=None):
        """Create checkpoint configuration."""
        return CheckpointConfig(
            save_interval=save_interval,
            save=save_dir,
            pretrained_checkpoint=pretrained_checkpoint,  # Uses imported checkpoint
            load=load_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        )

    def _create_rng_config(self, seed=1234):
        """Create RNG configuration."""
        return RNGConfig(seed=seed)

    def _create_lora_peft(self, dim=16, alpha=32, dropout=0.1):
        """Create LoRA PEFT configuration."""
        return LoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            dim=dim,
            alpha=alpha,
            dropout=dropout,
        )

    def _create_sft_config(
        self,
        train_iters,
        checkpoint_dir,
        tensorboard_dir,
        pretrained_checkpoint,
        seq_length=512,
    ):
        """Create SFT configuration using imported checkpoint."""
        model = self._create_model_provider(seq_length)

        return ConfigContainer(
            model=model,
            train=self._create_training_config(train_iters),
            optimizer=self._create_optimizer_config(),
            scheduler=self._create_scheduler_config(train_iters),
            ddp=self._create_ddp_config(),
            dataset=self._create_squad_dataset_config(seq_length),
            logger=self._create_logger_config(tensorboard_dir),
            tokenizer=self._create_tokenizer_config(),
            checkpoint=self._create_checkpoint_config(train_iters, checkpoint_dir, pretrained_checkpoint),
            rng=self._create_rng_config(),
        )

    def _create_lora_config(
        self,
        train_iters,
        checkpoint_dir,
        tensorboard_dir,
        pretrained_checkpoint,
        seq_length=512,
        load_checkpoint=None,
        scheduler_total_iters=None,
    ):
        """Create LoRA configuration using imported checkpoint."""
        model = self._create_model_provider(seq_length)
        lora_peft = self._create_lora_peft()

        scheduler_iters = scheduler_total_iters if scheduler_total_iters is not None else train_iters

        return ConfigContainer(
            model=model,
            train=self._create_training_config(train_iters),
            optimizer=self._create_optimizer_config(),
            scheduler=self._create_scheduler_config(scheduler_iters),
            ddp=self._create_ddp_config(),
            dataset=self._create_squad_dataset_config(seq_length),
            logger=self._create_logger_config(tensorboard_dir),
            tokenizer=self._create_tokenizer_config(),
            checkpoint=self._create_checkpoint_config(
                train_iters, checkpoint_dir, pretrained_checkpoint, load_checkpoint
            ),
            rng=self._create_rng_config(seed=5678),
            peft=lora_peft,
        )

    def _setup_directories(self, base_dir, test_type):
        """Setup test directories."""
        checkpoint_dir = os.path.join(base_dir, f"{test_type}_checkpoints")
        tensorboard_dir = os.path.join(base_dir, f"{test_type}_tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        return checkpoint_dir, tensorboard_dir
