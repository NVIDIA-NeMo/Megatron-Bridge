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
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.models.llama import Llama3ModelProvider
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_merge import merge_lora
from megatron.bridge.training.checkpointing import (
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    MockGPTDatasetConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.initialize import destroy_global_state
from megatron.bridge.training.model_load_save import load_megatron_model
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.checkpoint_utils import file_exists
from megatron.bridge.utils.instantiate_utils import instantiate
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    get_directory_size,
    initialize_distributed,
    verify_checkpoint_files,
)


@dataclass
class Llama3ModelProvider145M(Llama3ModelProvider):
    """Smaller Llama3 model for testing."""

    rotary_base: int = 500_000
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    make_vocab_size_divisible_by: int = 128


class TestLoRAMerge:
    """
    Test LoRA merge functionality with comprehensive end-to-end verification.

    Tests the complete pipeline: pretrain -> LoRA finetune -> merge -> verify
    including checkpoint loading, config preservation, and size validation.
    """

    @pytest.mark.run_only_on("GPU")
    def test_lora_merge_end_to_end(self, tmp_path):
        """Test complete LoRA merge pipeline: pretrain -> finetune -> merge -> verify all aspects."""
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        (
            pretrain_checkpoint_dir,
            pretrain_tensorboard_dir,
            lora_checkpoint_dir,
            lora_tensorboard_dir,
            merged_checkpoint_dir,
        ) = self._setup_directories(shared_base_dir)

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

            # Merge LoRA checkpoint
            lora_final_checkpoint = os.path.join(lora_checkpoint_dir, f"iter_{lora_iters:07d}")
            merge_lora(lora_final_checkpoint, merged_checkpoint_dir)

            pretrain_final_checkpoint = os.path.join(pretrain_checkpoint_dir, f"iter_{pretrain_iters:07d}")

            # Merged checkpoint is saved as iteration 0
            merged_final_checkpoint = os.path.join(merged_checkpoint_dir, "iter_0000000")

            self._verify_merged_checkpoint_loading(merged_final_checkpoint)
            self._verify_config_preservation(pretrain_final_checkpoint, lora_final_checkpoint, merged_final_checkpoint)
            self._verify_checkpoint_sizes(pretrain_final_checkpoint, lora_final_checkpoint, merged_final_checkpoint)

        finally:
            clear_directories(shared_base_dir)
            destroy_global_state()

    def _verify_merged_checkpoint_loading(self, merged_checkpoint_dir: str) -> None:
        """Verify that the merged checkpoint can be loaded successfully."""
        merged_model = load_megatron_model(
            checkpoint_path=merged_checkpoint_dir, use_cpu_init=True, return_state_dict=False
        )
        assert merged_model is not None, "Failed to load merged checkpoint"

    def _verify_config_preservation(
        self, original_pretrain_dir: str, lora_checkpoint_dir: str, merged_checkpoint_dir: str
    ) -> None:
        """Verify that all configuration is preserved except PEFT and model architecture matches original."""

        # Load all configs
        lora_run_config = read_run_config(get_checkpoint_run_config_filename(lora_checkpoint_dir))
        merged_run_config = read_run_config(get_checkpoint_run_config_filename(merged_checkpoint_dir))

        # Merged checkpoint config has no PEFT
        assert file_exists(get_checkpoint_run_config_filename(merged_checkpoint_dir)), (
            "Merged checkpoint missing run_config.yaml"
        )
        assert "peft" not in merged_run_config or merged_run_config["peft"] is None, (
            "Merged checkpoint should not contain PEFT configuration"
        )

        # All non-PEFT, non-checkpoint configs should be preserved from LoRA checkpoint
        preserve_configs = (
            "model",
            "train",
            "optimizer",
            "scheduler",
            "dataset",
            "logger",
            "tokenizer",
            "ddp",
            "dist",
            "rng",
        )

        for config_key in preserve_configs:
            if config_key in lora_run_config:
                assert config_key in merged_run_config, f"Missing preserved config: {config_key}"
                # For complex comparison, just check they can be instantiated the same way
                try:
                    lora_cfg_obj = instantiate(lora_run_config[config_key])
                    merged_cfg_obj = instantiate(merged_run_config[config_key])
                    assert type(lora_cfg_obj) == type(merged_cfg_obj), f"Config type mismatch for {config_key}"
                except Exception as e:
                    pytest.fail(f"Failed to instantiate preserved config {config_key}: {e}")

        # Checkpoint config should be updated appropriately
        merged_ckpt_cfg = merged_run_config["checkpoint"]
        assert merged_ckpt_cfg.get("pretrained_checkpoint") is None, (
            "pretrained_checkpoint should be None in merged config"
        )

    def _verify_checkpoint_sizes(
        self, original_pretrain_dir: str, lora_checkpoint_dir: str, merged_checkpoint_dir: str
    ) -> None:
        """Verify checkpoint sizes are as expected after merge."""

        # Get checkpoint sizes
        pretrain_size = get_directory_size(original_pretrain_dir)
        lora_size = get_directory_size(lora_checkpoint_dir)
        merged_size = get_directory_size(merged_checkpoint_dir)

        # LoRA checkpoint should be significantly smaller than pretrain (adapters vs full model)
        lora_to_pretrain_ratio = lora_size / pretrain_size
        assert lora_to_pretrain_ratio < 0.2, (
            f"LoRA checkpoint should be <20% of pretrain size, got {lora_to_pretrain_ratio:.1%}"
        )

        # Merged checkpoint should be similar size to pretrain (both contain only model weights)
        size_ratio = merged_size / pretrain_size
        assert 0.95 <= size_ratio <= 1.05, (
            f"Merged checkpoint should be 95-105% of pretrain size (both model weights only), got {size_ratio:.1%}"
        )

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

    def _create_squad_dataset_config(self, seq_length, seed=5678):
        """Create a SQuAD dataset configuration."""
        return HFDatasetConfig(
            dataset_name="squad",
            process_example_fn=process_squad_example,
            seq_length=seq_length,
            seed=seed,
            dataloader_type="single",
            num_workers=1,
            do_validation=False,
            do_test=False,
            val_proportion=None,
            rewrite=False,
        )

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
            save_optim=False,
            save_rng=False,
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

    def _create_pretrain_config(self, train_iters, checkpoint_dir, tensorboard_dir, seq_length=512):
        """Create complete pretrain configuration with model."""
        model = self._create_model_provider(seq_length)

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
        self, train_iters, checkpoint_dir, tensorboard_dir, pretrained_checkpoint_dir, seq_length=512
    ):
        """Create complete LoRA finetuning configuration with model and PEFT."""
        model = self._create_model_provider(seq_length)
        lora_peft = self._create_lora_peft()

        return ConfigContainer(
            model=model,
            train=self._create_training_config(train_iters),
            optimizer=self._create_optimizer_config(lr=1e-4),  # Lower LR for finetuning
            scheduler=self._create_scheduler_config(train_iters),
            ddp=self._create_ddp_config(),
            dataset=self._create_squad_dataset_config(seq_length),
            logger=self._create_logger_config(tensorboard_dir),
            tokenizer=self._create_finetune_tokenizer_config(),
            checkpoint=self._create_checkpoint_config(train_iters, checkpoint_dir, pretrained_checkpoint_dir),
            rng=self._create_rng_config(seed=5678),
            peft=lora_peft,
        )

    def _setup_directories(self, base_dir, suffix=""):
        """Setup test directories."""
        pretrain_checkpoint_dir = os.path.join(base_dir, f"pretrain_checkpoints{suffix}")
        pretrain_tensorboard_dir = os.path.join(base_dir, f"pretrain_tensorboard{suffix}")
        lora_checkpoint_dir = os.path.join(base_dir, f"lora_checkpoints{suffix}")
        lora_tensorboard_dir = os.path.join(base_dir, f"lora_tensorboard{suffix}")
        merged_checkpoint_dir = os.path.join(base_dir, f"merged_checkpoint{suffix}")

        if torch.distributed.get_rank() == 0:
            os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
            os.makedirs(pretrain_tensorboard_dir, exist_ok=True)
            os.makedirs(lora_checkpoint_dir, exist_ok=True)
            os.makedirs(lora_tensorboard_dir, exist_ok=True)
            os.makedirs(merged_checkpoint_dir, exist_ok=True)

        return (
            pretrain_checkpoint_dir,
            pretrain_tensorboard_dir,
            lora_checkpoint_dir,
            lora_tensorboard_dir,
            merged_checkpoint_dir,
        )
