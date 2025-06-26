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

import logging
import os
import re
import shutil

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.hub.core.utils.common_utils import print_rank_0
from megatron.hub.peft.lora import LoRA
from megatron.hub.training.config import (
    ConfigContainer,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.hub.training.finetune import finetune
from megatron.hub.training.gpt_step import forward_step


logger: logging.Logger = logging.getLogger(__name__)


class TestLoRAE2E:
    """
    End-to-end LoRA fine-tuning test.
    """

    @pytest.mark.run_only_on("GPU")
    def test_lora_finetuning_e2e(self, tmp_path):
        """
        Test end-to-end LoRA fine-tuning with checkpoint resume functionality.

        This test verifies the complete PEFT training workflow:
        1. Initial training phase: Train for initial steps and save checkpoint
        2. Resume phase: Resume from training checkpoint and complete training
        3. Verification: Ensure final training checkpoint maintains size reduction by saving adapter states only
        """
        pretrained_checkpoint_path = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/"
        checkpoint_dir = str(tmp_path / "checkpoints")

        try:
            initial_cfg = self._create_lora_config(
                pretrained_checkpoint_path,
                checkpoint_dir,
                train_iters=10,  # Train for 10 iterations initially
                save_interval=10,  # Save checkpoint after 10 iterations
            )

            finetune(config=initial_cfg, forward_step_func=forward_step)

            torch.distributed.barrier()

            # Verify initial checkpoint was created
            initial_checkpoint_dir = os.path.join(checkpoint_dir, "iter_0000010")

            if torch.distributed.get_rank() == 0:
                assert os.path.exists(initial_checkpoint_dir), (
                    f"Initial checkpoint not found at {initial_checkpoint_dir}"
                )
                metadata_file = os.path.join(initial_checkpoint_dir, ".metadata")
                assert os.path.exists(metadata_file), "Initial checkpoint metadata file not found"

            resume_cfg = self._create_lora_config(
                pretrained_checkpoint_path,
                checkpoint_dir,
                train_iters=20,  # Total of 20 iterations (10 more)
                save_interval=10,
                load_checkpoint=initial_checkpoint_dir,  # Resume from initial checkpoint
            )

            finetune(config=resume_cfg, forward_step_func=forward_step)

            if torch.distributed.get_rank() == 0:
                # Verify final checkpoint was created
                latest_tracker_file = os.path.join(checkpoint_dir, "latest_train_state.pt")
                assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

                final_iter_dir = os.path.join(checkpoint_dir, f"iter_{resume_cfg.train.train_iters:07d}")
                assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"

                metadata_file = os.path.join(final_iter_dir, ".metadata")
                assert os.path.exists(metadata_file), "Final checkpoint metadata file not found"

                # Verify checkpoint size is significantly smaller than base model
                self._verify_checkpoint_size_reduction(
                    pretrained_checkpoint_path, final_iter_dir, expected_reduction_factor=0.2
                )
                # Additional verification: Compare initial and final checkpoints
                self._verify_checkpoint_progression(initial_checkpoint_dir, final_iter_dir)

        finally:
            # pytest's tmp_path fixture doesn't clean up immediately.
            # Clean up manually.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    if os.path.exists(checkpoint_dir):
                        shutil.rmtree(checkpoint_dir)
                torch.distributed.barrier()

    def _create_lora_config(
        self,
        pretrained_checkpoint_path: str,
        save_dir: str,
        train_iters: int = 20,
        save_interval: int = 10,
        load_checkpoint: str | None = None,
    ) -> ConfigContainer:
        """Create LoRA configuration for Llama3 8B end-to-end test.

        Args:
            pretrained_checkpoint_path: Path to the pretrained Llama3 8B checkpoint
            save_dir: Directory where test checkpoints will be saved
            tensorboard_dir: Directory for tensorboard logs
            train_iters: Number of training iterations (default: 20)
            save_interval: Interval between checkpoints (default: 10)
            load_checkpoint: Path to checkpoint to load from (default: None)

        Returns:
            ConfigContainer with LoRA configuration for testing
        """
        # Load the original configuration from the checkpoint
        config_yaml_path = os.path.join(pretrained_checkpoint_path, "iter_0000000", "run_config.yaml")

        assert os.path.exists(config_yaml_path), f"Pretrained checkpoint not found at: {pretrained_checkpoint_path}"
        cfg = ConfigContainer.from_yaml(config_yaml_path)

        seq_length = 512

        # LoRA configuration
        lora_config = LoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            dim=8,
            alpha=16,
            dropout=0.1,
        )
        cfg.peft = lora_config

        # Training configuration for end-to-end test
        cfg.train = TrainingConfig(
            micro_batch_size=1,  # Reduced for memory efficiency
            global_batch_size=2,  # Must be divisible by (micro_batch_size * data_parallel_size)
            train_iters=train_iters,
            eval_iters=1,
            eval_interval=10,
        )

        # Checkpoint configuration
        cfg.checkpoint.save = save_dir
        cfg.checkpoint.load = load_checkpoint
        cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint_path
        cfg.checkpoint.save_interval = save_interval
        cfg.checkpoint.ckpt_format = "torch_dist"
        cfg.checkpoint.fully_parallel_save = True
        cfg.checkpoint.async_save = True

        # Model configuration for small GPU testing
        cfg.model.tensor_model_parallel_size = 1
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.sequence_parallel = False
        cfg.model.use_cpu_initialization = True
        cfg.model.cross_entropy_loss_fusion = False
        cfg.model.seq_length = seq_length

        # Distributed configuration
        cfg.dist = DistributedInitConfig()

        # Logger configuration
        cfg.logger = LoggerConfig(
            log_interval=5,
            logging_level="INFO",
        )

        # Optimizer configuration for LoRA fine-tuning
        cfg.optimizer = OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            use_distributed_optimizer=False,
            clip_grad=1.0,
            lr=1e-4,
            weight_decay=0.01,
            min_lr=0.1 * 1e-4,
        )

        # Scheduler configuration
        cfg.scheduler = SchedulerConfig(
            start_weight_decay=0.01,
            end_weight_decay=0.01,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_decay_iters=20,
            lr_warmup_iters=2,
            lr_warmup_init=0.0,
            override_opt_param_scheduler=True,
        )

        # DDP configuration
        cfg.ddp = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            use_distributed_optimizer=False,
        )
        cfg.dataset = GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        )
        tokenizer_path = os.path.join(pretrained_checkpoint_path, "hf_assets")
        assert os.path.exists(tokenizer_path), "Tokenizer assets not found"
        cfg.tokenizer = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=tokenizer_path,
        )
        cfg.rng = RNGConfig(seed=1234, data_parallel_random_init=False)

        return cfg

    def _verify_checkpoint_size_reduction(
        self, pretrained_path: str, adapter_checkpoint_path: str, expected_reduction_factor: float = 0.1
    ) -> None:
        """Verify that adapter checkpoint is significantly smaller than base model.

        Args:
            pretrained_path: Path to the pretrained base model checkpoint
            adapter_checkpoint_path: Path to the adapter checkpoint
            expected_reduction_factor: Maximum ratio of adapter size to base model size
        """
        # Get size of base model checkpoint
        base_model_size = get_directory_size(pretrained_path)

        # Get size of adapter checkpoint
        adapter_size = get_directory_size(adapter_checkpoint_path)

        # Calculate size reduction ratio
        size_ratio = adapter_size / base_model_size if base_model_size > 0 else 1.0

        logger.debug(f"Base model checkpoint size: {base_model_size / (1024**3):.2f} GB")
        logger.debug(f"Adapter checkpoint size: {adapter_size / (1024**3):.2f} GB")
        logger.debug(f"Size reduction ratio: {size_ratio:.4f}")
        logger.debug(f"Size reduction: {(1 - size_ratio) * 100:.1f}%")

        # Verify significant size reduction
        assert size_ratio < expected_reduction_factor, (
            f"Adapter checkpoint size ({adapter_size / (1024**3):.2f} GB) is not significantly smaller "
            f"than base model ({base_model_size / (1024**3):.2f} GB). "
            f"Expected ratio < {expected_reduction_factor}, got {size_ratio:.4f}"
        )

        # Ensure adapter checkpoint is not empty
        assert adapter_size > 0, "Adapter checkpoint is empty"

        if os.path.exists(os.path.join(adapter_checkpoint_path, ".metadata")):
            # Check that main model state files are much smaller in adapter checkpoint
            model_files = []
            for root, dirs, files in os.walk(adapter_checkpoint_path):
                for file in files:
                    if re.match(
                        r"__\d+_\d+\.distcp$", file
                    ):  # Distributed checkpoint pattern: __{global_rank}_{writer_rank}.distcp
                        model_files.append(os.path.join(root, file))

            if model_files:
                model_files_size = sum(os.path.getsize(f) for f in model_files)
                model_ratio = model_files_size / base_model_size if base_model_size > 0 else 1.0

                logger.debug(f"Model state files size in adapter checkpoint: {model_files_size / (1024**2):.2f} MB")
                logger.debug(f"Model files size ratio: {model_ratio:.6f}")

                # Model state files should be even smaller since they only contain adapters
                assert model_ratio < expected_reduction_factor / 2, (
                    f"Model state files in adapter checkpoint are larger than expected. "
                    f"Expected ratio < {expected_reduction_factor / 2:.6f}, got {model_ratio:.6f}"
                )

    def _verify_checkpoint_progression(self, initial_checkpoint_dir: str, final_checkpoint_dir: str) -> None:
        """Verify that the final checkpoint is a proper continuation of the initial checkpoint."""
        # Both checkpoints should exist
        assert os.path.exists(initial_checkpoint_dir), (
            f"Initial checkpoint directory not found: {initial_checkpoint_dir}"
        )
        assert os.path.exists(final_checkpoint_dir), f"Final checkpoint directory not found: {final_checkpoint_dir}"

        # Both should be distributed checkpoints with metadata
        initial_metadata = os.path.join(initial_checkpoint_dir, ".metadata")
        final_metadata = os.path.join(final_checkpoint_dir, ".metadata")

        assert os.path.exists(initial_metadata), "Initial checkpoint metadata not found"
        assert os.path.exists(final_metadata), "Final checkpoint metadata not found"

        # Both checkpoints should be similar in size (both are adapter-only checkpoints)
        initial_size = get_directory_size(initial_checkpoint_dir)
        final_size = get_directory_size(final_checkpoint_dir)

        logger.debug(f"Initial checkpoint size: {initial_size / (1024**2):.2f} MB")
        logger.debug(f"Final checkpoint size: {final_size / (1024**2):.2f} MB")

        # Size should be similar (within reasonable bounds) since both are adapter-only
        size_ratio = (
            abs(initial_size - final_size) / max(initial_size, final_size) if max(initial_size, final_size) > 0 else 0
        )
        assert size_ratio < 0.5, (
            f"Checkpoint sizes too different: initial={initial_size}, final={final_size}, ratio={size_ratio:.3f}"
        )

        print_rank_0("Checkpoint progression verification completed successfully")


def get_directory_size(path: str) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, IOError):
                pass  # Skip files that can't be accessed
    return total_size
