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

from megatron.hub.models.utils import forward_step
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
from megatron.hub.utils.config_utils import InstantiationMode


class TestLoRAE2E:
    """
    End-to-end LoRA fine-tuning test.
    """

    @pytest.mark.run_only_on("GPU")
    def test_lora_finetuning_e2e(self, tmp_path):
        """
        Test end-to-end LoRA fine-tuning with Llama3 8B.
        """
        pretrained_checkpoint_path = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/"
        checkpoint_dir = str(tmp_path / "checkpoints")
        tensorboard_dir = str(tmp_path / "tensorboard")

        try:
            # Create LoRA configuration
            cfg = self._create_lora_config(pretrained_checkpoint_path, checkpoint_dir, tensorboard_dir)
            finetune(config=cfg, forward_step_func=forward_step)

            # Verify checkpoint was created
            if torch.distributed.get_rank() == 0:
                latest_tracker_file = os.path.join(checkpoint_dir, "latest_train_state.pt")
                assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

                # Check for the final checkpoint directory
                final_iter_dir = os.path.join(checkpoint_dir, f"iter_{cfg.train.train_iters:07d}")
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

    def _create_lora_config(
        self, pretrained_checkpoint_path: str, save_dir: str, tensorboard_dir: str
    ) -> ConfigContainer:
        """Create LoRA configuration for Llama3 8B end-to-end test.

        Args:
            pretrained_checkpoint_path: Path to the pretrained Llama3 8B checkpoint
            save_dir: Directory where test checkpoints will be saved
            tensorboard_dir: Directory for tensorboard logs

        Returns:
            ConfigContainer with LoRA configuration for testing
        """
        # Load the original configuration from the checkpoint
        config_yaml_path = os.path.join(pretrained_checkpoint_path, "iter_0000000", "run_config.yaml")

        if not os.path.exists(config_yaml_path):
            pytest.skip(f"Pretrained checkpoint not found at: {pretrained_checkpoint_path}")

        cfg = ConfigContainer.from_yaml(config_yaml_path, mode=InstantiationMode.LENIENT)

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
            train_iters=20,  # Reduced for faster testing
            eval_iters=1,
            eval_interval=10,
        )

        # Checkpoint configuration
        cfg.checkpoint.save = save_dir
        cfg.checkpoint.load = None
        cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint_path
        cfg.checkpoint.save_interval = 10
        cfg.checkpoint.ckpt_format = "torch_dist"
        cfg.checkpoint.fully_parallel_save = True
        cfg.checkpoint.async_save = True

        # Model configuration for small GPU testing
        cfg.model.tensor_model_parallel_size = 1
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.sequence_parallel = False
        cfg.model.use_cpu_initialization = True
        cfg.model.cross_entropy_loss_fusion = False
        cfg.model.seq_length = 512  # Match dataset sequence length

        # Distributed configuration
        cfg.dist = DistributedInitConfig()

        # Logger configuration
        cfg.logger = LoggerConfig(
            log_interval=5,
            logging_level="INFO",
            tensorboard_dir=tensorboard_dir,
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
            sequence_length=512,  # Reduced sequence length for memory efficiency
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        )
        self._configure_tokenizer_for_checkpoint(cfg, pretrained_checkpoint_path)

        cfg.rng = RNGConfig(seed=1234, data_parallel_random_init=False)

        return cfg

    def _configure_tokenizer_for_checkpoint(self, cfg: ConfigContainer, checkpoint_path: str) -> None:
        """Configure the tokenizer appropriately based on the checkpoint structure."""
        # Check for HuggingFace assets directory
        hf_assets_path = os.path.join(checkpoint_path, "hf_assets")
        tokenizer_json_path = os.path.join(checkpoint_path, "tokenizer.json")
        tokenizer_model_path = os.path.join(checkpoint_path, "tokenizer.model")
        hf_tokenizer_json = os.path.join(hf_assets_path, "tokenizer.json")

        if os.path.exists(hf_assets_path) and os.path.exists(hf_tokenizer_json):
            cfg.tokenizer = TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_assets_path,
                vocab_size=128256,
            )
        elif os.path.exists(tokenizer_json_path):
            cfg.tokenizer = TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=checkpoint_path,
                vocab_size=128256,
            )
        elif os.path.exists(tokenizer_model_path):
            cfg.tokenizer = TokenizerConfig(
                tokenizer_type="SentencePieceTokenizer",
                tokenizer_model=tokenizer_model_path,
                vocab_size=128256,
            )
        else:
            cfg.tokenizer = TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model="meta-llama/Meta-Llama-3-8B",
                vocab_size=128256,
            )
