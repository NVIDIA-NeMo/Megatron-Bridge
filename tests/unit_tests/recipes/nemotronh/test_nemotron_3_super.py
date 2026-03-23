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

"""
Unit tests for Nemotron 3 Super recipe configuration builders.

Tests cover:
- Pretrain configuration defaults (parameterless API)
- Finetune configuration (handles both full SFT and PEFT via peft parameter)
- MoE-specific settings (expert parallelism, MTP)
- Parallelism and tokenizer configurations
"""

import os
import tempfile

import pytest

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_finetune_config,
    nemotron_3_super_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotron3SuperPretrain:
    """Test cases for Nemotron 3 Super pretrain recipe.

    Note: Pretrain config uses the parameterless API and returns fixed defaults.
    Customization is done by modifying the returned ConfigContainer after creation.
    """

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config returns correct default configuration."""
        config = nemotron_3_super_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check expert parallelism defaults
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.expert_model_parallel_size == 8

        # Check training configuration
        assert config.train.train_iters == 39735
        assert config.train.global_batch_size == 3072
        assert config.train.micro_batch_size == 1

        # Check dataset configuration
        assert config.dataset.seq_length == 8192

        # Check tokenizer (HuggingFace for this recipe)
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

        # Check precision
        assert config.mixed_precision == "nemotron_3_super_bf16_with_nvfp4_mixed"

    def test_pretrain_config_moe_settings(self):
        """Test MoE settings for pretrain config."""
        config = nemotron_3_super_pretrain_config()

        # Verify MoE settings
        assert config.model.moe_token_dispatcher_type == "alltoall"
        assert config.model.moe_shared_expert_overlap is False
        assert config.model.moe_grouped_gemm is True
        assert config.model.moe_permute_fusion is True

    def test_pretrain_config_mtp_settings(self):
        """Test MTP settings for pretrain config."""
        config = nemotron_3_super_pretrain_config()

        # Verify MTP settings from provider
        assert config.model.mtp_num_layers == 2
        assert config.model.mtp_hybrid_override_pattern == "*E"

        # Verify MTP recipe-level settings
        # Note: the recipe overrides mtp_use_repeated_layer=True (provider default is False)
        assert config.model.mtp_use_repeated_layer is True
        assert config.model.keep_mtp_spec_in_bf16 is True
        assert config.model.calculate_per_token_loss is True
        assert config.model.mtp_loss_scaling_factor == 0.3

    def test_pretrain_config_optimizer_settings(self):
        """Test optimizer settings for pretrain config."""
        config = nemotron_3_super_pretrain_config()

        # Verify optimizer configuration
        assert config.optimizer.lr == 4.5e-4
        assert config.optimizer.weight_decay == 0.1
        assert config.optimizer.min_lr == 4.5e-6
        assert config.scheduler.lr_warmup_iters == 333

    def test_pretrain_config_checkpoint_settings(self):
        """Test checkpoint settings for pretrain config."""
        config = nemotron_3_super_pretrain_config()

        # Verify checkpoint configuration
        assert config.checkpoint.save_interval == 200
        assert config.checkpoint.ckpt_assume_constant_structure is True
        assert config.checkpoint.dist_ckpt_strictness == "log_all"


@pytest.mark.unit
class TestNemotron3SuperFinetune:
    """Test cases for Nemotron 3 Super finetune recipe.

    The finetune config handles both full SFT and PEFT (LoRA/DoRA) via the peft parameter.
    """

    def test_finetune_config_default_lora(self):
        """Test finetune_config with default LoRA configuration."""
        config = nemotron_3_super_finetune_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Check default parallelism for LoRA
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check expert parallelism (EP=1 for LoRA)
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.expert_model_parallel_size == 1

        # Check PEFT config exists for LoRA
        assert config.peft is not None

        # Check LoRA LR
        assert config.optimizer.lr == 1e-4

        # Check tokenizer
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

        # Check precision
        assert config.mixed_precision == "bf16_mixed"

    def test_finetune_config_full_sft(self):
        """Test finetune_config for full SFT (no PEFT)."""
        config = nemotron_3_super_finetune_config(peft=None)

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Check parallelism for full SFT
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check expert parallelism (EP=8 for full SFT)
        assert config.model.expert_model_parallel_size == 8

        # No PEFT config for full SFT
        assert config.peft is None

        # Full SFT should use lower LR
        assert config.optimizer.lr == 5e-6

    def test_finetune_config_dora(self):
        """Test finetune_config with DoRA configuration."""
        config = nemotron_3_super_finetune_config(peft="dora")

        assert config.peft is not None
        # DoRA should also use higher LR
        assert config.optimizer.lr == 1e-4

    def test_finetune_config_custom_parallelism(self):
        """Test finetune config with custom parallelism applied after creation."""
        config = nemotron_3_super_finetune_config()

        # Modify parallelism settings after creation
        config.model.tensor_model_parallel_size = 2
        config.model.pipeline_model_parallel_size = 2
        config.model.context_parallel_size = 2
        config.model.sequence_parallel = True
        config.model.expert_tensor_parallel_size = 2
        config.model.expert_model_parallel_size = 4

        assert config.model.tensor_model_parallel_size == 2
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.context_parallel_size == 2
        assert config.model.sequence_parallel is True
        assert config.model.expert_tensor_parallel_size == 2
        assert config.model.expert_model_parallel_size == 4

    def test_finetune_config_custom_training_params(self):
        """Test finetune config with custom training parameters applied after creation."""
        config = nemotron_3_super_finetune_config()

        # Modify training settings after creation
        config.train.train_iters = 500
        config.train.global_batch_size = 64
        config.train.micro_batch_size = 2
        config.optimizer.lr = 5e-5

        assert config.train.train_iters == 500
        assert config.train.global_batch_size == 64
        assert config.train.micro_batch_size == 2
        assert config.optimizer.lr == 5e-5

    def test_finetune_config_with_pretrained_checkpoint(self):
        """Test finetune config with pretrained checkpoint applied after creation."""
        config = nemotron_3_super_finetune_config()

        # Set checkpoint path after creation
        config.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

        assert config.checkpoint.pretrained_checkpoint == "/path/to/checkpoint"

    def test_finetune_config_with_custom_directory(self):
        """Test custom directory configuration for finetune."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = nemotron_3_super_finetune_config()

            # Set directory configuration after creation
            run_dir = os.path.join(temp_dir, "finetune_run")
            expected_checkpoint_dir = os.path.join(run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(run_dir, "tb_logs")

            config.checkpoint.save = expected_checkpoint_dir
            config.logger.tensorboard_dir = expected_tensorboard_dir

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed"])
    def test_finetune_precision_config(self, precision):
        """Test precision configuration for finetune."""
        config = nemotron_3_super_finetune_config()

        # Modify precision after creation
        config.mixed_precision = precision

        assert config.mixed_precision == precision


@pytest.mark.unit
class TestNemotron3SuperCommon:
    """Test cases common to all Nemotron 3 Super recipes."""

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_3_super_pretrain_config,
            nemotron_3_super_finetune_config,
        ],
    )
    def test_config_container_structure(self, recipe_fn):
        """Test that all configs return proper ConfigContainer with correct model provider."""
        config = recipe_fn()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Check required sections exist
        assert config.train is not None
        assert config.optimizer is not None
        assert config.scheduler is not None
        assert config.dataset is not None
        assert config.logger is not None
        assert config.tokenizer is not None
        assert config.checkpoint is not None
        assert config.rng is not None
        assert config.ddp is not None
        assert config.mixed_precision is not None

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_3_super_pretrain_config,
            nemotron_3_super_finetune_config,
        ],
    )
    def test_ddp_configuration(self, recipe_fn):
        """Test distributed data parallel configuration."""
        config = recipe_fn()

        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is True
        assert config.ddp.use_distributed_optimizer is True

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_3_super_pretrain_config,
            nemotron_3_super_finetune_config,
        ],
    )
    def test_moe_model_configuration(self, recipe_fn):
        """Test MoE-specific model configuration from provider."""
        config = recipe_fn()

        # Check MoE settings from AutoBridge provider
        assert config.model.num_moe_experts == 512
        assert config.model.moe_ffn_hidden_size == 2688
        assert config.model.moe_shared_expert_intermediate_size == 5376
        assert config.model.moe_router_topk == 22
        assert config.model.moe_router_topk_scaling_factor == 5.0
        assert config.model.moe_latent_size == 1024
