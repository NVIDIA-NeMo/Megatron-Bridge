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
Unit tests for Nemotron 3 Nano recipe configuration builders.

Tests cover:
- Separate standard and MTP pretrain configurations
- SFT configuration with Hugging Face-derived model architecture
- PEFT configuration with Hugging Face-derived model architecture, LoRA, and DoRA
- MoE-specific settings (DeepEP, expert parallelism)
- Parallelism and tokenizer configurations
"""

import os
import tempfile
from inspect import signature
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.recipes.nemotronh.h100 import nemotron_3_nano as recipe_module
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_mtp_peft_config,
    nemotron_3_nano_mtp_pretrain_config,
    nemotron_3_nano_mtp_sft_config,
    nemotron_3_nano_peft_config,
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotron3NanoPretrain:
    """Test cases for Nemotron 3 Nano pretrain recipe.

    Most customization is done by modifying the returned ConfigContainer after
    creation; MTP uses a separate recipe.
    """

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config returns correct default configuration."""
        config = nemotron_3_nano_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)

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
        assert config.train.micro_batch_size == 2

        # Check dataset configuration
        assert config.dataset.seq_length == 8192

        # Check tokenizer (HuggingFace for this recipe)
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

        # Check comm overlap
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True
        assert config.comm_overlap.tp_comm_bootstrap_backend == "nccl"

        # Check precision
        assert config.mixed_precision == "bf16_mixed"

        # MTP is opt-in for pretraining.
        assert config.model.mtp_num_layers == 0
        assert config.model.mtp_hybrid_override_pattern is None

    def test_mtp_pretrain_config(self):
        """The dedicated MTP recipe enables the repeated Nano MTP head."""
        base_config = nemotron_3_nano_pretrain_config()
        config = nemotron_3_nano_mtp_pretrain_config()

        assert config.model.mtp_num_layers == 2
        assert config.model.mtp_hybrid_override_pattern == "*E"
        assert config.model.mtp_use_repeated_layer is True
        assert config.model.keep_mtp_spec_in_bf16 is True
        assert config.model.mtp_loss_scaling_factor == 0.3
        assert config.model.calculate_per_token_loss == base_config.model.calculate_per_token_loss
        assert config.model.use_te_rng_tracker == base_config.model.use_te_rng_tracker
        assert config.tokenizer.tokenizer_model == recipe_module._NEMOTRON_3_NANO_MTP_MODEL_ID

    def test_pretrain_recipes_do_not_expose_mtp_flag(self):
        """Standard and MTP pretraining use distinct parameterless factories."""
        assert "enable_mtp" not in signature(nemotron_3_nano_pretrain_config).parameters
        assert "enable_mtp" not in signature(nemotron_3_nano_mtp_pretrain_config).parameters

    def test_pretrain_config_deepep_enabled(self):
        """Test that DeepEP is enabled by default for MoE pretrain."""
        config = nemotron_3_nano_pretrain_config()

        # DeepEP should be enabled by default - check MoE dispatcher settings
        assert config.model.moe_token_dispatcher_type == "flex"
        assert config.model.moe_shared_expert_overlap is False
        assert config.model.moe_flex_dispatcher_backend == "deepep"

    def test_pretrain_config_moe_kernel_settings(self):
        """Test MoE kernel settings for pretrain config."""
        config = nemotron_3_nano_pretrain_config()

        # Verify MoE kernel selections
        assert config.model.attention_backend == "fused"
        assert config.model.moe_router_fusion is False
        assert config.model.moe_permute_fusion is True
        assert config.model.moe_grouped_gemm is True
        assert config.model.cross_entropy_loss_fusion is True
        assert config.model.cross_entropy_fusion_impl == "native"

    def test_pretrain_config_optimizer_settings(self):
        """Test optimizer settings for pretrain config."""
        config = nemotron_3_nano_pretrain_config()

        # Verify optimizer configuration
        assert config.optimizer.lr == 1.6e-3
        assert config.optimizer.weight_decay == 0.1
        assert config.optimizer.min_lr == 1.6e-5
        assert config.scheduler.lr_warmup_iters == 333

        # Verify precision settings
        assert config.optimizer.use_precision_aware_optimizer is False
        assert config.optimizer.main_grads_dtype is not None
        assert config.optimizer.main_params_dtype is not None

    def test_pretrain_config_checkpoint_settings(self):
        """Test checkpoint settings for pretrain config."""
        config = nemotron_3_nano_pretrain_config()

        # Verify checkpoint configuration
        assert config.checkpoint.save_interval == 200
        assert config.checkpoint.ckpt_assume_constant_structure is True
        assert config.checkpoint.dist_ckpt_strictness == "log_all"


@pytest.mark.unit
class TestNemotron3NanoSft:
    """Test cases for Nemotron 3 Nano SFT recipe.

    Standard and MTP recipes derive their model architecture from their
    respective hard-coded Hugging Face repositories.
    """

    def test_sft_config_default_parameters(self):
        """Test sft_config returns correct default configuration."""
        config = nemotron_3_nano_sft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)

        # Check default parallelism for SFT
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is False

        # Check expert parallelism
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.expert_model_parallel_size == 8

        # No PEFT config for full SFT
        assert config.peft is None

        # Full SFT should use lower LR
        assert config.optimizer.lr == 5e-6

        # Check tokenizer
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

        # Check precision
        assert config.mixed_precision == "bf16_mixed"

    def test_sft_config_deepep_settings(self):
        """Test that SFT config has correct DeepEP/MoE dispatcher settings."""
        config = nemotron_3_nano_sft_config()

        # Check MoE dispatcher settings
        assert config.model.moe_token_dispatcher_type is not None

    def test_sft_config_custom_parallelism(self):
        """Test SFT config with custom parallelism applied after creation."""
        config = nemotron_3_nano_sft_config()

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

    def test_sft_config_custom_training_params(self):
        """Test SFT config with custom training parameters applied after creation."""
        config = nemotron_3_nano_sft_config()

        # Modify training settings after creation
        config.train.train_iters = 500
        config.train.global_batch_size = 64
        config.train.micro_batch_size = 2
        config.optimizer.lr = 5e-5
        config.optimizer.min_lr = 1e-6

        assert config.train.train_iters == 500
        assert config.train.global_batch_size == 64
        assert config.train.micro_batch_size == 2
        assert config.optimizer.lr == 5e-5
        assert config.optimizer.min_lr == 1e-6

    def test_sft_config_with_pretrained_checkpoint(self):
        """Test SFT config with pretrained checkpoint applied after creation."""
        config = nemotron_3_nano_sft_config()

        # Set checkpoint path after creation
        config.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

        assert config.checkpoint.pretrained_checkpoint == "/path/to/checkpoint"

    @pytest.mark.parametrize(
        ("recipe_factory", "model_id", "mtp_num_layers"),
        [
            (nemotron_3_nano_sft_config, recipe_module._NEMOTRON_3_NANO_MODEL_ID, 0),
            (nemotron_3_nano_peft_config, recipe_module._NEMOTRON_3_NANO_MODEL_ID, 0),
            (nemotron_3_nano_mtp_sft_config, recipe_module._NEMOTRON_3_NANO_MTP_MODEL_ID, 2),
            (nemotron_3_nano_mtp_peft_config, recipe_module._NEMOTRON_3_NANO_MTP_MODEL_ID, 2),
        ],
    )
    def test_finetuning_derives_mtp_from_hf_model(self, recipe_factory, model_id, mtp_num_layers):
        """SFT and PEFT preserve the MTP architecture selected by AutoBridge."""
        provider = HybridModelProvider(
            mtp_num_layers=mtp_num_layers,
            mtp_hybrid_override_pattern="*E" if mtp_num_layers else None,
            mtp_use_repeated_layer=bool(mtp_num_layers),
            keep_mtp_spec_in_bf16=bool(mtp_num_layers),
            mtp_loss_scaling_factor=0.3 if mtp_num_layers else 0.1,
        )
        bridge = Mock()
        bridge.to_megatron_provider.return_value = provider

        with patch.object(recipe_module.AutoBridge, "from_hf_pretrained", return_value=bridge) as from_hf:
            config = recipe_factory()

        from_hf.assert_called_once_with(model_id)
        bridge.to_megatron_provider.assert_called_once_with(load_weights=False)
        assert config.model is provider
        assert config.model.mtp_num_layers == mtp_num_layers
        assert config.model.mtp_hybrid_override_pattern == ("*E" if mtp_num_layers else None)
        assert config.model.mtp_use_repeated_layer is bool(mtp_num_layers)
        assert config.model.keep_mtp_spec_in_bf16 is bool(mtp_num_layers)
        assert config.tokenizer.tokenizer_model == model_id

    def test_finetuning_recipes_do_not_expose_model_id(self):
        """Model selection is fixed by separate standard and MTP factories."""
        assert not signature(nemotron_3_nano_sft_config).parameters
        assert not signature(nemotron_3_nano_mtp_sft_config).parameters
        assert set(signature(nemotron_3_nano_peft_config).parameters) == {"peft_scheme"}
        assert set(signature(nemotron_3_nano_mtp_peft_config).parameters) == {"peft_scheme"}

    def test_sft_config_with_custom_directory(self):
        """Test custom directory configuration for SFT."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = nemotron_3_nano_sft_config()

            # Set directory configuration after creation
            run_dir = os.path.join(temp_dir, "finetune_run")
            expected_checkpoint_dir = os.path.join(run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(run_dir, "tb_logs")

            config.checkpoint.save = expected_checkpoint_dir
            config.logger.tensorboard_dir = expected_tensorboard_dir

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed"])
    def test_sft_precision_config(self, precision):
        """Test precision configuration for SFT."""
        config = nemotron_3_nano_sft_config()

        # Modify precision after creation
        config.mixed_precision = precision

        assert config.mixed_precision == precision


@pytest.mark.unit
class TestNemotron3NanoPeft:
    """Test cases for Nemotron 3 Nano PEFT recipe."""

    def test_peft_config_default_lora(self):
        """Test peft_config with default LoRA configuration."""
        config = nemotron_3_nano_peft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)

        # Check default parallelism for LoRA/DoRA
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is False

        # Check expert parallelism
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.expert_model_parallel_size == 8

        # Check PEFT config exists for LoRA
        assert config.peft is not None

        # Check tokenizer
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

        # Check precision
        assert config.mixed_precision == "bf16_mixed"

    def test_peft_config_lora_lr(self):
        """Test that LoRA uses higher learning rate."""
        config = nemotron_3_nano_peft_config(peft_scheme="lora")

        # LoRA should use higher LR (1e-4 default)
        assert config.optimizer.lr == 1e-4
        assert config.peft is not None

    def test_peft_config_dora(self):
        """Test peft_config with DoRA configuration."""
        config = nemotron_3_nano_peft_config(peft_scheme="dora")

        assert config.peft is not None
        # DoRA should also use higher LR
        assert config.optimizer.lr == 1e-4

    def test_peft_config_deepep_settings(self):
        """Test that PEFT config has correct MoE dispatcher settings."""
        config = nemotron_3_nano_peft_config()

        # Check MoE dispatcher settings exist
        assert config.model.moe_token_dispatcher_type is not None

    def test_peft_config_custom_parallelism(self):
        """Test PEFT config with custom parallelism applied after creation."""
        config = nemotron_3_nano_peft_config(peft_scheme="lora")

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

    def test_peft_config_custom_training_params(self):
        """Test PEFT config with custom training parameters applied after creation."""
        config = nemotron_3_nano_peft_config(peft_scheme="lora")

        # Modify training settings after creation
        config.train.train_iters = 500
        config.train.global_batch_size = 64
        config.train.micro_batch_size = 2
        config.optimizer.lr = 5e-5
        config.optimizer.min_lr = 1e-6

        assert config.train.train_iters == 500
        assert config.train.global_batch_size == 64
        assert config.train.micro_batch_size == 2
        assert config.optimizer.lr == 5e-5
        assert config.optimizer.min_lr == 1e-6

    def test_peft_config_with_pretrained_checkpoint(self):
        """Test PEFT config with pretrained checkpoint applied after creation."""
        config = nemotron_3_nano_peft_config(peft_scheme="lora")

        # Set checkpoint path after creation
        config.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

        assert config.checkpoint.pretrained_checkpoint == "/path/to/checkpoint"

    def test_peft_config_with_custom_directory(self):
        """Test custom directory configuration for PEFT."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = nemotron_3_nano_peft_config(peft_scheme="lora")

            # Set directory configuration after creation
            run_dir = os.path.join(temp_dir, "finetune_run")
            expected_checkpoint_dir = os.path.join(run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(run_dir, "tb_logs")

            config.checkpoint.save = expected_checkpoint_dir
            config.logger.tensorboard_dir = expected_tensorboard_dir

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed"])
    def test_peft_precision_config(self, precision):
        """Test precision configuration for PEFT."""
        config = nemotron_3_nano_peft_config(peft_scheme="lora")

        # Modify precision after creation
        config.mixed_precision = precision

        assert config.mixed_precision == precision


@pytest.mark.unit
class TestNemotron3NanoCommon:
    """Test cases common to all Nemotron 3 Nano recipes."""

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_3_nano_pretrain_config,
            nemotron_3_nano_sft_config,
            nemotron_3_nano_peft_config,
        ],
    )
    def test_config_container_structure(self, recipe_fn):
        """Test that all configs return proper ConfigContainer with correct model provider."""
        config = recipe_fn()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)

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
            nemotron_3_nano_pretrain_config,
            nemotron_3_nano_sft_config,
            nemotron_3_nano_peft_config,
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
            nemotron_3_nano_pretrain_config,
            nemotron_3_nano_sft_config,
            nemotron_3_nano_peft_config,
        ],
    )
    def test_moe_model_configuration(self, recipe_fn):
        """Test MoE-specific model configuration from provider."""
        config = recipe_fn()

        # Check MoE settings from HybridModelProvider
        assert config.model.num_moe_experts == 128
        assert config.model.moe_ffn_hidden_size == 1856
        assert config.model.moe_shared_expert_intermediate_size == 3712
        assert config.model.moe_router_topk == 6
        assert config.model.moe_router_topk_scaling_factor == 2.5
