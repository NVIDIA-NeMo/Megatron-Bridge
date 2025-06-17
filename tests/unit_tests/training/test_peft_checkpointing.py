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

import datetime
import logging
import os
from dataclasses import dataclass
from typing import Optional, Union
from unittest.mock import Mock, patch

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.module import MegatronModule

from megatron.hub.models import get_base_model
from megatron.hub.models.gpt import GPTConfig
from megatron.hub.peft.base import PEFT
from megatron.hub.peft.lora import LoRA
from megatron.hub.training.checkpointing import filter_peft_adapter_state_dict, get_model_state_dict, load_checkpoint
from megatron.hub.training.config import CheckpointConfig, ConfigContainer
from megatron.hub.training.state import GlobalState


@dataclass
class MockPEFT(PEFT):
    """Mock PEFT implementation for testing."""

    def __post_init__(self) -> None:
        """Set up mock parameters after dataclass initialization."""
        self.params_to_save = {
            "layer1.adapter.weight",
            "layer2.adapter.bias",
            "layer3.adapters.lora_A",
            "layer3.adapters.lora_B",
        }

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """Transform method that returns the module unchanged for testing."""
        return module

    def adapter_key_filter(self, key: Union[str, tuple]) -> bool:
        """Filter function that only allows adapter parameters."""
        if isinstance(key, tuple):
            return key[1].requires_grad
        return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters") or "lora_" in key


class TestPEFTCheckpointing:
    """Test suite for PEFT checkpoint filtering functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with both adapter and base parameters."""
        model = Mock()

        # Mock state dict with both adapter and base model parameters
        state_dict = {
            # Base model parameters (should be filtered out)
            "embedding.weight": torch.randn(1000, 512),
            "layer1.linear.weight": torch.randn(512, 512),
            "layer1.linear.bias": torch.randn(512),
            "layer2.attention.weight": torch.randn(512, 512),
            # Adapter parameters (should be kept)
            "layer1.adapter.weight": torch.randn(8, 512),
            "layer2.adapter.bias": torch.randn(8),
            "layer3.adapters.lora_A": torch.randn(8, 512),
            "layer3.adapters.lora_B": torch.randn(512, 8),
            "layer4.linear.adapter.weight": torch.randn(8, 512),
            # Additional base parameters
            "output.weight": torch.randn(512, 1000),
            "output.bias": torch.randn(1000),
        }

        model.sharded_state_dict.return_value = state_dict

        return model

    @pytest.fixture
    def mock_peft_config(self):
        """Create a mock PEFT configuration."""
        return MockPEFT()

    def test_get_model_state_dict_with_distributed_ckpt(self, mock_model, mock_peft_config):
        """Test that filtering works with distributed checkpointing."""
        # Test with distributed checkpointing (use_dist_ckpt=True)
        filtered_state = get_model_state_dict(mock_model, True, mock_peft_config)

        # Verify only adapter parameters are kept
        expected_keys = {
            "layer1.adapter.weight",
            "layer2.adapter.bias",
            "layer3.adapters.lora_A",
            "layer3.adapters.lora_B",
            "layer4.linear.adapter.weight",
        }

        assert set(filtered_state.keys()) == expected_keys
        assert len(filtered_state) == 5

        # Verify the correct method was called
        mock_model.sharded_state_dict.assert_called_once()

    def test_get_model_state_dict_without_peft(self, mock_model):
        """Test that full state dict is returned when PEFT is disabled."""
        # Test without PEFT (peft_config=None) using distributed checkpointing
        filtered_state = get_model_state_dict(mock_model, True, None)

        # Verify all parameters are kept
        expected_keys = {
            "embedding.weight",
            "layer1.linear.weight",
            "layer1.linear.bias",
            "layer2.attention.weight",
            "layer1.adapter.weight",
            "layer2.adapter.bias",
            "layer3.adapters.lora_A",
            "layer3.adapters.lora_B",
            "layer4.linear.adapter.weight",
            "output.weight",
            "output.bias",
        }

        assert set(filtered_state.keys()) == expected_keys
        assert len(filtered_state) == 11

    def test_empty_state_dict_with_peft(self, mock_peft_config):
        """Test filtering behavior with empty state dict."""
        model = Mock()
        model.sharded_state_dict.return_value = {}

        filtered_state = get_model_state_dict(model, True, mock_peft_config)

        assert filtered_state == {}

    def test_no_adapter_params_with_peft(self, mock_peft_config):
        """Test filtering when no adapter parameters are present."""
        model = Mock()
        state_dict = {
            "embedding.weight": torch.randn(1000, 512),
            "layer1.linear.weight": torch.randn(512, 512),
            "output.weight": torch.randn(512, 1000),
        }
        model.sharded_state_dict.return_value = state_dict

        filtered_state = get_model_state_dict(model, True, mock_peft_config)

        # Should be empty since no adapter parameters match the filter
        assert filtered_state == {}

    def test_checkpoint_size_reduction(self, mock_model, mock_peft_config):
        """Test that PEFT filtering significantly reduces checkpoint size."""
        # Get full state dict using distributed checkpointing
        full_state = get_model_state_dict(mock_model, True, None)

        # Get filtered state dict using distributed checkpointing
        filtered_state = get_model_state_dict(mock_model, True, mock_peft_config)

        # Calculate approximate size reduction
        full_params = sum(p.numel() for p in full_state.values())
        filtered_params = sum(p.numel() for p in filtered_state.values())

        # Should be significant reduction (adapter params are much smaller)
        reduction_ratio = filtered_params / full_params
        assert reduction_ratio < 0.1  # Less than 10% of original size

        # Verify we kept the right number of parameters
        assert len(filtered_state) == 5  # Only adapter parameters
        assert len(full_state) == 11  # All parameters


class TestPEFTCheckpointLoading:
    """Test suite for PEFT checkpoint loading functionality."""

    @pytest.fixture
    def mock_peft_config(self):
        """Create a mock PEFT configuration."""
        return MockPEFT()

    @pytest.fixture
    def sample_state_dict(self):
        """Create a sample state dict with mixed parameters."""
        return {
            # Base model parameters
            "embedding.weight": torch.randn(1000, 512),
            "layer1.linear.weight": torch.randn(512, 512),
            "layer2.attention.weight": torch.randn(512, 512),
            # Adapter parameters
            "layer1.adapter.weight": torch.randn(8, 512),
            "layer2.adapter.bias": torch.randn(8),
            "layer3.adapters.lora_A": torch.randn(8, 512),
            "layer3.adapters.lora_B": torch.randn(512, 8),
            # Base model output
            "output.weight": torch.randn(512, 1000),
        }

    def test_filter_peft_adapter_state_dict_basic(self, mock_peft_config, sample_state_dict):
        """Test basic filtering functionality."""
        filtered_dict = filter_peft_adapter_state_dict(sample_state_dict, mock_peft_config)

        # Should only contain adapter parameters
        expected_keys = {
            "layer1.adapter.weight",
            "layer2.adapter.bias",
            "layer3.adapters.lora_A",
            "layer3.adapters.lora_B",
        }

        assert set(filtered_dict.keys()) == expected_keys
        assert len(filtered_dict) == 4

        # Verify values are preserved correctly
        for key in expected_keys:
            assert torch.equal(filtered_dict[key], sample_state_dict[key])

    def test_filter_peft_adapter_state_dict_empty_input(self, mock_peft_config):
        """Test filtering with empty state dict."""
        filtered_dict = filter_peft_adapter_state_dict({}, mock_peft_config)
        assert filtered_dict == {}

    def test_filter_peft_adapter_state_dict_no_adapters(self, mock_peft_config):
        """Test filtering when no adapter parameters are present."""
        state_dict = {
            "embedding.weight": torch.randn(1000, 512),
            "layer1.linear.weight": torch.randn(512, 512),
            "output.weight": torch.randn(512, 1000),
        }

        filtered_dict = filter_peft_adapter_state_dict(state_dict, mock_peft_config)
        assert filtered_dict == {}

    def test_filter_peft_adapter_state_dict_all_adapters(self, mock_peft_config):
        """Test filtering when all parameters are adapters."""
        state_dict = {
            "layer1.adapter.weight": torch.randn(8, 512),
            "layer2.adapter.bias": torch.randn(8),
            "layer3.adapters.lora_A": torch.randn(8, 512),
        }

        filtered_dict = filter_peft_adapter_state_dict(state_dict, mock_peft_config)

        # All parameters should be kept
        assert set(filtered_dict.keys()) == set(state_dict.keys())
        assert len(filtered_dict) == 3

    @patch("megatron.hub.training.checkpointing._load_base_checkpoint")
    @patch("megatron.hub.training.checkpointing.checkpoint_exists")
    @patch("megatron.hub.training.checkpointing.filter_peft_adapter_state_dict")
    def test_load_checkpoint_peft_resume_detection(self, mock_filter, mock_checkpoint_exists, mock_load_base):
        """Test that PEFT resume is properly detected and triggers filtering."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        mock_state_dict = {
            "model": {
                "layer1.linear.weight": torch.randn(512, 512),
                "layer1.adapter.weight": torch.randn(8, 512),
            },
            "checkpoint_version": 3.0,
        }
        mock_load_base.return_value = (mock_state_dict, "/path/to/checkpoint", False, None)

        # Mock filtered result
        mock_filtered_dict = {"layer1.adapter.weight": torch.randn(8, 512)}
        mock_filter.return_value = mock_filtered_dict

        # Create mock global state for PEFT resume scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = MockPEFT()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = True
        mock_cfg.checkpoint.load_optim = True
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.hub.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.hub.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.hub.training.checkpointing.update_num_microbatches"),
            patch("megatron.hub.training.checkpointing.fix_query_key_value_ordering"),
            patch("megatron.hub.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.hub.training.checkpointing.set_checkpoint_version"),
            patch("torch.distributed.barrier"),
            patch("megatron.hub.training.checkpointing.print_rank_0"),
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0

            _ = load_checkpoint(
                mock_state,
                mock_model,
                None,  # No optimizer
                None,  # No scheduler
                strict=True,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )

            # Verify PEFT filtering was called
            mock_filter.assert_called_once_with(mock_state_dict["model"], mock_cfg.peft)

            # Verify model.load_state_dict was called with filtered dict and strict=False
            mock_model[0].load_state_dict.assert_called_once_with(mock_filtered_dict, strict=False)

    @patch("megatron.hub.training.checkpointing._load_base_checkpoint")
    @patch("megatron.hub.training.checkpointing.checkpoint_exists")
    def test_load_checkpoint_non_peft_regular_loading(self, mock_checkpoint_exists, mock_load_base):
        """Test that non-PEFT scenarios use regular loading without filtering."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        mock_state_dict = {
            "model": {
                "layer1.linear.weight": torch.randn(512, 512),
                "layer2.linear.weight": torch.randn(512, 512),
            },
            "checkpoint_version": 3.0,
        }
        mock_load_base.return_value = (mock_state_dict, "/path/to/checkpoint", False, None)

        # Create mock global state for non-PEFT scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = None  # No PEFT
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = None
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = True
        mock_cfg.checkpoint.load_optim = True
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.hub.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.hub.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.hub.training.checkpointing.update_num_microbatches"),
            patch("megatron.hub.training.checkpointing.fix_query_key_value_ordering"),
            patch("megatron.hub.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.hub.training.checkpointing.set_checkpoint_version"),
            patch("torch.distributed.barrier"),
            patch("megatron.hub.training.checkpointing.print_rank_0"),
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0

            _ = load_checkpoint(
                mock_state,
                mock_model,
                None,  # No optimizer
                None,  # No scheduler
                strict=True,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )

            # Verify model.load_state_dict was called with full dict and original strict value
            mock_model[0].load_state_dict.assert_called_once_with(mock_state_dict["model"], strict=True)

    @patch("megatron.hub.training.checkpointing._load_base_checkpoint")
    @patch("megatron.hub.training.checkpointing.checkpoint_exists")
    @patch("megatron.hub.training.checkpointing.filter_peft_adapter_state_dict")
    def test_load_checkpoint_peft_resume_multi_model(self, mock_filter, mock_checkpoint_exists, mock_load_base):
        """Test PEFT resume with multiple model chunks (pipeline parallelism)."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        mock_state_dict = {
            "model0": {"layer1.adapter.weight": torch.randn(8, 512)},
            "model1": {"layer2.adapter.weight": torch.randn(8, 512)},
            "checkpoint_version": 3.0,
        }
        mock_load_base.return_value = (mock_state_dict, "/path/to/checkpoint", False, None)

        # Mock filtered results
        mock_filter.side_effect = [
            {"layer1.adapter.weight": torch.randn(8, 512)},  # For model0
            {"layer2.adapter.weight": torch.randn(8, 512)},  # For model1
        ]

        # Create mock global state for PEFT resume scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = MockPEFT()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = True
        mock_cfg.checkpoint.load_optim = True
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0

        # Create mock models (2 chunks for pipeline parallelism)
        mock_model = [Mock(), Mock()]
        mock_model[0].load_state_dict = Mock()
        mock_model[1].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.hub.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.hub.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.hub.training.checkpointing.update_num_microbatches"),
            patch("megatron.hub.training.checkpointing.fix_query_key_value_ordering"),
            patch("megatron.hub.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.hub.training.checkpointing.set_checkpoint_version"),
            patch("megatron.core.mpu.set_virtual_pipeline_model_parallel_rank"),
            patch("torch.distributed.barrier"),
            patch("megatron.hub.training.checkpointing.print_rank_0"),
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0

            _ = load_checkpoint(
                mock_state,
                mock_model,
                None,  # No optimizer
                None,  # No scheduler
                strict=True,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )

            # Verify filtering was called for both models
            assert mock_filter.call_count == 2
            mock_filter.assert_any_call(mock_state_dict["model0"], mock_cfg.peft)
            mock_filter.assert_any_call(mock_state_dict["model1"], mock_cfg.peft)

            # Verify both models had load_state_dict called with strict=False
            mock_model[0].load_state_dict.assert_called_once()
            mock_model[1].load_state_dict.assert_called_once()

            # Verify strict=False was used for both calls
            args0, kwargs0 = mock_model[0].load_state_dict.call_args
            args1, kwargs1 = mock_model[1].load_state_dict.call_args
            assert kwargs0.get("strict", True) == False
            assert kwargs1.get("strict", True) == False

    def test_filter_peft_adapter_state_dict_uses_adapter_key_filter(self, sample_state_dict):
        """Test that filter_peft_adapter_state_dict correctly uses PEFT's adapter_key_filter method."""

        # Create a custom PEFT config with specific filtering logic
        class CustomPEFT(PEFT):
            def __init__(self):
                self.custom_adapter_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}

            def transform(self, module, name=None, prefix=None):
                return module

            def adapter_key_filter(self, key):
                # Custom logic: only allow specific keys
                return key in self.custom_adapter_keys

        custom_peft = CustomPEFT()

        filtered_dict = filter_peft_adapter_state_dict(sample_state_dict, custom_peft)

        # Should only contain the keys that the custom filter allows
        expected_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}
        assert set(filtered_dict.keys()) == expected_keys
        assert len(filtered_dict) == 2


@pytest.mark.run_only_on("GPU")
class TestPEFTCheckpointingIntegration:
    """Integration tests using real GPT models and LoRA PEFT configurations."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        """Setup and teardown parallel state for Megatron tests."""

        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            init_process_group_kwargs = {
                "backend": "nccl" if device_count > 0 else "gloo",
                "world_size": 1,
                "rank": 0,
                "timeout": datetime.timedelta(minutes=30),
            }

            dist.init_process_group(**init_process_group_kwargs)

        assert dist.is_initialized(), "Distributed backend not initialized"

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"

        from megatron.hub.training.initialize import _set_random_seed

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
        )

        yield

        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                # Clean up environment variables
                for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    @pytest.fixture
    def gpt_model_and_config(self):
        """Create a minimal GPT model with Megatron modules for integration testing."""

        # Create minimal GPT config for testing
        gpt_config = GPTConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=64,
            vocab_size=256,
            ffn_hidden_size=256,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        base_model = get_base_model(gpt_config)

        # Verify we got Megatron modules
        assert isinstance(base_model, list)
        assert len(base_model) > 0
        assert all(isinstance(chunk, MegatronModule) for chunk in base_model)

        # Move to CUDA if available
        if torch.cuda.is_available():
            base_model = [chunk.cuda() for chunk in base_model]

        # Create LoRA PEFT config
        lora_config = LoRA(
            target_modules=["linear_qkv", "linear_proj"],
            dim=8,
            alpha=16,
            dropout=0.1,
        )

        return base_model, lora_config

    def test_get_model_state_dict_integration_without_peft(self, gpt_model_and_config):
        """Test get_model_state_dict with real GPT model without PEFT."""
        model, lora_config = gpt_model_and_config

        # Test without PEFT on first model chunk using distributed checkpointing
        state_dict = get_model_state_dict(model[0], use_dist_ckpt=True, peft_config=None)

        # Verify we got a state dict with model parameters
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Check for expected GPT model components
        param_names = list(state_dict.keys())

        # Should have some transformer parameters
        has_transformer_params = any("transformer" in name or "layer" in name for name in param_names)
        assert has_transformer_params, f"No transformer parameters found in {param_names}"

    def test_get_model_state_dict_integration_with_peft(self, gpt_model_and_config):
        """Test get_model_state_dict with real GPT model and LoRA PEFT."""
        model, lora_config = gpt_model_and_config

        # Apply PEFT to the model
        peft_model = lora_config(model, training=True)

        # Set up params_to_save for the PEFT config
        lora_config.set_params_to_save(peft_model)

        # Get state dict without PEFT filtering
        full_state_dict = get_model_state_dict(peft_model[0], use_dist_ckpt=True, peft_config=None)

        # Get state dict with PEFT filtering
        filtered_state_dict = get_model_state_dict(peft_model[0], use_dist_ckpt=True, peft_config=lora_config)

        # Verify filtering worked
        assert len(filtered_state_dict) < len(full_state_dict), (
            f"Filtered state dict ({len(filtered_state_dict)}) should be smaller than "
            f"full state dict ({len(full_state_dict)})"
        )

        # Verify only adapter parameters are in filtered state dict
        for param_name in filtered_state_dict.keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should not be in filtered state dict"
            )

        # Verify some adapter parameters were found
        assert len(filtered_state_dict) > 0, "No adapter parameters found in filtered state dict"

        # Check that adapter parameters have expected naming patterns
        adapter_param_names = list(filtered_state_dict.keys())
        has_lora_params = any("lora" in name.lower() or "adapter" in name.lower() for name in adapter_param_names)

        assert has_lora_params, f"Expected LoRA or adapter parameters in {adapter_param_names}"

    def test_checkpoint_size_reduction_integration(self, gpt_model_and_config):
        """Test that PEFT filtering provides significant size reduction with real model."""
        model, lora_config = gpt_model_and_config

        # Apply PEFT to the model
        peft_model = lora_config(model, training=True)
        lora_config.set_params_to_save(peft_model)

        # Get both state dicts from first model chunk using distributed checkpointing
        full_state_dict = get_model_state_dict(peft_model[0], use_dist_ckpt=True, peft_config=None)
        filtered_state_dict = get_model_state_dict(peft_model[0], use_dist_ckpt=True, peft_config=lora_config)

        def count_parameters(state_dict):
            """Count parameters handling both regular tensors and ShardedTensor objects."""
            total_params = 0
            for param in state_dict.values():
                if isinstance(param, ShardedTensor):
                    import math

                    total_params += math.prod(param.local_shape)
                else:
                    # Skip unknown types
                    continue
            return total_params

        # Calculate parameter count reduction
        full_param_count = count_parameters(full_state_dict)
        filtered_param_count = count_parameters(filtered_state_dict)

        reduction_ratio = filtered_param_count / full_param_count if full_param_count > 0 else 1.0

        logging.debug(f"Full model parameters: {full_param_count}")
        logging.debug(f"Adapter parameters: {filtered_param_count}")
        logging.debug(f"Reduction ratio: {reduction_ratio:.4f}")
        logging.debug(f"Size reduction: {(1 - reduction_ratio) * 100:.1f}%")

        # Verify significant reduction (LoRA adapters should be much smaller)
        assert reduction_ratio < 0.5, f"Expected significant size reduction, got ratio {reduction_ratio:.4f}"

        # Verify we still have some parameters
        assert filtered_param_count > 0, "Filtered state dict should not be empty"


class TestPEFTCheckpointingValidation:
    """Simple validation tests to ensure test infrastructure works correctly."""

    def test_imports_work(self):
        """Test that all necessary imports are available."""
        # Verify core functions are importable
        assert filter_peft_adapter_state_dict is not None
        assert load_checkpoint is not None
        assert MockPEFT is not None

    def test_mock_peft_basic_functionality(self):
        """Test that MockPEFT behaves as expected."""
        mock_peft = MockPEFT()

        # Test adapter key filtering
        assert mock_peft.adapter_key_filter("layer1.adapter.weight") == True
        assert mock_peft.adapter_key_filter("layer1.linear.weight") == False
        assert mock_peft.adapter_key_filter("layer3.adapters.lora_A") == True

        # Test params_to_save is set
        assert hasattr(mock_peft, "params_to_save")
        assert len(mock_peft.params_to_save) > 0
