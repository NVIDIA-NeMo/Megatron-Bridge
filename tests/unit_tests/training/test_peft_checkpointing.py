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
import os
from dataclasses import dataclass
from unittest.mock import Mock, patch

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.checkpointing import (
    _is_model_section,
    apply_peft_adapter_filter_to_state_dict,
    load_checkpoint,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer
from megatron.bridge.training.state import GlobalState


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

    def transform(self, module: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
        """Transform method that returns the module unchanged for testing."""
        return module

    def adapter_key_filter(self, key: str | tuple) -> bool:
        """Filter function that only allows adapter parameters."""
        if isinstance(key, tuple):
            return key[1].requires_grad
        return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters") or "lora_" in key


class TestIsModelSection:
    """Test suite for _is_model_section helper function."""

    def test_is_model_section_single_model(self):
        """Test detection of single model section."""
        assert _is_model_section("model") == True

    def test_is_model_section_pipeline_models(self):
        """Test detection of pipeline model sections."""
        assert _is_model_section("model0") == True
        assert _is_model_section("model1") == True
        assert _is_model_section("model42") == True
        assert _is_model_section("model999") == True

    def test_is_model_section_non_model_sections(self):
        """Test that non-model sections are correctly identified."""
        assert _is_model_section("optimizer") == False
        assert _is_model_section("iteration") == False
        assert _is_model_section("checkpoint_version") == False
        assert _is_model_section("rng_state") == False
        assert _is_model_section("opt_param_scheduler") == False

    def test_is_model_section_invalid_model_keys(self):
        """Test that invalid model-like keys are correctly rejected."""
        assert _is_model_section("model_not_digit") == False
        assert _is_model_section("modelabc") == False
        assert _is_model_section("model_") == False
        assert _is_model_section("model10_extra") == False
        assert _is_model_section("my_model") == False
        assert _is_model_section("models") == False

    def test_is_model_section_edge_cases(self):
        """Test edge cases for model section detection."""
        assert _is_model_section("") == False
        assert _is_model_section("model00") == True  # Leading zeros are valid digits
        assert _is_model_section("model01") == True


class TestApplyPeftAdapterFilterToStateDict:
    """Test suite for apply_peft_adapter_filter_to_state_dict functionality.

    Tests the PEFT adapter filtering that processes complete checkpoint state dictionaries
    to retain only adapter parameters in model sections while preserving metadata.
    """

    @pytest.fixture
    def mock_peft_config(self):
        """Create a mock PEFT configuration."""
        return MockPEFT()

    @pytest.fixture
    def sample_complete_state_dict(self):
        """Create a sample complete state dict with all checkpoint components."""
        return {
            # Metadata
            "checkpoint_version": 3.0,
            "iteration": 1000,
            # Single model state
            "model": {
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
            },
            # Optimizer state
            "optimizer": {
                "state": {},
                "param_groups": [],
            },
            # Scheduler state
            "opt_param_scheduler": {
                "lr": 0.001,
            },
            # RNG state
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

    @pytest.fixture
    def sample_multi_model_state_dict(self):
        """Create a sample state dict with multiple model chunks (pipeline parallelism)."""
        return {
            # Metadata
            "checkpoint_version": 3.0,
            "iteration": 1000,
            # Multiple model states (pipeline parallelism)
            "model0": {
                "layer1.linear.weight": torch.randn(512, 512),
                "layer1.adapter.weight": torch.randn(8, 512),
            },
            "model1": {
                "layer2.attention.weight": torch.randn(512, 512),
                "layer2.adapter.bias": torch.randn(8),
            },
            "model2": {
                "layer3.output.weight": torch.randn(512, 1000),
                "layer3.adapters.lora_A": torch.randn(8, 512),
                "layer3.adapters.lora_B": torch.randn(512, 8),
            },
            # Optimizer and other states
            "optimizer": {"state": {}, "param_groups": []},
            "opt_param_scheduler": {"lr": 0.001},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

    def test_apply_peft_adapter_filter_single_model(self, mock_peft_config, sample_complete_state_dict):
        """Test filtering a complete state dict with a single model."""
        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_complete_state_dict, mock_peft_config)

        # Verify metadata is preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict
        assert "opt_param_scheduler" in filtered_dict
        assert "rng_state" in filtered_dict

        # Verify model state is filtered
        expected_adapter_keys = {
            "layer1.adapter.weight",
            "layer2.adapter.bias",
            "layer3.adapters.lora_A",
            "layer3.adapters.lora_B",
        }
        assert set(filtered_dict["model"].keys()) == expected_adapter_keys
        assert len(filtered_dict["model"]) == 4

        # Verify values are preserved correctly
        for key in expected_adapter_keys:
            assert torch.equal(filtered_dict["model"][key], sample_complete_state_dict["model"][key])

    def test_apply_peft_adapter_filter_multi_model(self, mock_peft_config, sample_multi_model_state_dict):
        """Test filtering a complete state dict with multiple model chunks."""
        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_multi_model_state_dict, mock_peft_config)

        # Verify metadata is preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict
        assert "opt_param_scheduler" in filtered_dict
        assert "rng_state" in filtered_dict

        # Verify each model chunk is filtered correctly
        assert set(filtered_dict["model0"].keys()) == {"layer1.adapter.weight"}
        assert set(filtered_dict["model1"].keys()) == {"layer2.adapter.bias"}
        assert set(filtered_dict["model2"].keys()) == {"layer3.adapters.lora_A", "layer3.adapters.lora_B"}

        # Verify values are preserved correctly
        for model_key in ["model0", "model1", "model2"]:
            for param_key in filtered_dict[model_key].keys():
                assert torch.equal(
                    filtered_dict[model_key][param_key], sample_multi_model_state_dict[model_key][param_key]
                )

    def test_apply_peft_adapter_filter_no_adapters_found(self, mock_peft_config):
        """Test filtering when no adapter parameters match the filter."""
        state_dict_no_adapters = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": {
                # Only base model parameters (no adapters)
                "embedding.weight": torch.randn(1000, 512),
                "layer1.linear.weight": torch.randn(512, 512),
                "output.weight": torch.randn(512, 1000),
            },
            "optimizer": {"state": {}, "param_groups": []},
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict_no_adapters, mock_peft_config)

        # Model section should be empty since no adapters match
        assert filtered_dict["model"] == {}

        # Non-model sections should be preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict

    def test_apply_peft_adapter_filter_no_model_states(self, mock_peft_config):
        """Test filtering when no model states are present."""
        state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict, mock_peft_config)

        # Should preserve all non-model keys
        assert filtered_dict == state_dict

    def test_apply_peft_adapter_filter_empty_model_states(self, mock_peft_config):
        """Test filtering when model states are empty."""
        state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": {},
            "model0": {},
            "model1": {},
            "optimizer": {"state": {}, "param_groups": []},
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict, mock_peft_config)

        # Model states should remain empty but present
        assert filtered_dict["model"] == {}
        assert filtered_dict["model0"] == {}
        assert filtered_dict["model1"] == {}
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000

    def test_apply_peft_adapter_filter_mixed_model_keys(self, mock_peft_config):
        """Test filtering with mixed model keys (some numerical, some not)."""
        state_dict = {
            "checkpoint_version": 3.0,
            "model": {"layer1.adapter.weight": torch.randn(8, 512)},
            "model0": {"layer2.adapter.bias": torch.randn(8)},
            "model5": {"layer3.adapters.lora_A": torch.randn(8, 512)},
            "model_not_digit": {"should.not.be.filtered": torch.randn(512)},  # Should not be filtered
            "modelabc": {"also.not.filtered": torch.randn(256)},  # Should not be filtered
            "optimizer": {"state": {}},
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict, mock_peft_config)

        # Verify correct models are filtered
        assert set(filtered_dict["model"].keys()) == {"layer1.adapter.weight"}
        assert set(filtered_dict["model0"].keys()) == {"layer2.adapter.bias"}
        assert set(filtered_dict["model5"].keys()) == {"layer3.adapters.lora_A"}

        # Verify non-model keys are preserved unchanged
        assert filtered_dict["model_not_digit"] == state_dict["model_not_digit"]
        assert filtered_dict["modelabc"] == state_dict["modelabc"]
        assert filtered_dict["optimizer"] == state_dict["optimizer"]

    def test_apply_peft_adapter_filter_uses_correct_filter_function(self, sample_complete_state_dict):
        """Test that apply_peft_adapter_filter_to_state_dict uses the PEFT config's adapter_key_filter method."""

        # Create a custom PEFT config with specific filtering logic
        class CustomPEFT(PEFT):
            def __init__(self):
                self.allowed_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}

            def transform(self, module, name=None, prefix=None):
                return module

            def adapter_key_filter(self, key):
                # Custom logic: only allow specific keys
                return key in self.allowed_keys

        custom_peft = CustomPEFT()

        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_complete_state_dict, custom_peft)

        # Should only contain the keys that the custom filter allows
        expected_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}
        assert set(filtered_dict["model"].keys()) == expected_keys
        assert len(filtered_dict["model"]) == 2

        # Verify metadata is still preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000


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

    def test_apply_peft_adapter_filter_uses_adapter_key_filter(self):
        """Test that apply_peft_adapter_filter_to_state_dict correctly uses PEFT's adapter_key_filter method."""
        # Create sample complete state dict with mixed parameters
        sample_complete_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": {
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
            },
            "optimizer": {"state": {}, "param_groups": []},
        }

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

        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_complete_state_dict, custom_peft)

        # Should only contain the keys that the custom filter allows in the model section
        expected_model_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}
        assert set(filtered_dict["model"].keys()) == expected_model_keys
        assert len(filtered_dict["model"]) == 2

        # Verify values are preserved correctly
        for key in expected_model_keys:
            assert torch.equal(filtered_dict["model"][key], sample_complete_state_dict["model"][key])

        # Verify non-model sections are preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.apply_peft_adapter_filter_to_state_dict")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    def test_load_checkpoint_peft_resume_detection(
        self, mock_generate_state_dict, mock_filter, mock_checkpoint_exists, mock_load_base
    ):
        """Test that PEFT resume is properly detected and triggers filtering."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        # Create deterministic tensors to avoid random comparison issues
        torch.manual_seed(42)
        base_weight = torch.randn(512, 512)
        adapter_weight = torch.randn(8, 512)

        # This is what generate_state_dict would return (full state dict)
        full_generated_state_dict = {
            "model": {
                "layer1.linear.weight": base_weight,
                "layer1.adapter.weight": adapter_weight,
            },
            "checkpoint_version": 3.0,
        }

        # This is what apply_peft_adapter_filter_to_state_dict should return (filtered)
        filtered_sharded_state_dict = {
            "model": {"layer1.adapter.weight": adapter_weight},
            "checkpoint_version": 3.0,
        }
        mock_filter.return_value = filtered_sharded_state_dict

        # _load_base_checkpoint should return only the parameters that were in the filtered sharded state dict
        # Since the filtering happens BEFORE _load_base_checkpoint, it should only load adapter parameters
        mock_load_base.return_value = (filtered_sharded_state_dict, "/path/to/checkpoint", False, None)

        # Create mock global state for PEFT resume scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = MockPEFT()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = False  # Disable RNG loading for focused testing
        mock_cfg.checkpoint.load_optim = False  # Disable optimizer loading for focused testing

        # Add necessary model config attributes
        mock_cfg.model = Mock()
        mock_cfg.model.tensor_model_parallel_size = 1
        mock_cfg.model.pipeline_model_parallel_size = 1
        mock_cfg.checkpoint.auto_detect_ckpt_format = False
        mock_cfg.checkpoint.ckpt_format = "torch_dist"
        mock_cfg.checkpoint.non_persistent_save_interval = None
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.train_state.step = 1000  # Set to integer for comparisons
        mock_state.train_state.floating_point_operations_so_far = 50000

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.bridge.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.bridge.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.bridge.training.checkpointing.update_num_microbatches"),
            patch("megatron.bridge.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
            patch("megatron.bridge.training.checkpointing.restore_sharded_modelopt_state"),
            patch("torch.distributed.barrier"),
            patch("megatron.bridge.training.checkpointing.print_rank_0"),
            patch("megatron.bridge.training.checkpointing.read_run_config") as mock_read_run_config,
            patch("megatron.bridge.training.checkpointing.unwrap_model") as mock_unwrap_model,
            patch("megatron.bridge.training.checkpointing.mpu.get_tensor_model_parallel_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.mpu.get_tensor_model_parallel_world_size", return_value=1),
            patch("megatron.bridge.training.checkpointing.mpu.get_pipeline_model_parallel_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.mpu.get_pipeline_model_parallel_world_size", return_value=1),
            patch("os.path.exists") as mock_exists,
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0
            mock_unwrap_model.return_value = mock_model

            # Mock file existence - run_config.yaml exists, train_state.pt doesn't (to use read_train_state mock)
            def mock_exists_side_effect(path):
                if "run_config.yaml" in path:
                    return True  # run_config.yaml exists
                elif "train_state.pt" in path:
                    return False  # train_state.pt doesn't exist, use mock
                return False

            mock_exists.side_effect = mock_exists_side_effect

            # Mock generate_state_dict to return the full state dict (before filtering)
            mock_generate_state_dict.return_value = full_generated_state_dict

            # Mock run config for non-PEFT scenario
            mock_run_config = {
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
                "checkpoint": {"save_rng": True, "save_optim": True, "fully_parallel_save": True},
            }
            mock_read_run_config.return_value = mock_run_config

            _ = load_checkpoint(
                mock_state,
                mock_model,
                None,  # No optimizer
                None,  # No scheduler
                strict=True,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )

            # Verify PEFT filtering was called on the generated sharded state dict
            mock_filter.assert_called_once_with(full_generated_state_dict, mock_cfg.peft)

            # Verify model.load_state_dict was called with filtered dict and strict=False
            mock_model[0].load_state_dict.assert_called_once_with(filtered_sharded_state_dict["model"], strict=False)

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    def test_load_checkpoint_non_peft_regular_loading(self, mock_checkpoint_exists, mock_load_base):
        """Test that non-PEFT scenarios use regular loading without filtering."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        torch.manual_seed(44)
        linear_weight_1 = torch.randn(512, 512)
        linear_weight_2 = torch.randn(512, 512)

        mock_state_dict = {
            "model": {
                "layer1.linear.weight": linear_weight_1,
                "layer2.linear.weight": linear_weight_2,
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
        mock_cfg.checkpoint.load_rng = False  # Disable RNG loading for focused testing
        mock_cfg.checkpoint.load_optim = False  # Disable optimizer loading for focused testing

        # Add necessary model config attributes
        mock_cfg.model = Mock()
        mock_cfg.model.tensor_model_parallel_size = 1
        mock_cfg.model.pipeline_model_parallel_size = 1
        mock_cfg.checkpoint.auto_detect_ckpt_format = False
        mock_cfg.checkpoint.ckpt_format = "torch_dist"
        mock_cfg.checkpoint.non_persistent_save_interval = None
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.train_state.step = 1000  # Set to integer for comparisons
        mock_state.train_state.floating_point_operations_so_far = 50000

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.bridge.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.bridge.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.bridge.training.checkpointing.update_num_microbatches"),
            patch("megatron.bridge.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
            patch("megatron.bridge.training.checkpointing.restore_sharded_modelopt_state"),
            patch("torch.distributed.barrier"),
            patch("megatron.bridge.training.checkpointing.print_rank_0"),
            patch("megatron.bridge.training.checkpointing.read_run_config") as mock_read_run_config,
            patch("megatron.bridge.training.checkpointing.unwrap_model") as mock_unwrap_model,
            patch("megatron.bridge.training.checkpointing.mpu.get_tensor_model_parallel_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.mpu.get_tensor_model_parallel_world_size", return_value=1),
            patch("megatron.bridge.training.checkpointing.mpu.get_pipeline_model_parallel_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.mpu.get_pipeline_model_parallel_world_size", return_value=1),
            patch("os.path.exists") as mock_exists,
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0
            mock_unwrap_model.return_value = mock_model

            # Mock file existence - run_config.yaml exists, train_state.pt doesn't (to use read_train_state mock)
            def mock_exists_side_effect(path):
                if "run_config.yaml" in path:
                    return True  # run_config.yaml exists
                elif "train_state.pt" in path:
                    return False  # train_state.pt doesn't exist, use mock
                return False

            mock_exists.side_effect = mock_exists_side_effect

            # Mock run config for non-PEFT scenario
            mock_run_config = {
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
                "checkpoint": {"save_rng": True, "save_optim": True, "fully_parallel_save": True},
            }
            mock_read_run_config.return_value = mock_run_config

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

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.apply_peft_adapter_filter_to_state_dict")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    def test_load_checkpoint_peft_resume_multi_model(
        self, mock_generate_state_dict, mock_filter, mock_checkpoint_exists, mock_load_base
    ):
        """Test PEFT resume with multiple model chunks (pipeline parallelism)."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        torch.manual_seed(43)
        base_weight_0 = torch.randn(512, 512)
        adapter_weight_0 = torch.randn(8, 512)
        base_weight_1 = torch.randn(512, 512)
        adapter_weight_1 = torch.randn(8, 512)

        # This is what generate_state_dict would return (full state dict)
        full_generated_state_dict = {
            "model0": {
                "layer1.linear.weight": base_weight_0,
                "layer1.adapter.weight": adapter_weight_0,
            },
            "model1": {
                "layer2.linear.weight": base_weight_1,
                "layer2.adapter.weight": adapter_weight_1,
            },
            "checkpoint_version": 3.0,
        }

        # This is what apply_peft_adapter_filter_to_state_dict should return (filtered)
        filtered_sharded_state_dict = {
            "model0": {"layer1.adapter.weight": adapter_weight_0},
            "model1": {"layer2.adapter.weight": adapter_weight_1},
            "checkpoint_version": 3.0,
        }
        mock_filter.return_value = filtered_sharded_state_dict

        # _load_base_checkpoint should return only the parameters that were in the filtered sharded state dict
        mock_load_base.return_value = (filtered_sharded_state_dict, "/path/to/checkpoint", False, None)

        # Create mock global state for PEFT resume scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = MockPEFT()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = False  # Disable RNG loading for focused testing
        mock_cfg.checkpoint.load_optim = False  # Disable optimizer loading for focused testing

        # Add necessary model config attributes
        mock_cfg.model = Mock()
        mock_cfg.model.tensor_model_parallel_size = 1
        mock_cfg.model.pipeline_model_parallel_size = 1
        mock_cfg.checkpoint.auto_detect_ckpt_format = False
        mock_cfg.checkpoint.ckpt_format = "torch_dist"
        mock_cfg.checkpoint.non_persistent_save_interval = None
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.train_state.step = 1000  # Set to integer for comparisons
        mock_state.train_state.floating_point_operations_so_far = 50000

        # Create mock models (2 chunks for pipeline parallelism)
        mock_model = [Mock(), Mock()]
        mock_model[0].load_state_dict = Mock()
        mock_model[1].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.bridge.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.bridge.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.bridge.training.checkpointing.update_num_microbatches"),
            patch("megatron.bridge.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
            patch("megatron.bridge.training.checkpointing.restore_sharded_modelopt_state"),
            patch("megatron.core.mpu.set_virtual_pipeline_model_parallel_rank"),
            patch("torch.distributed.barrier"),
            patch("megatron.bridge.training.checkpointing.print_rank_0"),
            patch("megatron.bridge.training.checkpointing.read_run_config") as mock_read_run_config,
            patch("megatron.bridge.training.checkpointing.unwrap_model") as mock_unwrap_model,
            patch("megatron.bridge.training.checkpointing.mpu.get_tensor_model_parallel_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.mpu.get_tensor_model_parallel_world_size", return_value=1),
            patch("megatron.bridge.training.checkpointing.mpu.get_pipeline_model_parallel_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.mpu.get_pipeline_model_parallel_world_size", return_value=1),
            patch("os.path.exists") as mock_exists,
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0
            mock_unwrap_model.return_value = mock_model

            # Mock file existence - run_config.yaml exists, train_state.pt doesn't (to use read_train_state mock)
            def mock_exists_side_effect(path):
                if "run_config.yaml" in path:
                    return True  # run_config.yaml exists
                elif "train_state.pt" in path:
                    return False  # train_state.pt doesn't exist, use mock
                return False

            mock_exists.side_effect = mock_exists_side_effect

            # Mock generate_state_dict to return the full state dict (before filtering)
            mock_generate_state_dict.return_value = full_generated_state_dict

            # Mock run config for multi-model PEFT scenario
            mock_run_config = {
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
                "checkpoint": {"save_rng": True, "save_optim": True, "fully_parallel_save": True},
            }
            mock_read_run_config.return_value = mock_run_config

            _ = load_checkpoint(
                mock_state,
                mock_model,
                None,  # No optimizer
                None,  # No scheduler
                strict=True,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )

            # Verify filtering was called once with the complete state dict
            mock_filter.assert_called_once_with(full_generated_state_dict, mock_cfg.peft)

            # Verify both models had load_state_dict called with filtered dicts and strict=False
            mock_model[0].load_state_dict.assert_called_once_with(filtered_sharded_state_dict["model0"], strict=False)
            mock_model[1].load_state_dict.assert_called_once_with(filtered_sharded_state_dict["model1"], strict=False)


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

        from megatron.bridge.training.initialize import _set_random_seed

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

        # Create minimal GPT model provider for testing
        model_provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=64,
            vocab_size=256,
            ffn_hidden_size=256,
        )

        # Create LoRA PEFT config
        lora_config = LoRA(
            target_modules=["linear_qkv", "linear_proj"],
            dim=8,
            alpha=16,
            dropout=0.1,
        )

        return model_provider, lora_config

    def _create_lora_pre_wrap_hook(self, lora_config: LoRA):
        """Create a pre-wrap hook that applies LoRA to the model.

        Args:
            lora_config: LoRA configuration instance

        Returns:
            A callable hook that can be registered with the model provider
        """

        def lora_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            """Pre-wrap hook that applies LoRA transformation.

            Args:
                model: List of base model modules before distributed wrapping

            Returns:
                List of LoRA-transformed model modules
            """
            return lora_config(model, training=True)

        return lora_pre_wrap_hook

    def test_apply_peft_adapter_filter_integration_with_peft(self, gpt_model_and_config):
        """Test apply_peft_adapter_filter_to_state_dict with real GPT model and LoRA PEFT."""
        model_provider, lora_config = gpt_model_and_config

        # Register LoRA pre-wrap hook and get model with PEFT applied
        lora_hook = self._create_lora_pre_wrap_hook(lora_config)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()
        peft_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        # Verify we got Megatron modules
        assert isinstance(peft_model, list)
        assert len(peft_model) > 0
        assert all(isinstance(chunk, MegatronModule) for chunk in peft_model)

        # Move to CUDA
        peft_model = [chunk.cuda() for chunk in peft_model]

        # Set up params_to_save for the PEFT config
        lora_config.set_params_to_save(peft_model)

        # Create a complete state dict as it would appear in checkpointing
        complete_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": peft_model[0].sharded_state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        # Filter for adapter parameters only using the main function
        filtered_state_dict = apply_peft_adapter_filter_to_state_dict(complete_state_dict, lora_config)

        # Verify filtering worked on the model section
        assert len(filtered_state_dict["model"]) < len(complete_state_dict["model"]), (
            f"Filtered model state dict ({len(filtered_state_dict['model'])}) should be smaller than "
            f"full model state dict ({len(complete_state_dict['model'])})"
        )

        # Verify only adapter parameters are in filtered model state dict
        for param_name in filtered_state_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should not be in filtered state dict"
            )

        # Verify some adapter parameters were found
        assert len(filtered_state_dict["model"]) > 0, "No adapter parameters found in filtered state dict"

        # Check that adapter parameters have expected naming patterns
        adapter_param_names = list(filtered_state_dict["model"].keys())
        has_lora_params = any("lora" in name.lower() or "adapter" in name.lower() for name in adapter_param_names)

        assert has_lora_params, f"Expected LoRA or adapter parameters in {adapter_param_names}"

        # Verify non-model sections are preserved
        assert filtered_state_dict["checkpoint_version"] == 3.0
        assert filtered_state_dict["iteration"] == 1000
        assert "optimizer" in filtered_state_dict
        assert "rng_state" in filtered_state_dict

    def test_apply_peft_adapter_filter_integration(self, gpt_model_and_config):
        """Test apply_peft_adapter_filter_to_state_dict with real model state dict."""
        model_provider, lora_config = gpt_model_and_config

        # Register LoRA pre-wrap hook and get model with PEFT applied
        lora_hook = self._create_lora_pre_wrap_hook(lora_config)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()
        peft_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        peft_model = [chunk.cuda() for chunk in peft_model]
        lora_config.set_params_to_save(peft_model)

        # Create a realistic complete state dict
        complete_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": peft_model[0].sharded_state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        # Apply PEFT filtering
        filtered_dict = apply_peft_adapter_filter_to_state_dict(complete_state_dict, lora_config)

        # Verify metadata is preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict
        assert "rng_state" in filtered_dict

        # Verify model state is filtered
        original_model_param_count = len(complete_state_dict["model"])
        filtered_model_param_count = len(filtered_dict["model"])

        assert filtered_model_param_count < original_model_param_count, (
            f"Expected filtering to reduce parameters from {original_model_param_count} "
            f"to fewer, but got {filtered_model_param_count}"
        )

        # Verify only adapter parameters remain in model
        for param_name in filtered_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should not be in filtered model state dict"
            )

    def test_adapter_filtering_with_distributed_model(self, gpt_model_and_config):
        """Test that adapter filtering works with distributed models (DDP/FSDP wrapped)."""
        model_provider, lora_config = gpt_model_and_config

        # Register LoRA pre-wrap hook
        lora_hook = self._create_lora_pre_wrap_hook(lora_config)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()

        # Create DDP config
        ddp_config = DistributedDataParallelConfig()

        # Get the model with distributed wrappers (DDP) and PEFT applied via hook
        distributed_model = model_provider.provide_distributed_model(
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
            use_torch_fsdp2=False,
            wrap_with_ddp=True,
            data_parallel_random_init=False,
        )
        distributed_model = [chunk.cuda() for chunk in distributed_model]
        lora_config.set_params_to_save(distributed_model)

        # Verify the model is wrapped with DDP
        assert len(distributed_model) > 0

        # Create a complete state dict from the distributed model
        complete_distributed_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": distributed_model[0].sharded_state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        # Test apply_peft_adapter_filter_to_state_dict with distributed model
        filtered_state_dict = apply_peft_adapter_filter_to_state_dict(complete_distributed_state_dict, lora_config)

        # Verify filtering worked
        assert len(filtered_state_dict["model"]) > 0, "Should find adapter parameters in distributed model"

        # Verify only adapter parameters are in filtered state dict
        for param_name in filtered_state_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should be an adapter parameter"
            )

        # Check that adapter parameters have expected naming patterns
        adapter_param_names = list(filtered_state_dict["model"].keys())
        has_lora_params = any("lora" in name.lower() or "adapter" in name.lower() for name in adapter_param_names)
        assert has_lora_params, f"Expected LoRA or adapter parameters in {adapter_param_names}"

        # Verify metadata is preserved in the filtered result
        assert filtered_state_dict["checkpoint_version"] == 3.0
        assert filtered_state_dict["iteration"] == 1000
        assert "optimizer" in filtered_state_dict
        assert "rng_state" in filtered_state_dict

        # Verify model state is filtered correctly
        original_model_param_count = len(complete_distributed_state_dict["model"])
        filtered_model_param_count = len(filtered_state_dict["model"])

        assert filtered_model_param_count < original_model_param_count, (
            f"Expected filtering to reduce parameters from {original_model_param_count} "
            f"to fewer, but got {filtered_model_param_count}"
        )

        # Verify only adapter parameters remain in model state
        for param_name in filtered_state_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should be an adapter parameter in filtered distributed model state dict"
            )


class TestPEFTCheckpointingValidation:
    """Simple validation tests to ensure test infrastructure works correctly."""

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
