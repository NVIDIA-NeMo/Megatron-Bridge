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
Unit tests for DoRA PEFT main class.

Tests DoRA parameter-efficient fine-tuning implementation.
"""

import datetime
import os
from unittest.mock import Mock, patch

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.dora_layers import DoRALinear, ParallelLinearDoRAAdapter
from tests.unit_tests.peft.test_utils import MockModelParallelConfig


class SimpleModel(nn.Module):
    """Simple test model with various linear layers."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 512)
        self.linear_qkv = nn.Linear(512, 1536)  # Should be matched
        self.linear_proj = nn.Linear(512, 512)  # Should be matched
        self.linear_fc1 = nn.Linear(512, 2048)  # Should be matched
        self.linear_fc2 = nn.Linear(2048, 512)  # Should be matched
        self.output_projection = nn.Linear(512, 1000)  # Should NOT be matched (not in target_modules)
        self.layernorm = nn.LayerNorm(512)

        # Add mock config attributes to linear modules for DoRA compatibility
        for module in (self.linear_qkv, self.linear_proj, self.linear_fc1, self.linear_fc2, self.output_projection):
            module.config = MockModelParallelConfig()
            module.config.sequence_parallel = False


class TestDoRA:
    """Test the DoRA PEFT class."""

    def test_dora_initialization_defaults(self):
        """Test DoRA initialization with default parameters."""
        dora = DoRA()

        assert dora.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        assert dora.dim == 32
        assert dora.alpha == 64
        assert dora.dropout == 0.0
        assert dora.dropout_position == "pre"
        assert dora.lora_A_init_method == "xavier"
        assert dora.lora_B_init_method == "zero"

    def test_dora_initialization_custom(self):
        """Test DoRA initialization with custom parameters."""
        dora = DoRA(
            target_modules=["linear_qkv", "linear_proj"],
            dim=16,
            alpha=32,
            dropout=0.1,
            dropout_position="pre",
            lora_A_init_method="uniform",
            lora_B_init_method="xavier",
        )

        assert dora.target_modules == ["linear_qkv", "linear_proj"]
        assert dora.dim == 16
        assert dora.alpha == 32
        assert dora.dropout == 0.1
        assert dora.dropout_position == "pre"
        assert dora.lora_A_init_method == "uniform"
        assert dora.lora_B_init_method == "xavier"

    def test_post_init_dropout_position_assertion(self):
        """Test that DoRA raises assertion error for invalid dropout position."""
        with pytest.raises(AssertionError, match="DoRA only supports pre-adapter dropout"):
            DoRA(dropout_position="post")

    def test_post_init_valid_dropout_position(self):
        """Test that DoRA accepts valid dropout position."""
        dora = DoRA(dropout_position="pre")
        # Should not raise any exception
        assert dora.dropout_position == "pre"

    def test_inheritance(self):
        """Test that DoRA inherits from PEFT and ModuleMatcher."""
        from megatron.bridge.peft.base import PEFT
        from megatron.bridge.peft.module_matcher import ModuleMatcher

        dora = DoRA()
        assert isinstance(dora, PEFT)
        assert isinstance(dora, ModuleMatcher)

    @patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear")
    @patch("megatron.bridge.peft.dora.logger")
    def test_transform_matched_module(
        self, mock_logger, mock_get_attributes, mock_row_linear, mock_col_linear, mock_gather
    ):
        """Test transform method when module matches target."""
        # Set up mocks for parallel linear layers with proper weight attributes
        mock_linear_in = Mock()
        mock_linear_out = Mock()

        # Create actual tensors and ensure they're accessible as .weight
        linear_in_weight = torch.randn(32, 512)  # dim=32, in_features=512
        linear_out_weight = torch.randn(256, 32)  # out_features=256, dim=32

        # Use configure_mock to ensure .weight returns actual tensors
        mock_linear_in.configure_mock(**{"weight": linear_in_weight})
        mock_linear_out.configure_mock(**{"weight": linear_out_weight})

        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        # Mock the gather function to avoid parallel state initialization
        # The gather function receives (512, 32) and should return (512, 32)
        mock_gather.return_value = torch.randn(512, 32)

        # Set up mocks
        mock_get_attributes.return_value = (
            False,
            512,
            256,
            False,
            True,
            True,
        )  # input_is_parallel, in_features, out_features, disable_tp_comm, disable_sp_comm, base_linear_is_parallel

        # Create test module with config
        test_module = nn.Linear(512, 256)
        test_module.config = MockModelParallelConfig()
        test_module.config.sequence_parallel = False

        # Create DoRA instance
        dora = DoRA(target_modules=["linear_test"])

        # Mock the match method to return a match
        with patch.object(dora, "match", return_value=("linear_test", "test.linear_test")):
            # Patch _get_weight_norm to avoid complex mock interactions
            with patch("megatron.bridge.peft.dora_layers.DoRALinear._get_weight_norm", return_value=torch.randn(256)):
                result = dora.transform(test_module, name="linear_test", prefix="test")

            # Verify DoRALinear was created
            assert isinstance(result, DoRALinear)
            assert result.to_wrap is test_module
            assert isinstance(result.adapter, ParallelLinearDoRAAdapter)

            # Verify logging
            mock_logger.info.assert_called_once_with("Adding DoRA to: test.linear_test")

            # Verify adapter attributes were retrieved
            mock_get_attributes.assert_called_once_with(test_module)

    def test_transform_unmatched_module(self):
        """Test transform method when module doesn't match target."""
        test_module = nn.Linear(512, 256)
        dora = DoRA(target_modules=["linear_qkv"])

        # Mock the match method to return None (no match)
        with patch.object(dora, "match", return_value=None):
            result = dora.transform(test_module, name="linear_other", prefix="test")

            # Should return the original module unchanged
            assert result is test_module

    @patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear")
    def test_transform_adapter_parameters(self, mock_get_attributes, mock_row_linear, mock_col_linear, mock_gather):
        """Test that transform creates adapter with correct parameters."""
        # Set up mocks for parallel linear layers with proper weight attributes
        mock_linear_in = Mock()
        mock_linear_out = Mock()

        # Create actual tensors and ensure they're accessible as .weight
        linear_in_weight = torch.randn(16, 256)  # dim=16, in_features=256
        linear_out_weight = torch.randn(128, 16)  # out_features=128, dim=16

        # Use configure_mock to ensure .weight returns actual tensors
        mock_linear_in.configure_mock(**{"weight": linear_in_weight})
        mock_linear_out.configure_mock(**{"weight": linear_out_weight})

        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        # Mock the gather function to avoid requiring parallel state initialization
        # When input_is_parallel=True, gather is called on linear_out.weight.T (which is 16x128)
        # and should return a tensor that when transposed gives the right shape for matrix mult
        def mock_gather_func(tensor):
            # Return the input tensor as-is for this test
            return tensor

        mock_gather.side_effect = mock_gather_func

        mock_get_attributes.return_value = (True, 256, 128, False, False, True)

        test_module = nn.Linear(256, 128)
        test_module.config = MockModelParallelConfig()
        test_module.config.sequence_parallel = True

        dora = DoRA(
            dim=16,
            alpha=32,
            dropout=0.2,
            dropout_position="pre",
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
        )

        with patch.object(dora, "match", return_value=("linear_test", "test.linear_test")):
            # Patch _get_weight_norm to avoid complex mock interactions
            with patch("megatron.bridge.peft.dora_layers.DoRALinear._get_weight_norm", return_value=torch.randn(128)):
                result = dora.transform(test_module, name="linear_test", prefix="test")

                adapter = result.adapter
                assert adapter.dim == 16
                assert adapter.alpha == 32
                assert adapter.dropout.p == 0.2  # dropout is a Dropout module, check .p attribute
                assert adapter.dropout_position == "pre"
                assert adapter.input_is_parallel == True
                assert adapter.disable_sequence_parallel_comm == False

    @patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear")
    def test_transform_with_simple_model(self, mock_get_attributes, mock_row_linear, mock_col_linear, mock_gather):
        """Test transform application to a simple model."""
        # Set up mocks for parallel linear layers
        mock_linear_in = Mock()
        mock_linear_out = Mock()

        # Create actual tensors for weight attributes
        mock_linear_in.configure_mock(**{"weight": torch.randn(32, 512)})  # Default dim=32
        mock_linear_out.configure_mock(**{"weight": torch.randn(1536, 32)})  # QKV output features=1536

        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out] * 10  # Repeat for multiple calls
        mock_gather.return_value = torch.randn(512, 32)

        # Mock get_adapter_attributes_from_linear to return appropriate values
        def mock_get_attributes_func(module):
            return (
                False,
                module.in_features,
                module.out_features,
                False,
                not module.config.sequence_parallel,
                True,
            )

        mock_get_attributes.side_effect = mock_get_attributes_func

        model = SimpleModel()
        dora = DoRA(target_modules=["linear_qkv", "linear_proj"])

        # Patch _get_weight_norm to avoid complex mock interactions
        with patch("megatron.bridge.peft.dora_layers.DoRALinear._get_weight_norm") as mock_norm:
            mock_norm.return_value = torch.randn(1536)  # Match QKV output features

            # Apply DoRA transform to specific modules
            model.linear_qkv = dora.transform(model.linear_qkv, name="linear_qkv", prefix="model")

            # Update mock for different output dimensions
            mock_linear_out.configure_mock(**{"weight": torch.randn(512, 32)})  # Proj output features=512
            mock_norm.return_value = torch.randn(512)  # Match proj output features

            model.linear_proj = dora.transform(model.linear_proj, name="linear_proj", prefix="model")

        # Check that target modules were transformed
        assert isinstance(model.linear_qkv, DoRALinear)
        assert isinstance(model.linear_proj, DoRALinear)

        # Check that non-target modules were not transformed
        assert isinstance(model.linear_fc1, nn.Linear)
        assert isinstance(model.linear_fc2, nn.Linear)
        assert isinstance(model.output_projection, nn.Linear)

    @patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear")
    def test_full_model_application(self, mock_get_attributes, mock_row_linear, mock_col_linear, mock_gather):
        """Test applying DoRA to a full model using the PEFT interface."""
        # Set up mocks for parallel linear layers
        mock_linear_in = Mock()
        mock_linear_out = Mock()

        # Create actual tensors for weight attributes
        mock_linear_in.configure_mock(**{"weight": torch.randn(32, 512)})  # Default dim=32
        mock_linear_out.configure_mock(**{"weight": torch.randn(512, 32)})  # General output features

        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out] * 10  # Repeat for multiple calls
        mock_gather.return_value = torch.randn(512, 32)

        # Mock get_adapter_attributes_from_linear to return appropriate values
        def mock_get_attributes_func(module):
            return (
                False,
                module.in_features,
                module.out_features,
                False,
                not module.config.sequence_parallel,
                True,
            )

        mock_get_attributes.side_effect = mock_get_attributes_func

        model = SimpleModel()
        dora = DoRA(target_modules=["linear_qkv", "linear_proj"])

        # Patch _get_weight_norm to avoid complex mock interactions
        with patch("megatron.bridge.peft.dora_layers.DoRALinear._get_weight_norm") as mock_norm:
            mock_norm.return_value = torch.randn(1536)  # Will be updated per module

            # Apply DoRA to the entire model
            adapted_model = dora(model, training=True)

        # Model should be the same object but with transformed modules
        assert adapted_model is model

        # Check model is in training mode
        assert model.training

    @patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear")
    def test_wildcard_matching(self, mock_get_attributes, mock_row_linear, mock_col_linear, mock_gather):
        """Test DoRA with wildcard pattern matching."""
        # Set up mocks for parallel linear layers
        mock_linear_in = Mock()
        mock_linear_out = Mock()

        # Create actual tensors for weight attributes
        mock_linear_in.configure_mock(**{"weight": torch.randn(32, 10)})  # Default dim=32, in_features=10
        mock_linear_out.configure_mock(**{"weight": torch.randn(10, 32)})  # out_features=10, dim=32

        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_gather.return_value = torch.randn(10, 32)

        dora = DoRA(target_modules=["*.linear_qkv"])

        # Test matching
        test_module = nn.Linear(10, 10)
        test_module.config = Mock()
        test_module.config.sequence_parallel = True

        # Should match with wildcard
        with patch(
            "megatron.bridge.peft.dora.get_adapter_attributes_from_linear",
            return_value=(False, 10, 10, False, False, True),
        ):
            with patch("megatron.bridge.peft.dora_layers.DoRALinear._get_weight_norm", return_value=torch.randn(10)):
                result = dora.transform(test_module, name="linear_qkv", prefix="layer.0.attention")
                assert isinstance(result, DoRALinear)

    def test_exclude_modules_functionality(self):
        """Test DoRA with exclude_modules parameter."""
        # Test that exclude_modules works (inherited from ModuleMatcher)
        dora = DoRA(target_modules=[], exclude_modules=["linear_proj"])

        # This functionality should be available through ModuleMatcher
        assert hasattr(dora, "exclude_modules")

    def test_dora_transform_idempotent(self):
        """Test that DoRA transform is idempotent (applying twice has same effect as applying once)."""
        test_module = nn.Linear(512, 256)
        test_module.config = MockModelParallelConfig()
        test_module.config.sequence_parallel = False

        dora = DoRA(target_modules=["linear_test"], dim=8, alpha=16)

        # Mock all the necessary functions for DoRA transform
        with patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear") as mock_get_attrs:
            mock_get_attrs.return_value = (False, 512, 256, False, True, True)

            with patch("megatron.bridge.peft.utils.ColumnParallelLinear") as mock_col_linear:
                # Create mocks for the adapters that will be created
                mock_linear_in = Mock()
                mock_linear_out = Mock()
                mock_linear_in.configure_mock(**{"weight": torch.randn(8, 512)})
                mock_linear_out.configure_mock(**{"weight": torch.randn(256, 8)})
                mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

                with patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region") as mock_gather:
                    mock_gather.return_value = torch.randn(512, 8)

                    with patch("megatron.bridge.peft.dora_layers.DoRALinear._get_weight_norm") as mock_norm:
                        mock_norm.return_value = torch.randn(256)

                        with patch.object(dora, "match", return_value=("linear_test", "test.linear_test")):
                            # Apply DoRA first time
                            first_transform = dora.transform(test_module, name="linear_test", prefix="test")

                            # Verify first transformation worked
                            assert isinstance(first_transform, DoRALinear)
                            assert first_transform.to_wrap is test_module

                            # Apply DoRA second time to the already-transformed module
                            second_transform = dora.transform(first_transform, name="linear_test", prefix="test")

                            # Verify idempotency: second transformation should return identical object
                            assert second_transform is first_transform
                            assert isinstance(second_transform, DoRALinear)

    def test_dora_transform_idempotent_full_model(self):
        """Test that DoRA transform is idempotent when applied to a full model."""
        model = SimpleModel()
        dora = DoRA(target_modules=["linear_qkv", "linear_proj"], dim=8, alpha=16)

        # Mock all the necessary functions for DoRA transform
        with patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear") as mock_get_attrs:

            def mock_get_attributes_func(module):
                return (
                    False,
                    module.in_features,
                    module.out_features,
                    False,
                    not module.config.sequence_parallel,
                    True,
                )

            mock_get_attrs.side_effect = mock_get_attributes_func

            with patch("megatron.bridge.peft.utils.ColumnParallelLinear") as mock_col_linear:
                # Create mocks for the adapters that will be created
                mock_linear_in = Mock()
                mock_linear_out = Mock()
                mock_linear_in.configure_mock(**{"weight": torch.randn(8, 512)})
                mock_linear_out.configure_mock(**{"weight": torch.randn(512, 8)})
                mock_col_linear.side_effect = [mock_linear_in, mock_linear_out] * 10  # Multiple calls

                with patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region") as mock_gather:
                    mock_gather.return_value = torch.randn(512, 8)

                    with patch("megatron.bridge.peft.dora_layers.DoRALinear._get_weight_norm") as mock_norm:
                        mock_norm.return_value = torch.randn(512)  # Will be updated per module

                        # Apply DoRA first time
                        first_transform = dora(model, training=True)

                        # Store references to transformed modules
                        first_linear_qkv = first_transform.linear_qkv
                        first_linear_proj = first_transform.linear_proj
                        first_linear_fc1 = first_transform.linear_fc1  # Should remain unchanged

                        # Verify first transformation worked
                        assert isinstance(first_linear_qkv, DoRALinear)
                        assert isinstance(first_linear_proj, DoRALinear)
                        assert isinstance(first_linear_fc1, nn.Linear)  # Not targeted

                        # Apply DoRA second time to the already-transformed model
                        second_transform = dora(first_transform, training=True)

                        # Verify idempotency: second transformation should return identical objects
                        assert second_transform.linear_qkv is first_linear_qkv
                        assert second_transform.linear_proj is first_linear_proj
                        assert second_transform.linear_fc1 is first_linear_fc1

                        # Verify the module types are still correct
                        assert isinstance(second_transform.linear_qkv, DoRALinear)
                        assert isinstance(second_transform.linear_proj, DoRALinear)
                        assert isinstance(second_transform.linear_fc1, nn.Linear)


class TestDoRAMegatronIntegration:
    """Test DoRA integration with Megatron models (requires GPU)."""

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
                for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    def _create_dora_pre_wrap_hook(self, dora_config: DoRA):
        """Create a pre-wrap hook that applies DoRA to the model.

        Args:
            dora_config: DoRA configuration instance

        Returns:
            A callable hook that can be registered with the model provider
        """

        def dora_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            """Pre-wrap hook that applies DoRA transformation.

            Args:
                model: List of base model modules before distributed wrapping

            Returns:
                List of DoRA-transformed model modules
            """
            return dora_config(model, training=True)

        return dora_pre_wrap_hook

    def test_dora_with_gpt_model(self):
        """Test DoRA application to a real GPT model using pre-wrap hooks."""
        # Create a minimal GPT configuration
        model_provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            vocab_size=1000,
            ffn_hidden_size=256,
        )

        # Create DoRA instance targeting linear layers
        dora = DoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"], dim=8, alpha=16, dropout=0.0
        )

        # Register DoRA pre-wrap hook
        dora_hook = self._create_dora_pre_wrap_hook(dora)
        model_provider.register_pre_wrap_hook(dora_hook)
        model_provider.finalize()

        # Get the model with DoRA applied via hook
        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        # Verify we got a list of Megatron modules
        assert isinstance(adapted_model, list)
        assert len(adapted_model) > 0
        assert all(isinstance(chunk, MegatronModule) for chunk in adapted_model)

        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Verify that DoRA was applied to target modules
        found_dora_modules = []
        for chunk in adapted_model:
            for name, module in chunk.named_modules():
                if isinstance(module, DoRALinear):
                    found_dora_modules.append(name)

        # Should find some DoRA modules
        assert len(found_dora_modules) > 0

        # Verify model is in training mode
        for chunk in adapted_model:
            assert chunk.training

    def test_dora_parameter_counting(self):
        """Test that DoRA adds the expected number of parameters using pre-wrap hooks."""
        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=100,
            ffn_hidden_size=128,
        )
        model_provider.finalize()

        # Get base model first to count original parameters
        base_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        base_model = [chunk.cuda() for chunk in base_model]

        # Count original parameters
        original_params = sum(p.numel() for chunk in base_model for p in chunk.parameters())

        # Create fresh model provider for DoRA application
        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=100,
            ffn_hidden_size=128,
        )

        # Create DoRA and register hook
        dora = DoRA(target_modules=["linear_qkv"], dim=4, alpha=8)
        dora_hook = self._create_dora_pre_wrap_hook(dora)
        model_provider.register_pre_wrap_hook(dora_hook)
        model_provider.finalize()

        # Get DoRA-adapted model using hook
        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Count parameters after DoRA
        adapted_params = sum(p.numel() for chunk in adapted_model for p in chunk.parameters())

        # DoRA should add parameters (low-rank matrices + magnitude vectors)
        assert adapted_params > original_params

    def test_dora_transform_idempotent_megatron_model(self):
        """Test that DoRA transform is idempotent when applied via pre-wrap hooks."""
        # Create a minimal GPT configuration
        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=100,
            ffn_hidden_size=128,
        )

        # Create DoRA instance
        dora = DoRA(target_modules=["linear_qkv", "linear_proj"], dim=4, alpha=8)

        # Register hook and apply DoRA first time
        dora_hook = self._create_dora_pre_wrap_hook(dora)
        model_provider.register_pre_wrap_hook(dora_hook)
        model_provider.finalize()
        first_transform = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        first_transform = [chunk.cuda() for chunk in first_transform]

        # Store references to the transformed model chunks
        first_chunks = [chunk for chunk in first_transform]

        # Verify we got DoRA modules in the first transformation
        found_dora_modules_first = []
        for chunk in first_transform:
            for name, module in chunk.named_modules():
                if isinstance(module, DoRALinear):
                    found_dora_modules_first.append((chunk, name, module))

        assert len(found_dora_modules_first) > 0, "Should have found DoRA modules in first transformation"

        # Apply DoRA second time to the already-transformed model
        # Note: In the pre-wrap hook pattern, we need to apply DoRA directly since
        # the model provider has already been called
        second_transform = dora(first_transform, training=True)

        # Verify idempotency: should return the same model chunks
        assert len(second_transform) == len(first_transform)
        for i, (first_chunk, second_chunk) in enumerate(zip(first_chunks, second_transform)):
            assert second_chunk is first_chunk, f"Chunk {i} should be identical object"

        # Verify DoRA modules are identical objects
        found_dora_modules_second = []
        for chunk in second_transform:
            for name, module in chunk.named_modules():
                if isinstance(module, DoRALinear):
                    found_dora_modules_second.append((chunk, name, module))

        # Should have same number of DoRA modules
        assert len(found_dora_modules_second) == len(found_dora_modules_first)

        # Each DoRA module should be the identical object
        for (first_chunk, first_name, first_module), (second_chunk, second_name, second_module) in zip(
            found_dora_modules_first, found_dora_modules_second
        ):
            assert first_chunk is second_chunk
            assert first_name == second_name
            assert second_module is first_module, f"DoRA module {first_name} should be identical object"

    @pytest.mark.pleasefixme  # This test is too slow for unit tests (>0.5s)
    def test_dora_forward_pass(self):
        """Test that DoRA adapted model can perform forward pass using pre-wrap hooks."""
        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=128,
            ffn_hidden_size=128,
        )

        # Create DoRA and register hook
        dora = DoRA(target_modules=["linear_qkv", "linear_proj"], dim=4, alpha=8)
        dora_hook = self._create_dora_pre_wrap_hook(dora)
        model_provider.register_pre_wrap_hook(dora_hook)
        model_provider.finalize()

        # Get DoRA-adapted model using hook
        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Test forward pass with proper Megatron input format
        batch_size, seq_len = 2, 8

        # Get model device (model is on CUDA, inputs need to match)
        model_device = next(adapted_model[0].parameters()).device

        # Create input tensors in the format expected by Megatron models
        input_ids = torch.randint(0, model_provider.vocab_size, (batch_size, seq_len), device=model_device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=model_device).unsqueeze(0).expand(batch_size, -1)

        # Create 4D causal attention mask [batch_size, 1, seq_len, seq_len]
        # True values are masked out (don't attend), False values attend
        attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=model_device)) < 0.5
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # Run forward pass using the standard codebase pattern
        forward_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            for chunk in adapted_model:
                output = chunk(**forward_args)

                # Verify output shape and that DoRA is active
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                expected_shape = (batch_size, seq_len, model_provider.vocab_size)
                assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

                # Count DoRA adaptations
                dora_count = sum(1 for _, m in chunk.named_modules() if isinstance(m, DoRALinear))
                assert dora_count > 0, "Should have DoRA adaptations applied"

    def test_dora_different_targets(self):
        """Test DoRA with different target module configurations using pre-wrap hooks."""

        # Test different target configurations
        target_configs = [
            ["linear_qkv"],
            ["linear_proj"],
            ["linear_fc1", "linear_fc2"],
            ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        ]

        for targets in target_configs:
            # Create fresh model provider for each configuration
            model_provider = GPTModelProvider(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=2,
                vocab_size=100,
                ffn_hidden_size=128,
            )

            # Create DoRA and register hook
            dora = DoRA(target_modules=targets, dim=4, alpha=8)
            dora_hook = self._create_dora_pre_wrap_hook(dora)
            model_provider.register_pre_wrap_hook(dora_hook)
            model_provider.finalize()

            # Get adapted model using hook
            adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
            adapted_model = [chunk.cuda() for chunk in adapted_model]

            # Count DoRA modules
            dora_count = sum(
                1 for chunk in adapted_model for _, module in chunk.named_modules() if isinstance(module, DoRALinear)
            )

            # Should find some DoRA modules for each configuration
            assert dora_count > 0
