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

"""Unit tests for megatron.bridge.training.mixed_precision module."""

from dataclasses import dataclass, fields
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.t5_provider import T5ModelProvider
from megatron.bridge.training.config import DistributedDataParallelConfig, OptimizerConfig
from megatron.bridge.training.mixed_precision import (
    MixedPrecisionConfig,
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_delayed_scaling_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
    fp16_mixed,
    fp16_with_fp8_current_scaling_mixed,
    fp16_with_fp8_delayed_scaling_mixed,
    fp16_with_fp8_subchannel_scaling_mixed,
    fp16_with_mxfp8_mixed,
    get_mixed_precision_config,
    nemotron_h_bf16_with_fp8_current_scaling_mixed,
    update_config_with_precision_overrides,
)


class TestMegatronMixedPrecisionConfig:
    def test_fp8_configurations(self):
        config = MixedPrecisionConfig(
            fp8="e5m2",
            fp8_recipe="mxfp8",
            fp8_margin=1,
            fp8_amax_history_len=24,
            fp8_amax_compute_algo="max",
            fp8_wgrad=False,
            fp8_dot_product_attention=True,
            fp8_multi_head_attention=True,
            fp8_param=False,
            fp8_param_gather=False,
        )

        assert config.fp8 == "e5m2"
        assert config.fp8_recipe == "mxfp8"
        assert config.fp8_margin == 1
        assert config.fp8_amax_history_len == 24
        assert config.fp8_amax_compute_algo == "max"
        assert config.fp8_wgrad is False
        assert config.fp8_dot_product_attention is True
        assert config.fp8_multi_head_attention is True
        assert config.fp8_param is False
        assert config.fp8_param_gather is False

    def test_fp8_param_initialization_from_fp8_param_gather(self):
        """Test that fp8_param is initialized from fp8_param_gather when None."""
        # Test with fp8_param_gather=True
        config = MixedPrecisionConfig(fp8_param=None, fp8_param_gather=True)
        assert config.fp8_param is True
        assert config.fp8_param_gather is True

        # Test with fp8_param_gather=False
        config = MixedPrecisionConfig(fp8_param=None, fp8_param_gather=False)
        assert config.fp8_param is False
        assert config.fp8_param_gather is False

        # Test default behavior (both should default to False)
        config = MixedPrecisionConfig()
        assert config.fp8_param is False
        assert config.fp8_param_gather is False

    def test_fp8_param_synchronization(self):
        """Test that fp8_param and fp8_param_gather stay synchronized."""
        # Test initial synchronization
        config = MixedPrecisionConfig(fp8_param_gather=True)
        assert config.fp8_param is True
        assert config.fp8_param_gather is True

        # Test changing fp8_param_gather updates fp8_param
        config.fp8_param_gather = False
        assert config.fp8_param is False
        assert config.fp8_param_gather is False

        # Test changing fp8_param updates fp8_param_gather
        config.fp8_param = True
        assert config.fp8_param is True
        assert config.fp8_param_gather is True

        # Test with explicit values during initialization
        config2 = MixedPrecisionConfig(fp8_param=True, fp8_param_gather=False)
        # After initialization, they should be synchronized (fp8_param_gather wins as it's set last)
        assert config2.fp8_param is False
        assert config2.fp8_param_gather is False

        # Test None initialization
        config3 = MixedPrecisionConfig(fp8_param=None, fp8_param_gather=True)
        assert config3.fp8_param is True
        assert config3.fp8_param_gather is True

    def test_mxfp8_param_gather_validation(self):
        """Test that mxfp8 recipe with fp8_param_gather=True requires reuse_grad_buf_for_mxfp8_param_ag=True."""
        # Valid configuration: fp8_param_gather=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=True
        config_valid = MixedPrecisionConfig(
            fp8_param_gather=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=True
        )
        # Should not raise any assertion error
        assert config_valid.fp8_param_gather is True
        assert config_valid.fp8_recipe == "mxfp8"
        assert config_valid.reuse_grad_buf_for_mxfp8_param_ag is True

        # Invalid configuration: fp8_param_gather=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=False
        with pytest.raises(AssertionError, match="When fp8_param_gather=True and fp8_recipe='mxfp8'"):
            config_invalid = MixedPrecisionConfig(
                fp8_param_gather=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=False
            )
            config_invalid.finalize()

        # Valid configuration: fp8_param_gather=False with mxfp8 recipe (assertion doesn't apply)
        config_param_gather_false = MixedPrecisionConfig(
            fp8_param_gather=False, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=False
        )
        assert config_param_gather_false.fp8_param_gather is False
        assert config_param_gather_false.fp8_recipe == "mxfp8"
        assert config_param_gather_false.reuse_grad_buf_for_mxfp8_param_ag is False

        # Valid configuration: fp8_param_gather=True with non-mxfp8 recipe (assertion doesn't apply)
        config_other_recipe = MixedPrecisionConfig(
            fp8_param_gather=True, fp8_recipe="delayed", reuse_grad_buf_for_mxfp8_param_ag=False
        )
        assert config_other_recipe.fp8_param_gather is True
        assert config_other_recipe.fp8_recipe == "delayed"
        assert config_other_recipe.reuse_grad_buf_for_mxfp8_param_ag is False

    def test_mxfp8_validation_after_field_modification(self):
        """Test that the mxfp8 validation works after modifying fields and re-running finalize()."""
        # Start with a valid configuration
        config = MixedPrecisionConfig(
            fp8_param_gather=True, fp8_recipe="delayed", reuse_grad_buf_for_mxfp8_param_ag=False
        )

        # Modify to make it invalid (mxfp8 with reuse_grad_buf_for_mxfp8_param_ag=False)
        config.fp8_recipe = "mxfp8"

        # Re-running finalize() should trigger the assertion
        with pytest.raises(AssertionError, match="When fp8_param_gather=True and fp8_recipe='mxfp8'"):
            config.finalize()

        # Fix the configuration
        config.reuse_grad_buf_for_mxfp8_param_ag = True
        # This should not raise any error
        config.finalize()

    def test_fp8_param_matching_fp8_param_gather(self):
        """Test that matching values for fp8_param and fp8_param_gather work correctly."""
        # Both True
        config = MixedPrecisionConfig(fp8_param=True, fp8_param_gather=True)
        assert config.fp8_param is True
        assert config.fp8_param_gather is True

        # Both False
        config = MixedPrecisionConfig(fp8_param=False, fp8_param_gather=False)
        assert config.fp8_param is False
        assert config.fp8_param_gather is False

    @patch("logging.debug")
    def test_setup_with_gpt_config(self, mock_log):
        mixed_precision_config = MixedPrecisionConfig(
            fp16=True, bf16=False, params_dtype=torch.float16, loss_scale=1024.0
        )

        # Create mock GPTConfig with necessary attributes
        gpt_config = MagicMock(spec=GPTModelProvider)
        gpt_config.fp16 = False
        gpt_config.bf16 = True
        gpt_config.params_dtype = torch.float32
        gpt_config.loss_scale = None

        # Call setup
        mixed_precision_config.setup(gpt_config)

        # Verify attributes were updated
        assert gpt_config.fp16 is True
        assert gpt_config.bf16 is False
        assert gpt_config.params_dtype == torch.float16
        assert gpt_config.loss_scale == 1024.0

        # Verify logging was called for the specific overwritten values
        debug_calls = [str(call) for call in mock_log.call_args_list]
        assert any("fp16" in call and "False -> True" in call for call in debug_calls)
        assert any("bf16" in call and "True -> False" in call for call in debug_calls)
        assert any("params_dtype" in call for call in debug_calls)
        assert any("loss_scale" in call and "None -> 1024.0" in call for call in debug_calls)

    @patch("logging.debug")
    def test_setup_with_t5_config(self, mock_log):
        mixed_precision_config = MixedPrecisionConfig(
            bf16=True, params_dtype=torch.bfloat16, autocast_enabled=True, autocast_dtype=torch.bfloat16
        )

        # Create mock T5Config
        t5_config = MagicMock(spec=T5ModelProvider)
        t5_config.bf16 = False
        t5_config.params_dtype = torch.float32
        t5_config.autocast_enabled = False
        t5_config.autocast_dtype = None

        # Call setup
        mixed_precision_config.setup(t5_config)

        # Verify attributes were updated
        assert t5_config.bf16 is True
        assert t5_config.params_dtype == torch.bfloat16
        assert t5_config.autocast_enabled is True
        assert t5_config.autocast_dtype == torch.bfloat16

    @patch("logging.debug")
    def test_setup_with_optimizer_config(self, mock_log):
        mixed_precision_config = MixedPrecisionConfig(
            grad_reduce_in_fp32=False, loss_scale=512.0, initial_loss_scale=1024.0
        )

        # Create mock configs
        model_config = MagicMock(spec=GPTModelProvider)
        optimizer_config = MagicMock(spec=OptimizerConfig)
        optimizer_config.grad_reduce_in_fp32 = True
        optimizer_config.loss_scale = None
        optimizer_config.initial_loss_scale = None

        # Call setup
        mixed_precision_config.setup(model_config, optimizer_config=optimizer_config)

        # Verify optimizer config was updated
        assert optimizer_config.grad_reduce_in_fp32 is False
        assert optimizer_config.loss_scale == 512.0
        assert optimizer_config.initial_loss_scale == 1024.0

    @patch("logging.debug")
    def test_setup_with_ddp_config(self, mock_log):
        mixed_precision_config = MixedPrecisionConfig(grad_reduce_in_fp32=False, fp16=True)

        # Create mock configs
        model_config = MagicMock(spec=GPTModelProvider)
        ddp_config = MagicMock(spec=DistributedDataParallelConfig)
        ddp_config.grad_reduce_in_fp32 = True
        ddp_config.fp16 = False

        # Call setup
        mixed_precision_config.setup(model_config, ddp_config=ddp_config)

        # Verify DDP config was updated
        assert ddp_config.grad_reduce_in_fp32 is False
        assert ddp_config.fp16 is True

    def test_setup_with_all_configs(self):
        mixed_precision_config = MixedPrecisionConfig(
            bf16=True, params_dtype=torch.bfloat16, grad_reduce_in_fp32=False
        )

        # Create mock configs
        model_config = MagicMock(spec=GPTModelProvider)
        model_config.bf16 = False
        model_config.params_dtype = torch.float32

        optimizer_config = MagicMock(spec=OptimizerConfig)
        optimizer_config.grad_reduce_in_fp32 = True

        ddp_config = MagicMock(spec=DistributedDataParallelConfig)
        ddp_config.grad_reduce_in_fp32 = True

        # Call setup
        mixed_precision_config.setup(model_config, optimizer_config, ddp_config)

        # Verify all configs were updated
        assert model_config.bf16 is True
        assert model_config.params_dtype == torch.bfloat16
        assert optimizer_config.grad_reduce_in_fp32 is False
        assert ddp_config.grad_reduce_in_fp32 is False


class TestUpdateConfigWithDtypeOverrides:
    @patch("logging.debug")
    def test_update_with_matching_fields(self, mock_log):
        mixed_precision_config = MixedPrecisionConfig(fp16=True, bf16=False, params_dtype=torch.float16)

        # Create mock config with matching attributes
        @dataclass
        class MockConfig:
            fp16: bool = False
            bf16: bool = True
            params_dtype: torch.dtype = torch.float32
            other_field: str = "unchanged"

        config = MockConfig()

        # Update config
        updated_config = update_config_with_precision_overrides(mixed_precision_config, config)

        # Verify updates
        assert updated_config.fp16 is True
        assert updated_config.bf16 is False
        assert updated_config.params_dtype == torch.float16
        assert updated_config.other_field == "unchanged"

        # Verify logging
        assert mock_log.call_count == 3

    def test_update_with_no_matching_fields(self):
        mixed_precision_config = MixedPrecisionConfig(fp16=True)

        # Create mock config with no matching attributes
        @dataclass
        class MockConfig:
            some_other_field: str = "value"
            another_field: int = 42

        config = MockConfig()

        # Update config (should not change anything)
        updated_config = update_config_with_precision_overrides(mixed_precision_config, config)

        # Verify nothing changed
        assert updated_config.some_other_field == "value"
        assert updated_config.another_field == 42

    def test_update_with_partial_matching_fields(self):
        mixed_precision_config = MixedPrecisionConfig(fp16=True, loss_scale=1024.0, fp8_margin=2)

        # Create mock config with some matching attributes
        @dataclass
        class MockConfig:
            fp16: bool = False
            loss_scale: float = None
            unrelated_field: str = "unchanged"

        config = MockConfig()

        # Update config
        updated_config = update_config_with_precision_overrides(mixed_precision_config, config)

        # Verify only matching fields were updated
        assert updated_config.fp16 is True
        assert updated_config.loss_scale == 1024.0
        assert updated_config.unrelated_field == "unchanged"

    def test_update_preserves_none_values(self):
        mixed_precision_config = MixedPrecisionConfig(params_dtype=None, loss_scale=None)

        @dataclass
        class MockConfig:
            params_dtype: torch.dtype = torch.float32
            loss_scale: float = 512.0

        config = MockConfig()

        # Update config
        updated_config = update_config_with_precision_overrides(mixed_precision_config, config)

        # Verify None values override existing values
        assert updated_config.params_dtype is None
        assert updated_config.loss_scale is None

    def test_update_returns_same_object(self):
        mixed_precision_config = MixedPrecisionConfig(fp16=True)

        @dataclass
        class MockConfig:
            fp16: bool = False

        config = MockConfig()

        # Update config
        updated_config = update_config_with_precision_overrides(mixed_precision_config, config)

        # Verify it's the same object
        assert updated_config is config


class TestIntegration:
    def test_fp16_configuration_flow(self):
        mixed_precision_config = MixedPrecisionConfig(
            fp16=True,
            params_dtype=torch.float16,
            loss_scale=1024.0,
            initial_loss_scale=2048.0,
            min_loss_scale=1.0,
            loss_scale_window=1000.0,
            hysteresis=2.0,
        )

        # Create configs
        model_config = MagicMock(spec=GPTModelProvider)
        for field in fields(mixed_precision_config):
            setattr(model_config, field.name, None)

        optimizer_config = MagicMock(spec=OptimizerConfig)
        optimizer_config.loss_scale = None
        optimizer_config.initial_loss_scale = None
        optimizer_config.min_loss_scale = None
        optimizer_config.loss_scale_window = None
        optimizer_config.hysteresis = None

        # Apply configuration
        mixed_precision_config.setup(model_config, optimizer_config)

        # Verify FP16 settings
        assert model_config.fp16 is True
        assert model_config.params_dtype == torch.float16
        assert optimizer_config.loss_scale == 1024.0
        assert optimizer_config.initial_loss_scale == 2048.0
        assert optimizer_config.min_loss_scale == 1.0
        assert optimizer_config.loss_scale_window == 1000.0
        assert optimizer_config.hysteresis == 2.0

    def test_bf16_configuration_flow(self):
        mixed_precision_config = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=2,
        )

        # Create model config
        model_config = MagicMock(spec=GPTModelProvider)
        for field in fields(mixed_precision_config):
            setattr(model_config, field.name, None)

        # Apply configuration
        mixed_precision_config.setup(model_config)

        # Verify BF16 settings
        assert model_config.bf16 is True
        assert model_config.params_dtype == torch.bfloat16
        assert model_config.autocast_enabled is True
        assert model_config.autocast_dtype == torch.bfloat16
        assert model_config.first_last_layers_bf16 is True
        assert model_config.num_layers_at_start_in_bf16 == 2
        assert model_config.num_layers_at_end_in_bf16 == 2

    def test_fp8_configuration_flow(self):
        mixed_precision_config = MixedPrecisionConfig(
            fp8="e4m3",
            fp8_recipe="delayed",
            fp8_margin=1,
            fp8_amax_history_len=24,
            fp8_amax_compute_algo="most_recent",
            fp8_wgrad=True,
            fp8_dot_product_attention=True,
            fp8_multi_head_attention=True,
            fp8_param=True,
            fp8_param_gather=True,
        )

        # Create model config
        model_config = MagicMock(spec=GPTModelProvider)
        for field in fields(mixed_precision_config):
            setattr(model_config, field.name, None)

        # Apply configuration
        mixed_precision_config.setup(model_config)

        # Verify FP8 settings
        assert model_config.fp8 == "e4m3"
        assert model_config.fp8_recipe == "delayed"
        assert model_config.fp8_margin == 1
        assert model_config.fp8_amax_history_len == 24
        assert model_config.fp8_amax_compute_algo == "most_recent"
        assert model_config.fp8_wgrad is True
        assert model_config.fp8_dot_product_attention is True
        assert model_config.fp8_multi_head_attention is True
        assert model_config.fp8_param is True
        assert model_config.fp8_param_gather is True


class TestMixedPrecisionRecipes:
    def test_bf16_mixed(self):
        config = bf16_mixed()

        assert config.bf16 is True
        assert config.fp16 is False
        assert config.params_dtype == torch.bfloat16
        assert config.pipeline_dtype == torch.bfloat16
        assert config.autocast_enabled is False
        assert config.grad_reduce_in_fp32 is True
        # Base BF16 recipe should have fp8_param as False
        assert config.fp8_param is False
        assert config.fp8_param_gather is False

    def test_fp16_mixed(self):
        config = fp16_mixed()

        assert config.fp16 is True
        assert config.bf16 is False
        assert config.params_dtype == torch.half
        assert config.pipeline_dtype == torch.half
        assert config.autocast_enabled is False
        assert config.grad_reduce_in_fp32 is False
        # Base FP16 recipe should have fp8_param as False
        assert config.fp8_param is False
        assert config.fp8_param_gather is False

    def test_bf16_with_fp8_delayed_scaling_mixed(self):
        config = bf16_with_fp8_delayed_scaling_mixed()

        # Should inherit BF16 settings
        assert config.bf16 is True
        assert config.params_dtype == torch.bfloat16
        assert config.pipeline_dtype == torch.bfloat16

        # FP8 specific settings
        assert config.fp8 == "hybrid"
        assert config.fp8_recipe == "delayed"
        assert config.fp8_margin == 0
        assert config.fp8_amax_history_len == 1024
        assert config.fp8_amax_compute_algo == "max"
        assert config.fp8_param_gather is True
        # fp8_param should now be kept in sync with fp8_param_gather
        assert config.fp8_param is True

    def test_fp16_with_fp8_delayed_scaling_mixed(self):
        config = fp16_with_fp8_delayed_scaling_mixed()

        # Should inherit FP16 settings
        assert config.fp16 is True
        assert config.params_dtype == torch.half
        assert config.pipeline_dtype == torch.half
        assert config.grad_reduce_in_fp32 is False

        # FP8 specific settings
        assert config.fp8 == "hybrid"
        assert config.fp8_recipe == "delayed"
        assert config.fp8_margin == 0
        assert config.fp8_amax_history_len == 1024
        assert config.fp8_amax_compute_algo == "max"
        assert config.fp8_param_gather is True
        # fp8_param should now be kept in sync with fp8_param_gather
        assert config.fp8_param is True

    def test_bf16_with_mxfp8_mixed(self):
        config = bf16_with_mxfp8_mixed()

        # Should inherit BF16 settings
        assert config.bf16 is True
        assert config.params_dtype == torch.bfloat16

        # MXFP8 specific settings
        assert config.fp8 == "e4m3"
        assert config.fp8_recipe == "mxfp8"
        assert config.fp8_param_gather is True
        assert config.reuse_grad_buf_for_mxfp8_param_ag is True
        # Verify fp8_param is initialized from fp8_param_gather
        assert config.fp8_param is True

    def test_fp16_with_mxfp8_mixed(self):
        config = fp16_with_mxfp8_mixed()

        # Should inherit FP16 settings
        assert config.fp16 is True
        assert config.params_dtype == torch.half

        # MXFP8 specific settings
        assert config.fp8 == "e4m3"
        assert config.fp8_recipe == "mxfp8"
        assert config.fp8_param_gather is True
        assert config.reuse_grad_buf_for_mxfp8_param_ag is True
        # Verify fp8_param is initialized from fp8_param_gather
        assert config.fp8_param is True

    def test_bf16_with_fp8_current_scaling_mixed(self):
        config = bf16_with_fp8_current_scaling_mixed()

        # Should inherit BF16 settings
        assert config.bf16 is True
        assert config.params_dtype == torch.bfloat16

        # Tensorwise scaling specific settings
        assert config.fp8 == "hybrid"
        assert config.fp8_recipe == "tensorwise"
        assert config.first_last_layers_bf16 is True
        assert config.num_layers_at_start_in_bf16 == 1
        assert config.num_layers_at_end_in_bf16 == 1
        assert config.fp8_param_gather is True
        # fp8_param should now be kept in sync with fp8_param_gather
        assert config.fp8_param is True

    def test_nemotron_h_bf16_with_fp8_current_scaling_mixed(self):
        config = nemotron_h_bf16_with_fp8_current_scaling_mixed()

        # Should inherit BF16 settings
        assert config.bf16 is True
        assert config.params_dtype == torch.bfloat16

        # Nemotron variant with more layers
        assert config.fp8 == "hybrid"
        assert config.fp8_recipe == "tensorwise"
        assert config.first_last_layers_bf16 is True
        assert config.num_layers_at_start_in_bf16 == 2
        assert config.num_layers_at_end_in_bf16 == 2
        assert config.fp8_param_gather is True
        # fp8_param should now be kept in sync with fp8_param_gather
        assert config.fp8_param is True

    def test_fp16_with_fp8_current_scaling_mixed(self):
        config = fp16_with_fp8_current_scaling_mixed()

        # Should inherit FP16 settings
        assert config.fp16 is True
        assert config.params_dtype == torch.half

        # Tensorwise scaling specific settings
        assert config.fp8 == "hybrid"
        assert config.fp8_recipe == "tensorwise"
        assert config.first_last_layers_bf16 is True
        assert config.num_layers_at_start_in_bf16 == 1
        assert config.num_layers_at_end_in_bf16 == 1
        assert config.fp8_param_gather is True
        # fp8_param should now be kept in sync with fp8_param_gather
        assert config.fp8_param is True

    def test_bf16_with_fp8_subchannel_scaling_mixed(self):
        config = bf16_with_fp8_subchannel_scaling_mixed()

        # Should inherit BF16 settings
        assert config.bf16 is True
        assert config.params_dtype == torch.bfloat16

        # Blockwise scaling specific settings
        assert config.fp8 == "hybrid"
        assert config.fp8_recipe == "blockwise"
        assert config.fp8_param_gather is False
        # Verify fp8_param is initialized from fp8_param_gather
        assert config.fp8_param is False

    def test_fp16_with_fp8_subchannel_scaling_mixed(self):
        config = fp16_with_fp8_subchannel_scaling_mixed()

        # Should inherit FP16 settings
        assert config.fp16 is True
        assert config.params_dtype == torch.half

        # Blockwise scaling specific settings
        assert config.fp8 == "hybrid"
        assert config.fp8_recipe == "blockwise"
        assert config.fp8_param_gather is False
        # Verify fp8_param is initialized from fp8_param_gather
        assert config.fp8_param is False

    def test_recipe_returns_new_instance(self):
        """Test that each recipe returns a new instance."""
        config1 = bf16_mixed()
        config2 = bf16_mixed()

        assert config1 is not config2

        # Modifying one should not affect the other
        config1.fp8 = "test"
        assert config2.fp8 is None

    def test_recipe_with_setup(self):
        """Test that recipe configs work with the setup method."""
        config = bf16_with_fp8_delayed_scaling_mixed()

        # Create mock model config
        model_config = MagicMock(spec=GPTModelProvider)
        for field in fields(config):
            setattr(model_config, field.name, None)

        # Create mock optimizer config with relevant fields
        optimizer_config = MagicMock(spec=OptimizerConfig)
        optimizer_config.grad_reduce_in_fp32 = None
        optimizer_config.loss_scale = None
        optimizer_config.initial_loss_scale = None
        optimizer_config.min_loss_scale = None
        optimizer_config.loss_scale_window = None
        optimizer_config.hysteresis = None

        # Create mock DDP config with relevant fields
        ddp_config = MagicMock(spec=DistributedDataParallelConfig)
        ddp_config.grad_reduce_in_fp32 = None
        ddp_config.fp16 = None
        ddp_config.bf16 = None
        ddp_config.fp8 = None

        # Apply configuration to all configs
        config.setup(model_config, optimizer_config, ddp_config)

        # Verify model config settings were applied
        assert model_config.bf16 is True
        assert model_config.params_dtype == torch.bfloat16
        assert model_config.fp8 == "hybrid"
        assert model_config.fp8_recipe == "delayed"
        assert model_config.grad_reduce_in_fp32 is True

        # Verify optimizer config settings were applied
        assert optimizer_config.grad_reduce_in_fp32 is True

        # Verify DDP config settings were applied
        assert ddp_config.grad_reduce_in_fp32 is True
        assert ddp_config.bf16 is True


class TestRegisterAndGetMixedPrecisionConfig:
    """Tests for the `register` decorator and `get_mixed_precision_config` helper."""

    def test_register_decorator_adds_recipe(self):
        """Ensure that the `register` decorator adds the factory function to the global registry
        and that each invocation returns a fresh `MixedPrecisionConfig` instance.
        """
        # Local import to avoid polluting global namespace before test discovery.
        from megatron.bridge.training.mixed_precision import (
            MIXED_PRECISION_RECIPES,
            MixedPrecisionConfig,
            get_mixed_precision_config,
            register,
        )

        @register  # noqa: WPS430 – intentional decorator usage inside test
        def custom_fp32_config() -> MixedPrecisionConfig:  # pylint: disable=missing-docstring
            return MixedPrecisionConfig(fp32=True)

        # The recipe should now be registered under its function name.
        assert "custom_fp32_config" in MIXED_PRECISION_RECIPES

        # Fetch two separate instances via different access paths.
        cfg_from_dict = MIXED_PRECISION_RECIPES["custom_fp32_config"]()
        cfg_from_helper = get_mixed_precision_config("custom_fp32_config")

        # They should both be `MixedPrecisionConfig` instances and distinct objects.
        assert isinstance(cfg_from_dict, MixedPrecisionConfig)
        assert isinstance(cfg_from_helper, MixedPrecisionConfig)
        assert cfg_from_dict is not cfg_from_helper
        assert cfg_from_helper.fp32 is True

    def test_get_mixed_precision_config_invalid_name(self):
        """Verify that an unknown recipe name raises a clear `ValueError`."""
        with pytest.raises(ValueError) as exc_info:
            get_mixed_precision_config("does_not_exist")

        assert "Unknown mixed-precision recipe" in str(exc_info.value)
