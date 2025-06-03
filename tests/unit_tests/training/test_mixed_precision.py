"""Unit tests for nemo_lm.training.mixed_precision module."""

from dataclasses import dataclass, fields
from unittest.mock import MagicMock, patch

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo_lm.models.gpt import GPTConfig
from nemo_lm.models.t5 import T5Config
from nemo_lm.training.mixed_precision import MegatronMixedPrecisionConfig, update_config_with_precision_overrides


class TestMegatronMixedPrecisionConfig:
    def test_fp8_configurations(self):
        config = MegatronMixedPrecisionConfig(
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

    @patch("logging.debug")
    def test_setup_with_gpt_config(self, mock_log):
        mixed_precision_config = MegatronMixedPrecisionConfig(
            fp16=True, bf16=False, params_dtype=torch.float16, loss_scale=1024.0
        )

        # Create mock GPTConfig with necessary attributes
        gpt_config = MagicMock(spec=GPTConfig)
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
        mixed_precision_config = MegatronMixedPrecisionConfig(
            bf16=True, params_dtype=torch.bfloat16, autocast_enabled=True, autocast_dtype=torch.bfloat16
        )

        # Create mock T5Config
        t5_config = MagicMock(spec=T5Config)
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
        mixed_precision_config = MegatronMixedPrecisionConfig(
            grad_reduce_in_fp32=False, loss_scale=512.0, initial_loss_scale=1024.0
        )

        # Create mock configs
        model_config = MagicMock(spec=GPTConfig)
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
        mixed_precision_config = MegatronMixedPrecisionConfig(grad_reduce_in_fp32=False, fp16=True)

        # Create mock configs
        model_config = MagicMock(spec=GPTConfig)
        ddp_config = MagicMock(spec=DistributedDataParallelConfig)
        ddp_config.grad_reduce_in_fp32 = True
        ddp_config.fp16 = False

        # Call setup
        mixed_precision_config.setup(model_config, ddp_config=ddp_config)

        # Verify DDP config was updated
        assert ddp_config.grad_reduce_in_fp32 is False
        assert ddp_config.fp16 is True

    def test_setup_with_all_configs(self):
        mixed_precision_config = MegatronMixedPrecisionConfig(
            bf16=True, params_dtype=torch.bfloat16, grad_reduce_in_fp32=False
        )

        # Create mock configs
        model_config = MagicMock(spec=GPTConfig)
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
        mixed_precision_config = MegatronMixedPrecisionConfig(fp16=True, bf16=False, params_dtype=torch.float16)

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
        mixed_precision_config = MegatronMixedPrecisionConfig(fp16=True)

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
        mixed_precision_config = MegatronMixedPrecisionConfig(fp16=True, loss_scale=1024.0, fp8_margin=2)

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
        mixed_precision_config = MegatronMixedPrecisionConfig(params_dtype=None, loss_scale=None)

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
        mixed_precision_config = MegatronMixedPrecisionConfig(fp16=True)

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
        mixed_precision_config = MegatronMixedPrecisionConfig(
            fp16=True,
            params_dtype=torch.float16,
            loss_scale=1024.0,
            initial_loss_scale=2048.0,
            min_loss_scale=1.0,
            loss_scale_window=1000.0,
            hysteresis=2.0,
        )

        # Create configs
        model_config = MagicMock(spec=GPTConfig)
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
        mixed_precision_config = MegatronMixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=2,
        )

        # Create model config
        model_config = MagicMock(spec=GPTConfig)
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
        mixed_precision_config = MegatronMixedPrecisionConfig(
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
        model_config = MagicMock(spec=GPTConfig)
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
