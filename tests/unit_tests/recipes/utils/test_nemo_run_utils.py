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

"""Tests for nemo_run_utils module."""

import pytest


pytest.importorskip("nemo_run")


import dataclasses
import enum
import functools

import nemo_run as run
import torch.nn.init as init
from megatron.core.optimizer import OptimizerConfig

from megatron.bridge.recipes.utils.nemo_run_utils import prepare_config_for_nemo_run
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)


# Test enums for enum handling tests
class MockAttnBackend(enum.Enum):
    """Mock attention backend enum for testing."""

    AUTO = 5
    FUSED = 1
    FLASH = 2


class MockPrecision(enum.Enum):
    """Mock precision enum for testing."""

    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"


# Test dataclasses and utilities
def dummy_init_function(tensor, mean=0.0, std=0.01):
    """Dummy initialization function for testing."""
    return init.normal_(tensor, mean=mean, std=std)


def another_init_function(tensor, gain=1.0):
    """Another dummy initialization function for testing."""
    return init.xavier_uniform_(tensor, gain=gain)


@dataclasses.dataclass
class MockModelConfig:
    """Mock model config for testing."""

    hidden_size: int = 512
    init_method: any = None
    output_layer_init_method: any = None
    bias_init_method: any = None
    weight_init_method: any = None
    attn_backend: any = None
    attention_backend: any = None


@dataclasses.dataclass
class MockTrainConfig:
    """Mock train config for testing."""

    precision: any = None
    backend: any = None


@dataclasses.dataclass
class MockConfigContainer:
    """Mock config container for testing."""

    model: MockModelConfig = dataclasses.field(default_factory=MockModelConfig)
    train: MockTrainConfig = dataclasses.field(default_factory=MockTrainConfig)


class TestPrepareConfigForNemoRun:
    """Test prepare_config_for_nemo_run function."""

    def test_no_partial_objects(self):
        """Test that configs without functools.partial objects are unchanged."""
        # Create a config without any functools.partial objects
        model_config = MockModelConfig()
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should return the same config
        assert result is config
        assert result.model.init_method is None
        assert result.model.output_layer_init_method is None

    def test_init_method_partial_wrapping(self):
        """Test that init_method functools.partial is properly wrapped."""
        # Create a config with functools.partial in init_method
        partial_init = functools.partial(dummy_init_function, mean=0.0, std=0.02)
        model_config = MockModelConfig(init_method=partial_init)
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should wrap the partial with run.Partial
        assert isinstance(result.model.init_method, run.Partial)
        assert result.model.init_method.__fn_or_cls__ == dummy_init_function

        # Verify the original arguments are preserved
        assert result.model.init_method.mean == 0.0
        assert result.model.init_method.std == 0.02

    def test_output_layer_init_method_partial_wrapping(self):
        """Test that output_layer_init_method functools.partial is properly wrapped."""
        # Create a config with functools.partial in output_layer_init_method
        partial_init = functools.partial(dummy_init_function, mean=0.0, std=0.00125)
        model_config = MockModelConfig(output_layer_init_method=partial_init)
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should wrap the partial with run.Partial
        assert isinstance(result.model.output_layer_init_method, run.Partial)
        assert result.model.output_layer_init_method.__fn_or_cls__ == dummy_init_function

        # Verify the original arguments are preserved
        assert result.model.output_layer_init_method.mean == 0.0
        assert result.model.output_layer_init_method.std == 0.00125

    def test_enum_conversion_model_config(self):
        """Test that enum objects in model config are converted to values."""
        # Create a config with enum objects
        model_config = MockModelConfig(attn_backend=MockAttnBackend.AUTO, attention_backend=MockAttnBackend.FLASH)
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Enums should be converted to their values
        assert result.model.attn_backend == 5  # MockAttnBackend.AUTO.value
        assert result.model.attention_backend == 2  # MockAttnBackend.FLASH.value

    def test_enum_conversion_train_config(self):
        """Test that enum objects in train config are converted to values."""
        # Create a config with enum objects
        train_config = MockTrainConfig(precision=MockPrecision.FP16)
        config = MockConfigContainer(train=train_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Enums should be converted to their values
        assert result.train.precision == "fp16"  # MockPrecision.FP16.value

    def test_multiple_partial_objects(self):
        """Test that multiple functools.partial objects are all wrapped."""
        # Create a config with multiple functools.partial objects
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        output_partial = functools.partial(another_init_function, gain=1.5)
        bias_partial = functools.partial(init.constant_, val=0.0)

        model_config = MockModelConfig(
            init_method=init_partial, output_layer_init_method=output_partial, bias_init_method=bias_partial
        )
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # All partials should be wrapped
        assert isinstance(result.model.init_method, run.Partial)
        assert isinstance(result.model.output_layer_init_method, run.Partial)
        assert isinstance(result.model.bias_init_method, run.Partial)

        # Verify correct targets
        assert result.model.init_method.__fn_or_cls__ == dummy_init_function
        assert result.model.output_layer_init_method.__fn_or_cls__ == another_init_function
        assert result.model.bias_init_method.__fn_or_cls__ == init.constant_

        # Verify arguments are preserved
        assert result.model.init_method.mean == 0.0
        assert result.model.init_method.std == 0.01
        assert result.model.output_layer_init_method.gain == 1.5
        assert result.model.bias_init_method.val == 0.0

    def test_mixed_partial_and_enum_objects(self):
        """Test handling when config has both partials and enums."""
        # Create a config with both partials and enums
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        model_config = MockModelConfig(init_method=init_partial, attn_backend=MockAttnBackend.FUSED)
        train_config = MockTrainConfig(precision=MockPrecision.BF16)
        config = MockConfigContainer(model=model_config, train=train_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Partial should be wrapped
        assert isinstance(result.model.init_method, run.Partial)
        assert result.model.init_method.__fn_or_cls__ == dummy_init_function

        # Enums should be converted
        assert result.model.attn_backend == 1  # MockAttnBackend.FUSED.value
        assert result.train.precision == "bf16"  # MockPrecision.BF16.value

    def test_mixed_partial_and_non_partial(self):
        """Test handling when some fields are partials and others are not."""
        # Create a config with mixed types
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        regular_function = another_init_function  # Not a partial

        model_config = MockModelConfig(init_method=init_partial, output_layer_init_method=regular_function)
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Only the partial should be wrapped
        assert isinstance(result.model.init_method, run.Partial)
        assert result.model.output_layer_init_method == regular_function  # Unchanged

        # Verify the partial wrapping
        assert result.model.init_method.__fn_or_cls__ == dummy_init_function

    def test_with_real_gpt_config(self):
        """Test with a real GPTConfig to ensure compatibility."""
        # Import actual configs for realistic testing, but avoid HF downloads by mocking AutoBridge
        from unittest import mock as _mock

        from megatron.bridge.recipes.llama.llama3 import llama3_8b_pretrain_config as pretrain_config

        with _mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights: bool = False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
            # Get a real model config (provider) without contacting HF
            model_cfg = pretrain_config().model

        # Create a minimal ConfigContainer with required fields
        config = ConfigContainer(
            model=model_cfg,
            train=TrainingConfig(micro_batch_size=1, global_batch_size=1, train_iters=10),
            optimizer=OptimizerConfig(),
            scheduler=SchedulerConfig(),
            dataset=GPTDatasetConfig(
                sequence_length=2048,
                random_seed=42,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            ),
            logger=LoggerConfig(),
            tokenizer=TokenizerConfig(),
            checkpoint=CheckpointConfig(),
        )

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should return a valid config
        assert isinstance(result, ConfigContainer)
        assert result.model is not None

        # If the real config has functools.partial objects, they should be wrapped
        if hasattr(result.model, "init_method") and result.model.init_method is not None:
            # If it was a partial, it should now be a run.Partial
            if isinstance(result.model.init_method, run.Partial):
                assert hasattr(result.model.init_method, "__fn_or_cls__")

    def test_logging_output_partials(self, caplog):
        """Test that the function logs which partial fields were wrapped."""
        import logging

        # Create a config with functools.partial objects
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        output_partial = functools.partial(another_init_function, gain=1.0)

        model_config = MockModelConfig(init_method=init_partial, output_layer_init_method=output_partial)
        config = MockConfigContainer(model=model_config)

        # Process the config with logging
        with caplog.at_level(logging.DEBUG):
            prepare_config_for_nemo_run(config)

        # Should log which fields were wrapped
        assert "Wrapped the following fields with run.Partial" in caplog.text
        assert "model_config.init_method" in caplog.text
        assert "model_config.output_layer_init_method" in caplog.text

    def test_logging_output_enums(self, caplog):
        """Test that the function logs which enum fields were converted."""
        import logging

        # Create a config with enum objects
        model_config = MockModelConfig(attn_backend=MockAttnBackend.AUTO)
        train_config = MockTrainConfig(precision=MockPrecision.FP32)
        config = MockConfigContainer(model=model_config, train=train_config)

        # Process the config with logging
        with caplog.at_level(logging.DEBUG):
            prepare_config_for_nemo_run(config)

        # Should log which enum fields were fixed
        assert "Fixed YAML serialization for enum fields" in caplog.text

    def test_no_logging_when_no_partials_or_enums(self, caplog):
        """Test that no logging occurs when there are no partials or enums to process."""
        import logging

        # Create a config without any functools.partial objects or enums
        model_config = MockModelConfig()
        config = MockConfigContainer(model=model_config)

        # Process the config with logging
        with caplog.at_level(logging.DEBUG):
            prepare_config_for_nemo_run(config)

        # Should not log anything about wrapping or fixing
        assert "Wrapped the following fields" not in caplog.text
        assert "Fixed YAML serialization" not in caplog.text

    def test_preserves_partial_args_and_kwargs(self):
        """Test that both args and kwargs of functools.partial are preserved."""

        # Create a partial with both args and kwargs
        def complex_init_function(tensor, arg1, arg2, kwarg1=None, kwarg2=None):
            return tensor

        partial_init = functools.partial(
            complex_init_function, "positional_arg1", "positional_arg2", kwarg1="keyword_arg1", kwarg2="keyword_arg2"
        )

        model_config = MockModelConfig(init_method=partial_init)
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should wrap the partial with run.Partial
        assert isinstance(result.model.init_method, run.Partial)

        # Verify all arguments are preserved
        wrapped_partial = result.model.init_method
        assert wrapped_partial.__fn_or_cls__ == complex_init_function
        assert wrapped_partial.kwarg1 == "keyword_arg1"
        assert wrapped_partial.kwarg2 == "keyword_arg2"

    def test_edge_case_missing_attributes(self):
        """Test handling when model_config doesn't have expected attributes."""

        @dataclasses.dataclass
        class MinimalModelConfig:
            hidden_size: int = 512
            # Missing init_method and output_layer_init_method

        @dataclasses.dataclass
        class MinimalConfigContainer:
            model: MinimalModelConfig = dataclasses.field(default_factory=MinimalModelConfig)

        config = MinimalConfigContainer()

        # Should not raise an error
        result = prepare_config_for_nemo_run(config)
        assert result is config

    def test_enum_conversion_preserves_non_enum_values(self):
        """Test that non-enum values are left unchanged during enum processing."""
        # Create a config with mixed types
        model_config = MockModelConfig(
            attn_backend=MockAttnBackend.AUTO,  # Enum - should be converted
            attention_backend="manual_string",  # String - should be unchanged
            hidden_size=1024,  # Int - should be unchanged
        )
        config = MockConfigContainer(model=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Enum should be converted
        assert result.model.attn_backend == 5  # MockAttnBackend.AUTO.value

        # Non-enum values should be unchanged
        assert result.model.attention_backend == "manual_string"
        assert result.model.hidden_size == 1024


# Integration test
class TestNemoRunCompatibility:
    """Test that the prepared config works with NeMo Run."""

    def test_serialization_compatibility(self):
        """Test that the wrapped config can be serialized by NeMo Run."""
        # Create a config with functools.partial objects
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        model_config = MockModelConfig(init_method=init_partial)
        config = MockConfigContainer(model=model_config)

        # Process the config
        prepared_config = prepare_config_for_nemo_run(config)

        # Test that it can be wrapped in run.Partial (basic serialization test)
        try:
            # This would fail if there are still unserializable objects
            # Use a function that accepts a config parameter
            def test_func(config):
                return config

            partial_func = run.Partial(test_func, config=prepared_config)
            assert partial_func is not None
        except Exception as e:
            pytest.fail(f"NeMo Run serialization failed: {e}")

    def test_serialization_compatibility_with_enums(self):
        """Test that configs with converted enums can be serialized by NeMo Run."""
        # Create a config with enum objects
        model_config = MockModelConfig(attn_backend=MockAttnBackend.FUSED)
        config = MockConfigContainer(model=model_config)

        # Process the config
        prepared_config = prepare_config_for_nemo_run(config)

        # Test that it can be wrapped in run.Partial
        try:
            # Use a function that accepts a config parameter
            def test_func(config):
                return config

            partial_func = run.Partial(test_func, config=prepared_config)
            assert partial_func is not None
            # Verify the enum was converted
            assert prepared_config.model.attn_backend == 1  # MockAttnBackend.FUSED.value
        except Exception as e:
            pytest.fail(f"NeMo Run serialization with enums failed: {e}")

    def test_run_partial_equivalence(self):
        """Test that run.Partial wrapped functions are equivalent to original partials."""
        # Create original partial
        original_partial = functools.partial(dummy_init_function, mean=0.5, std=0.02)

        # Create config and process it
        model_config = MockModelConfig(init_method=original_partial)
        config = MockConfigContainer(model=model_config)
        prepared_config = prepare_config_for_nemo_run(config)

        # The wrapped function should have the same behavior
        wrapped_partial = prepared_config.model.init_method

        # Test that they have the same target function
        assert wrapped_partial.__fn_or_cls__ == original_partial.func

        # Test that they have the same arguments
        assert wrapped_partial.mean == original_partial.keywords["mean"]
        assert wrapped_partial.std == original_partial.keywords["std"]
