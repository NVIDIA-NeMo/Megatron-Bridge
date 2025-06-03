#!/usr/bin/env python3
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

import dataclasses
import functools
from unittest.mock import Mock, patch

import pytest
from megatron.core.optimizer import OptimizerConfig
import nemo_run as run
import torch.nn.init as init

from nemo_lm.models.gpt import GPTConfig
from nemo_lm.training.config import (
    ConfigContainer,
    TrainingConfig,
    GPTDatasetConfig,
    LoggerConfig,
    TokenizerConfig,
    CheckpointConfig,
    SchedulerConfig,
)
from nemo_lm.utils.nemo_run_utils import prepare_config_for_nemo_run


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


@dataclasses.dataclass
class MockConfigContainer:
    """Mock config container for testing."""

    model_config: MockModelConfig = dataclasses.field(default_factory=MockModelConfig)


class TestPrepareConfigForNemoRun:
    """Test prepare_config_for_nemo_run function."""

    def test_no_partial_objects(self):
        """Test that configs without functools.partial objects are unchanged."""
        # Create a config without any functools.partial objects
        model_config = MockModelConfig()
        config = MockConfigContainer(model_config=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should return the same config
        assert result is config
        assert result.model_config.init_method is None
        assert result.model_config.output_layer_init_method is None

    def test_init_method_partial_wrapping(self):
        """Test that init_method functools.partial is properly wrapped."""
        # Create a config with functools.partial in init_method
        partial_init = functools.partial(dummy_init_function, mean=0.0, std=0.02)
        model_config = MockModelConfig(init_method=partial_init)
        config = MockConfigContainer(model_config=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should wrap the partial with run.Partial
        assert isinstance(result.model_config.init_method, run.Partial)
        assert result.model_config.init_method._target_ == dummy_init_function

        # Verify the original arguments are preserved
        assert result.model_config.init_method.mean == 0.0
        assert result.model_config.init_method.std == 0.02

    def test_output_layer_init_method_partial_wrapping(self):
        """Test that output_layer_init_method functools.partial is properly wrapped."""
        # Create a config with functools.partial in output_layer_init_method
        partial_init = functools.partial(dummy_init_function, mean=0.0, std=0.00125)
        model_config = MockModelConfig(output_layer_init_method=partial_init)
        config = MockConfigContainer(model_config=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should wrap the partial with run.Partial
        assert isinstance(result.model_config.output_layer_init_method, run.Partial)
        assert result.model_config.output_layer_init_method._target_ == dummy_init_function

        # Verify the original arguments are preserved
        assert result.model_config.output_layer_init_method.mean == 0.0
        assert result.model_config.output_layer_init_method.std == 0.00125

    def test_multiple_partial_objects(self):
        """Test that multiple functools.partial objects are all wrapped."""
        # Create a config with multiple functools.partial objects
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        output_partial = functools.partial(another_init_function, gain=1.5)
        bias_partial = functools.partial(init.constant_, val=0.0)

        model_config = MockModelConfig(
            init_method=init_partial, output_layer_init_method=output_partial, bias_init_method=bias_partial
        )
        config = MockConfigContainer(model_config=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # All partials should be wrapped
        assert isinstance(result.model_config.init_method, run.Partial)
        assert isinstance(result.model_config.output_layer_init_method, run.Partial)
        assert isinstance(result.model_config.bias_init_method, run.Partial)

        # Verify correct targets
        assert result.model_config.init_method._target_ == dummy_init_function
        assert result.model_config.output_layer_init_method._target_ == another_init_function
        assert result.model_config.bias_init_method._target_ == init.constant_

        # Verify arguments are preserved
        assert result.model_config.init_method.mean == 0.0
        assert result.model_config.init_method.std == 0.01
        assert result.model_config.output_layer_init_method.gain == 1.5
        assert result.model_config.bias_init_method.val == 0.0

    def test_mixed_partial_and_non_partial(self):
        """Test handling when some fields are partials and others are not."""
        # Create a config with mixed types
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        regular_function = another_init_function  # Not a partial

        model_config = MockModelConfig(init_method=init_partial, output_layer_init_method=regular_function)
        config = MockConfigContainer(model_config=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Only the partial should be wrapped
        assert isinstance(result.model_config.init_method, run.Partial)
        assert result.model_config.output_layer_init_method == regular_function  # Unchanged

        # Verify the partial wrapping
        assert result.model_config.init_method._target_ == dummy_init_function

    def test_with_real_gpt_config(self):
        """Test with a real GPTConfig to ensure compatibility."""
        # Import actual configs for realistic testing
        from nemo_lm.recipes.llm.llama3_8b import get_llama3_8b_model_config

        # Get a real model config
        model_config = get_llama3_8b_model_config()

        # Create a minimal ConfigContainer with required fields
        config = ConfigContainer(
            model_config=model_config,
            train_config=TrainingConfig(micro_batch_size=1, global_batch_size=1, train_iters=10),
            optimizer_config=OptimizerConfig(),
            scheduler_config=SchedulerConfig(),
            dataset_config=GPTDatasetConfig(seq_length=2048, random_seed=42, data_path=[]),
            logger_config=LoggerConfig(),
            tokenizer_config=TokenizerConfig(),
            checkpoint_config=CheckpointConfig(),
        )

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should return a valid config
        assert isinstance(result, ConfigContainer)
        assert result.model_config is not None

        # If the real config has functools.partial objects, they should be wrapped
        if hasattr(result.model_config, 'init_method') and result.model_config.init_method is not None:
            # If it was a partial, it should now be a run.Partial
            if isinstance(result.model_config.init_method, run.Partial):
                assert hasattr(result.model_config.init_method, '_target_')

    def test_logging_output(self, caplog):
        """Test that the function logs which fields were wrapped."""
        import logging

        # Create a config with functools.partial objects
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        output_partial = functools.partial(another_init_function, gain=1.0)

        model_config = MockModelConfig(init_method=init_partial, output_layer_init_method=output_partial)
        config = MockConfigContainer(model_config=model_config)

        # Process the config with logging
        with caplog.at_level(logging.INFO):
            prepare_config_for_nemo_run(config)

        # Should log which fields were wrapped
        assert "Wrapped the following fields with run.Partial" in caplog.text
        assert "model_config.init_method" in caplog.text
        assert "model_config.output_layer_init_method" in caplog.text

    def test_no_logging_when_no_partials(self, caplog):
        """Test that no logging occurs when there are no partials to wrap."""
        import logging

        # Create a config without any functools.partial objects
        model_config = MockModelConfig()
        config = MockConfigContainer(model_config=model_config)

        # Process the config with logging
        with caplog.at_level(logging.INFO):
            prepare_config_for_nemo_run(config)

        # Should not log anything about wrapping
        assert "Wrapped the following fields" not in caplog.text

    def test_preserves_partial_args_and_kwargs(self):
        """Test that both args and kwargs of functools.partial are preserved."""

        # Create a partial with both args and kwargs
        def complex_init_function(tensor, arg1, arg2, kwarg1=None, kwarg2=None):
            return tensor

        partial_init = functools.partial(
            complex_init_function, "positional_arg1", "positional_arg2", kwarg1="keyword_arg1", kwarg2="keyword_arg2"
        )

        model_config = MockModelConfig(init_method=partial_init)
        config = MockConfigContainer(model_config=model_config)

        # Process the config
        result = prepare_config_for_nemo_run(config)

        # Should wrap the partial with run.Partial
        assert isinstance(result.model_config.init_method, run.Partial)

        # Verify all arguments are preserved
        wrapped_partial = result.model_config.init_method
        assert wrapped_partial._target_ == complex_init_function
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
            model_config: MinimalModelConfig = dataclasses.field(default_factory=MinimalModelConfig)

        config = MinimalConfigContainer()

        # Should not raise an error
        result = prepare_config_for_nemo_run(config)
        assert result is config


# Integration test
class TestNemoRunCompatibility:
    """Test that the prepared config works with NeMo Run."""

    def test_serialization_compatibility(self):
        """Test that the wrapped config can be serialized by NeMo Run."""
        # Create a config with functools.partial objects
        init_partial = functools.partial(dummy_init_function, mean=0.0, std=0.01)
        model_config = MockModelConfig(init_method=init_partial)
        config = MockConfigContainer(model_config=model_config)

        # Process the config
        prepared_config = prepare_config_for_nemo_run(config)

        # Test that it can be wrapped in run.Partial (basic serialization test)
        try:
            # This would fail if there are still unserializable objects
            partial_func = run.Partial(lambda cfg: cfg, config=prepared_config)
            assert partial_func is not None
        except Exception as e:
            pytest.fail(f"NeMo Run serialization failed: {e}")

    def test_run_partial_equivalence(self):
        """Test that run.Partial wrapped functions are equivalent to original partials."""
        # Create original partial
        original_partial = functools.partial(dummy_init_function, mean=0.5, std=0.02)

        # Create config and process it
        model_config = MockModelConfig(init_method=original_partial)
        config = MockConfigContainer(model_config=model_config)
        prepared_config = prepare_config_for_nemo_run(config)

        # The wrapped function should have the same behavior
        wrapped_partial = prepared_config.model_config.init_method

        # Test that they have the same target function
        assert wrapped_partial._target_ == original_partial.func

        # Test that they have the same arguments
        assert wrapped_partial.mean == original_partial.keywords['mean']
        assert wrapped_partial.std == original_partial.keywords['std']
