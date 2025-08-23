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

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.training.setup import _apply_runtime_configurations, _validate_and_set_vocab_size


class TestValidateAndSetVocabSize:
    """Test cases for the _validate_and_set_vocab_size function."""

    def test_vocab_size_none_uses_tokenizer_vocab_size(self):
        """Test that None vocab_size uses tokenizer's vocab size and enables padding."""
        vocab_size, should_pad_vocab = _validate_and_set_vocab_size(
            model_vocab_size=None,
            tokenizer_vocab_size=32004,
        )
        assert vocab_size == 32004
        assert should_pad_vocab is True

    def test_vocab_size_smaller_than_tokenizer_raises_error(self):
        """Test that vocab_size smaller than tokenizer raises ValueError."""
        with pytest.raises(ValueError, match="cannot be smaller than tokenizer's vocab_size"):
            _validate_and_set_vocab_size(
                model_vocab_size=30000,
                tokenizer_vocab_size=32004,
            )

    def test_vocab_size_larger_than_tokenizer_returns_same_value(self):
        """Test that vocab_size larger than tokenizer returns the same value and disables padding."""
        vocab_size, should_pad_vocab = _validate_and_set_vocab_size(
            model_vocab_size=40960,
            tokenizer_vocab_size=32004,
        )
        assert vocab_size == 40960
        assert should_pad_vocab is False

    def test_vocab_size_equal_to_tokenizer_returns_same_value(self):
        """Test that vocab_size equal to tokenizer returns the same value and disables padding."""
        vocab_size, should_pad_vocab = _validate_and_set_vocab_size(
            model_vocab_size=32004,
            tokenizer_vocab_size=32004,
        )
        assert vocab_size == 32004
        assert should_pad_vocab is False


class TestApplyRuntimeConfigurations:
    """Test cases for the _apply_runtime_configurations function."""

    def test_both_configs_none_no_op(self):
        """Test that function does nothing when both mixed_precision and comm_overlap are None."""
        cfg = Mock()
        cfg.mixed_precision = None
        cfg.comm_overlap = None

        _apply_runtime_configurations(cfg)

        # Should not call any setup methods
        assert cfg.mixed_precision is None
        assert cfg.comm_overlap is None

    @patch("megatron.bridge.training.setup.get_mixed_precision_config")
    def test_mixed_precision_string_conversion(self, mock_get_mixed_precision_config):
        """Test mixed precision config conversion from string."""
        cfg = Mock()
        cfg.mixed_precision = "bf16"
        cfg.comm_overlap = None

        mock_converted_config = Mock()
        mock_get_mixed_precision_config.return_value = mock_converted_config

        _apply_runtime_configurations(cfg)

        # Should convert string to config object and call setup
        mock_get_mixed_precision_config.assert_called_once_with("bf16")
        assert cfg.mixed_precision == mock_converted_config
        mock_converted_config.setup.assert_called_once_with(cfg.model, cfg.optimizer, cfg.ddp)

    def test_mixed_precision_object_direct_setup(self):
        """Test mixed precision config when already an object."""
        cfg = Mock()
        mock_mp_config = Mock()
        cfg.mixed_precision = mock_mp_config
        cfg.comm_overlap = None

        _apply_runtime_configurations(cfg)

        # Should call setup directly without conversion
        mock_mp_config.setup.assert_called_once_with(cfg.model, cfg.optimizer, cfg.ddp)

    def test_comm_overlap_setup(self):
        """Test communication overlap configuration setup."""
        cfg = Mock()
        cfg.mixed_precision = None
        mock_comm_config = Mock()
        cfg.comm_overlap = mock_comm_config

        _apply_runtime_configurations(cfg)

        # Should call setup on comm_overlap
        mock_comm_config.setup.assert_called_once_with(cfg.model, cfg.optimizer, cfg.ddp)

    @patch("megatron.bridge.training.setup.get_mixed_precision_config")
    def test_both_configs_present(self, mock_get_mixed_precision_config):
        """Test when both mixed_precision and comm_overlap are configured."""
        cfg = Mock()
        cfg.mixed_precision = "fp16"
        mock_comm_config = Mock()
        cfg.comm_overlap = mock_comm_config

        mock_converted_config = Mock()
        mock_get_mixed_precision_config.return_value = mock_converted_config

        _apply_runtime_configurations(cfg)

        # Both configurations should be applied
        mock_get_mixed_precision_config.assert_called_once_with("fp16")
        assert cfg.mixed_precision == mock_converted_config
        mock_converted_config.setup.assert_called_once_with(cfg.model, cfg.optimizer, cfg.ddp)
        mock_comm_config.setup.assert_called_once_with(cfg.model, cfg.optimizer, cfg.ddp)

    @patch("megatron.bridge.training.setup.get_mixed_precision_config")
    def test_mixed_precision_object_with_comm_overlap(self, mock_get_mixed_precision_config):
        """Test mixed precision object (no conversion needed) with comm overlap."""
        cfg = Mock()
        mock_mp_config = Mock()
        cfg.mixed_precision = mock_mp_config
        mock_comm_config = Mock()
        cfg.comm_overlap = mock_comm_config

        _apply_runtime_configurations(cfg)

        # Should not call get_mixed_precision_config for conversion
        mock_get_mixed_precision_config.assert_not_called()
        # Both should call setup
        mock_mp_config.setup.assert_called_once_with(cfg.model, cfg.optimizer, cfg.ddp)
        mock_comm_config.setup.assert_called_once_with(cfg.model, cfg.optimizer, cfg.ddp)

    @patch("megatron.bridge.training.setup.get_mixed_precision_config")
    def test_mixed_precision_setup_exception_propagated(self, mock_get_mixed_precision_config):
        """Test that exceptions from mixed precision setup are propagated."""
        cfg = Mock()
        cfg.mixed_precision = "invalid"
        cfg.comm_overlap = None

        mock_converted_config = Mock()
        mock_converted_config.setup.side_effect = ValueError("Invalid mixed precision config")
        mock_get_mixed_precision_config.return_value = mock_converted_config

        with pytest.raises(ValueError, match="Invalid mixed precision config"):
            _apply_runtime_configurations(cfg)
