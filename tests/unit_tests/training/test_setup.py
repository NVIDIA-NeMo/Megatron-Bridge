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

from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.training.setup import _validate_and_set_vocab_size, runtime_config_update


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


class TestRuntimeConfigUpdate:
    """Test cases for the runtime_config_update function."""

    def test_runtime_config_update_basic_flow(self):
        """Test basic flow with no mixed precision or comm overlap."""
        mock_cfg = MagicMock()
        mock_cfg.mixed_precision = None
        mock_cfg.comm_overlap = None

        runtime_config_update(mock_cfg)

        # Should call validate twice - before and after processing
        assert mock_cfg.validate.call_count == 2

    def test_runtime_config_update_with_mixed_precision_string(self):
        """Test mixed precision handling when it's a string."""
        mock_cfg = MagicMock()
        mock_cfg.mixed_precision = "bf16"
        mock_cfg.comm_overlap = None

        mock_mixed_precision_config = MagicMock()

        with patch(
            "megatron.bridge.training.setup.get_mixed_precision_config", return_value=mock_mixed_precision_config
        ) as mock_get_mp:
            runtime_config_update(mock_cfg)

            # Should convert string to config object
            mock_get_mp.assert_called_once_with("bf16")
            assert mock_cfg.mixed_precision == mock_mixed_precision_config

            # Should call setup on the mixed precision config
            mock_mixed_precision_config.setup.assert_called_once_with(mock_cfg.model, mock_cfg.optimizer, mock_cfg.ddp)

            # Should call validate twice
            assert mock_cfg.validate.call_count == 2

    def test_runtime_config_update_with_mixed_precision_object(self):
        """Test mixed precision handling when it's already an object."""
        mock_cfg = MagicMock()
        mock_mixed_precision_config = MagicMock()
        mock_cfg.mixed_precision = mock_mixed_precision_config
        mock_cfg.comm_overlap = None

        with patch("megatron.bridge.training.setup.get_mixed_precision_config") as mock_get_mp:
            runtime_config_update(mock_cfg)

            # Should not call get_mixed_precision_config
            mock_get_mp.assert_not_called()

            # Should call setup on the existing mixed precision config
            mock_mixed_precision_config.setup.assert_called_once_with(mock_cfg.model, mock_cfg.optimizer, mock_cfg.ddp)

            # Should call validate twice
            assert mock_cfg.validate.call_count == 2

    def test_runtime_config_update_with_comm_overlap(self):
        """Test communication overlap handling."""
        mock_cfg = MagicMock()
        mock_cfg.mixed_precision = None
        mock_comm_overlap_config = MagicMock()
        mock_cfg.comm_overlap = mock_comm_overlap_config

        runtime_config_update(mock_cfg)

        # Should call setup on the comm overlap config
        mock_comm_overlap_config.setup.assert_called_once_with(mock_cfg.model, mock_cfg.optimizer, mock_cfg.ddp)

        # Should call validate twice
        assert mock_cfg.validate.call_count == 2

    def test_runtime_config_update_with_both_mixed_precision_and_comm_overlap(self):
        """Test handling both mixed precision and communication overlap."""
        mock_cfg = MagicMock()
        mock_cfg.mixed_precision = "fp16"
        mock_comm_overlap_config = MagicMock()
        mock_cfg.comm_overlap = mock_comm_overlap_config

        mock_mixed_precision_config = MagicMock()

        with patch(
            "megatron.bridge.training.setup.get_mixed_precision_config", return_value=mock_mixed_precision_config
        ):
            runtime_config_update(mock_cfg)

            # Should process mixed precision
            assert mock_cfg.mixed_precision == mock_mixed_precision_config
            mock_mixed_precision_config.setup.assert_called_once_with(mock_cfg.model, mock_cfg.optimizer, mock_cfg.ddp)

            # Should process comm overlap
            mock_comm_overlap_config.setup.assert_called_once_with(mock_cfg.model, mock_cfg.optimizer, mock_cfg.ddp)

            # Should call validate twice
            assert mock_cfg.validate.call_count == 2

    def test_runtime_config_update_call_order(self):
        """Test that operations are called in the correct order."""
        mock_cfg = MagicMock()
        mock_cfg.mixed_precision = "bf16"
        mock_comm_overlap_config = MagicMock()
        mock_cfg.comm_overlap = mock_comm_overlap_config

        mock_mixed_precision_config = MagicMock()
        call_order = []

        def track_validate(*args, **kwargs):
            call_order.append("validate")

        def track_mp_setup(*args, **kwargs):
            call_order.append("mixed_precision_setup")

        def track_comm_setup(*args, **kwargs):
            call_order.append("comm_overlap_setup")

        mock_cfg.validate.side_effect = track_validate
        mock_mixed_precision_config.setup.side_effect = track_mp_setup
        mock_comm_overlap_config.setup.side_effect = track_comm_setup

        with patch(
            "megatron.bridge.training.setup.get_mixed_precision_config", return_value=mock_mixed_precision_config
        ):
            runtime_config_update(mock_cfg)

        # Verify the correct order of operations
        expected_order = ["validate", "mixed_precision_setup", "comm_overlap_setup", "validate"]
        assert call_order == expected_order
