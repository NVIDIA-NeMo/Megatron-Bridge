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

from megatron.bridge.training.setup import _validate_and_set_vocab_size, maybe_log_and_save_config


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


class TestMaybeLogAndSaveConfig:
    """Tests for maybe_log_and_save_config."""

    @patch("megatron.bridge.training.setup.get_rank_safe", return_value=0)
    def test_rank_zero_saves_and_logs(self, mock_get_rank, tmp_path, capsys):
        filepath = tmp_path / "config.yaml"
        cfg = Mock()
        cfg.logger.save_config_filepath = str(filepath)
        cfg.to_yaml = Mock()
        cfg.print_yaml = Mock()

        maybe_log_and_save_config(cfg)

        cfg.to_yaml.assert_called_once_with(str(filepath))
        cfg.print_yaml.assert_called_once()
        captured = capsys.readouterr()
        assert "------- Task Configuration -------" in captured.out
        assert "----------------------------------" in captured.out

    @patch("megatron.bridge.training.setup.get_rank_safe", return_value=1)
    def test_non_zero_rank_noop(self, mock_get_rank):
        cfg = Mock()
        cfg.logger.save_config_filepath = "unused"
        cfg.to_yaml = Mock()
        cfg.print_yaml = Mock()

        maybe_log_and_save_config(cfg)

        cfg.to_yaml.assert_not_called()
        cfg.print_yaml.assert_not_called()

    @patch("megatron.bridge.training.setup.get_rank_safe", return_value=0)
    def test_save_failure_is_logged(self, mock_get_rank, capsys):
        cfg = Mock()
        cfg.logger.save_config_filepath = "path"

        def raise_io_error(_):
            raise IOError("boom")

        cfg.to_yaml.side_effect = raise_io_error
        cfg.print_yaml = Mock()

        maybe_log_and_save_config(cfg)

        captured = capsys.readouterr()
        assert "Error saving config" in captured.out
        cfg.print_yaml.assert_called_once()
