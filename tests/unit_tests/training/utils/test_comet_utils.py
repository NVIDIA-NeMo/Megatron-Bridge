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

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.training.utils.comet_utils import (
    on_load_checkpoint_success,
    on_save_checkpoint_success,
)


@pytest.mark.unit
class TestOnSaveCheckpointSuccess:
    """Test cases for on_save_checkpoint_success function."""

    def test_noop_when_comet_logger_is_none(self):
        """Test that the function does nothing when comet_logger is None."""
        on_save_checkpoint_success(
            checkpoint_path="/path/to/checkpoint",
            save_dir="/path/to",
            iteration=100,
            comet_logger=None,
        )

    def test_logs_checkpoint_metadata(self):
        """Test that log_other is called with correct checkpoint information."""
        mock_comet = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            on_save_checkpoint_success(
                checkpoint_path=str(checkpoint_path),
                save_dir=tmpdir,
                iteration=1000,
                comet_logger=mock_comet,
            )

            assert mock_comet.log_other.call_count == 2
            calls = mock_comet.log_other.call_args_list
            assert calls[0][0] == ("last_saved_checkpoint", str(checkpoint_path.resolve()))
            assert calls[1][0] == ("last_saved_iteration", 1000)

    def test_handles_exception_gracefully(self):
        """Test that exceptions are caught and logged, not raised."""
        mock_comet = MagicMock()
        mock_comet.log_other.side_effect = Exception("Comet connection error")

        with patch("megatron.bridge.training.utils.comet_utils.print_rank_last") as mock_print:
            on_save_checkpoint_success(
                checkpoint_path="/path/to/checkpoint",
                save_dir="/path/to",
                iteration=100,
                comet_logger=mock_comet,
            )

            mock_print.assert_called_once()
            error_msg = mock_print.call_args[0][0]
            assert "Failed to log checkpoint information to Comet ML" in error_msg
            assert "Comet connection error" in error_msg


@pytest.mark.unit
class TestOnLoadCheckpointSuccess:
    """Test cases for on_load_checkpoint_success function."""

    def test_noop_when_comet_logger_is_none(self):
        """Test that the function does nothing when comet_logger is None."""
        on_load_checkpoint_success(
            checkpoint_path="/path/to/checkpoint",
            load_dir="/path/to",
            comet_logger=None,
        )

    def test_logs_correct_metadata(self):
        """Test that log_other is called with correct checkpoint information."""
        mock_comet = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            on_load_checkpoint_success(
                checkpoint_path=str(checkpoint_path),
                load_dir=tmpdir,
                comet_logger=mock_comet,
            )

            assert mock_comet.log_other.call_count == 2
            calls = mock_comet.log_other.call_args_list
            assert calls[0][0] == ("last_loaded_checkpoint", str(checkpoint_path.resolve()))
            assert calls[1][0] == ("checkpoint_base_dir", str(Path(tmpdir).resolve()))

    def test_resolves_relative_paths(self):
        """Test that relative paths are resolved to absolute paths."""
        mock_comet = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            on_load_checkpoint_success(
                checkpoint_path=str(checkpoint_path),
                load_dir=tmpdir,
                comet_logger=mock_comet,
            )

            calls = mock_comet.log_other.call_args_list
            assert Path(calls[0][0][1]).is_absolute()
            assert Path(calls[1][0][1]).is_absolute()

    def test_handles_exception_gracefully(self):
        """Test that exceptions are caught and logged, not raised."""
        mock_comet = MagicMock()
        mock_comet.log_other.side_effect = Exception("Comet API error")

        with patch("megatron.bridge.training.utils.comet_utils.print_rank_last") as mock_print:
            on_load_checkpoint_success(
                checkpoint_path="/path/to/checkpoint",
                load_dir="/path/to",
                comet_logger=mock_comet,
            )

            mock_print.assert_called_once()
            error_msg = mock_print.call_args[0][0]
            assert "Failed to record loaded checkpoint information to Comet ML" in error_msg
            assert "Comet API error" in error_msg
