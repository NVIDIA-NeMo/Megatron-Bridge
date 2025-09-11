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

"""Tests for common_utils module."""

import os
from unittest.mock import patch

import pytest

from megatron.bridge.utils.common_utils import (
    get_local_rank_preinit,
    get_rank_safe,
    get_world_size_safe,
    is_last_rank,
    print_rank_0,
    print_rank_last,
)


class TestGetRankSafe:
    """Test get_rank_safe function."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_initialized_torch_distributed(self, mock_get_rank, mock_is_initialized):
        """Test get_rank_safe when torch.distributed is initialized."""
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 2

        result = get_rank_safe()

        assert result == 2
        mock_is_initialized.assert_called_once()
        mock_get_rank.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch.dict(os.environ, {"RANK": "3"})
    def test_uninitialized_torch_distributed_with_env_var(self, mock_is_initialized):
        """Test get_rank_safe when torch.distributed is not initialized but RANK env var exists."""
        mock_is_initialized.return_value = False

        result = get_rank_safe()

        assert result == 3
        mock_is_initialized.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch.dict(os.environ, {}, clear=True)
    def test_uninitialized_torch_distributed_no_env_var(self, mock_is_initialized):
        """Test get_rank_safe when torch.distributed is not initialized and no RANK env var."""
        mock_is_initialized.return_value = False

        result = get_rank_safe()

        assert result == 0
        mock_is_initialized.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch.dict(os.environ, {"RANK": "invalid"})
    def test_invalid_rank_env_var(self, mock_is_initialized):
        """Test get_rank_safe with invalid RANK environment variable."""
        mock_is_initialized.return_value = False

        with pytest.raises(ValueError):
            get_rank_safe()


class TestGetWorldSizeSafe:
    """Test get_world_size_safe function."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    def test_initialized_torch_distributed(self, mock_get_world_size, mock_is_initialized):
        """Test get_world_size_safe when torch.distributed is initialized."""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 4

        result = get_world_size_safe()

        assert result == 4
        mock_is_initialized.assert_called_once()
        mock_get_world_size.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch.dict(os.environ, {"WORLD_SIZE": "8"})
    def test_uninitialized_torch_distributed_with_env_var(self, mock_is_initialized):
        """Test get_world_size_safe when torch.distributed is not initialized but WORLD_SIZE env var exists."""
        mock_is_initialized.return_value = False

        result = get_world_size_safe()

        assert result == 8
        mock_is_initialized.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch.dict(os.environ, {}, clear=True)
    def test_uninitialized_torch_distributed_no_env_var(self, mock_is_initialized):
        """Test get_world_size_safe when torch.distributed is not initialized and no WORLD_SIZE env var."""
        mock_is_initialized.return_value = False

        result = get_world_size_safe()

        assert result == 1
        mock_is_initialized.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch.dict(os.environ, {"WORLD_SIZE": "invalid"})
    def test_invalid_world_size_env_var(self, mock_is_initialized):
        """Test get_world_size_safe with invalid WORLD_SIZE environment variable."""
        mock_is_initialized.return_value = False

        with pytest.raises(ValueError):
            get_world_size_safe()


class TestGetLocalRankPreinit:
    """Test get_local_rank_preinit function."""

    @patch.dict(os.environ, {"LOCAL_RANK": "2"})
    def test_with_local_rank_env_var(self):
        """Test get_local_rank_preinit with LOCAL_RANK environment variable."""
        result = get_local_rank_preinit()
        assert result == 2

    @patch.dict(os.environ, {}, clear=True)
    def test_without_local_rank_env_var(self):
        """Test get_local_rank_preinit without LOCAL_RANK environment variable."""
        result = get_local_rank_preinit()
        assert result == 0

    @patch.dict(os.environ, {"LOCAL_RANK": "invalid"})
    def test_invalid_local_rank_env_var(self):
        """Test get_local_rank_preinit with invalid LOCAL_RANK environment variable."""
        with pytest.raises(ValueError):
            get_local_rank_preinit()


class TestPrintRank0:
    """Test print_rank_0 function."""

    @patch("megatron.bridge.utils.common_utils.get_rank_safe")
    @patch("builtins.print")
    def test_print_on_rank_0(self, mock_print, mock_get_rank_safe):
        """Test print_rank_0 prints message when rank is 0."""
        mock_get_rank_safe.return_value = 0
        message = "Test message"

        print_rank_0(message)

        mock_get_rank_safe.assert_called_once()
        mock_print.assert_called_once_with(message, flush=True)

    @patch("megatron.bridge.utils.common_utils.get_rank_safe")
    @patch("builtins.print")
    def test_no_print_on_non_zero_rank(self, mock_print, mock_get_rank_safe):
        """Test print_rank_0 does not print message when rank is not 0."""
        mock_get_rank_safe.return_value = 1
        message = "Test message"

        print_rank_0(message)

        mock_get_rank_safe.assert_called_once()
        mock_print.assert_not_called()


class TestIsLastRank:
    """Test is_last_rank function."""

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_is_last_rank_true(self, mock_get_world_size, mock_get_rank):
        """Test is_last_rank returns True when current rank is the last rank."""
        mock_get_rank.return_value = 3
        mock_get_world_size.return_value = 4

        result = is_last_rank()

        assert result is True
        mock_get_rank.assert_called_once()
        mock_get_world_size.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_is_last_rank_false(self, mock_get_world_size, mock_get_rank):
        """Test is_last_rank returns False when current rank is not the last rank."""
        mock_get_rank.return_value = 1
        mock_get_world_size.return_value = 4

        result = is_last_rank()

        assert result is False
        mock_get_rank.assert_called_once()
        mock_get_world_size.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_single_rank_is_last(self, mock_get_world_size, mock_get_rank):
        """Test is_last_rank returns True when there's only one rank."""
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        result = is_last_rank()

        assert result is True
        mock_get_rank.assert_called_once()
        mock_get_world_size.assert_called_once()


class TestPrintRankLast:
    """Test print_rank_last function."""

    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.utils.common_utils.is_last_rank")
    @patch("builtins.print")
    def test_print_on_last_rank_when_initialized(self, mock_print, mock_is_last_rank, mock_is_initialized):
        """Test print_rank_last prints message when torch.distributed is initialized and rank is last."""
        mock_is_initialized.return_value = True
        mock_is_last_rank.return_value = True
        message = "Test message"

        print_rank_last(message)

        mock_is_initialized.assert_called_once()
        mock_is_last_rank.assert_called_once()
        mock_print.assert_called_once_with(message, flush=True)

    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.utils.common_utils.is_last_rank")
    @patch("builtins.print")
    def test_no_print_on_non_last_rank_when_initialized(self, mock_print, mock_is_last_rank, mock_is_initialized):
        """Test print_rank_last does not print message when torch.distributed is initialized and rank is not last."""
        mock_is_initialized.return_value = True
        mock_is_last_rank.return_value = False
        message = "Test message"

        print_rank_last(message)

        mock_is_initialized.assert_called_once()
        mock_is_last_rank.assert_called_once()
        mock_print.assert_not_called()

    @patch("torch.distributed.is_initialized")
    @patch("builtins.print")
    def test_print_when_not_initialized(self, mock_print, mock_is_initialized):
        """Test print_rank_last prints message when torch.distributed is not initialized."""
        mock_is_initialized.return_value = False
        message = "Test message"

        print_rank_last(message)

        mock_is_initialized.assert_called_once()
        mock_print.assert_called_once_with(message, flush=True)


class TestIntegration:
    """Integration tests for common_utils functions."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("builtins.print")
    def test_print_functions_integration(self, mock_print, mock_get_world_size, mock_get_rank, mock_is_initialized):
        """Test integration of print functions with rank determination."""
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # Test print_rank_0
        print_rank_0("Rank 0 message")

        # Test print_rank_last (rank 0 is also last rank when world_size=1)
        print_rank_last("Last rank message")

        # Both should print since rank 0 is both first and last in a single-rank scenario
        assert mock_print.call_count == 2

    def test_environment_variable_integration(self):
        """Test integration with environment variables."""
        test_env = {"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}

        with patch.dict(os.environ, test_env):
            with patch("torch.distributed.is_initialized", return_value=False):
                assert get_rank_safe() == 2
                assert get_world_size_safe() == 4
                assert get_local_rank_preinit() == 1
