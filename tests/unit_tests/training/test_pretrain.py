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

"""Unit tests for pretrain module process group cleanup."""

from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.training.pretrain import _maybe_destroy_process_group, _pretrain


class TestDestroyProcessGroupIfNeeded:
    """Test process group destruction logic."""

    @patch("megatron.bridge.training.pretrain.dist")
    def test_destroy_when_should_destroy_and_initialized(self, mock_dist):
        """Test process group is destroyed when both conditions are met."""
        mock_dist.is_initialized.return_value = True

        _maybe_destroy_process_group(should_destroy=True)

        mock_dist.barrier.assert_called_once()
        mock_dist.destroy_process_group.assert_called_once()

    @patch("megatron.bridge.training.pretrain.dist")
    def test_no_destroy_when_should_not_destroy(self, mock_dist):
        """Test no destruction when should_destroy is False."""
        mock_dist.is_initialized.return_value = True

        _maybe_destroy_process_group(should_destroy=False)

        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()

    @patch("megatron.bridge.training.pretrain.dist")
    def test_no_destroy_when_not_initialized(self, mock_dist):
        """Test no destruction when process group is not initialized."""
        mock_dist.is_initialized.return_value = False

        _maybe_destroy_process_group(should_destroy=True)

        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()

    @patch("megatron.bridge.training.pretrain.dist")
    def test_no_destroy_when_neither_condition_met(self, mock_dist):
        """Test no destruction when both conditions are false."""
        mock_dist.is_initialized.return_value = False

        _maybe_destroy_process_group(should_destroy=False)

        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()


class TestPretrainProcessGroupOwnership:
    """Test process group ownership across exceptional pretrain exits."""

    def test_framework_owned_process_group_is_destroyed_when_setup_raises(self):
        """Test Bridge destroys a process group initialized during failed setup."""
        state = MagicMock()

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.destroy_global_state") as mock_destroy_global_state,
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch(
                "megatron.bridge.training.pretrain.setup",
                side_effect=RuntimeError("setup failed after distributed initialization"),
            ),
        ):
            mock_dist.is_initialized.side_effect = [False, True]

            with pytest.raises(RuntimeError, match="setup failed after distributed initialization"):
                _pretrain(state, MagicMock())

        mock_destroy_global_state.assert_called_once_with()
        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_called_once_with()

    def test_caller_owned_process_group_is_preserved_when_setup_raises(self):
        """Test Bridge preserves a process group that existed before setup."""
        state = MagicMock()

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup", side_effect=RuntimeError("setup failed")),
        ):
            mock_dist.is_initialized.return_value = True

            with pytest.raises(RuntimeError, match="setup failed"):
                _pretrain(state, MagicMock())

        mock_dist.destroy_process_group.assert_not_called()

    def test_inprocess_restart_wrapper_retains_cleanup_ownership_when_setup_raises(self):
        """Test NVRx retains cleanup ownership for wrapped pretrain failures."""
        state = MagicMock()
        store = MagicMock()
        inprocess_call_wrapper = MagicMock()
        inprocess_call_wrapper.iteration = 1

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.destroy_global_state") as mock_destroy_global_state,
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup", side_effect=RuntimeError("setup failed")),
        ):
            mock_dist.is_initialized.return_value = False

            with pytest.raises(RuntimeError, match="setup failed"):
                _pretrain(
                    state,
                    MagicMock(),
                    store=store,
                    inprocess_call_wrapper=inprocess_call_wrapper,
                )

        mock_dist.PrefixStore.assert_called_once_with("1", store)
        mock_destroy_global_state.assert_not_called()
        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()
