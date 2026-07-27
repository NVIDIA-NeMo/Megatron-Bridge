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

from unittest.mock import Mock, patch

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


class TestPretrainInProcessRestartRetry:
    """_pretrain forwards the in-process-restart-retry flag to train()."""

    @pytest.mark.parametrize(
        "wrapper_iteration, expected_retry",
        [
            (None, False),  # no in-process restart
            (0, False),  # first attempt (fresh launch or ordinary resume)
            (3, True),  # recovery re-entry
        ],
    )
    def test_forwards_is_inprocess_restart_retry(self, wrapper_iteration, expected_retry):
        state = Mock()
        state.cfg.validation.skip_train = False
        state.cfg.train.train_iters = 10
        state.train_state.do_train = True
        state.train_state.do_valid = False
        state.train_state.do_test = False

        if wrapper_iteration is None:
            wrapper = None
        else:
            wrapper = Mock()
            wrapper.iteration = wrapper_iteration

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup") as mock_setup,
            patch("megatron.bridge.training.pretrain.train") as mock_train,
            patch("megatron.bridge.training.pretrain.barrier_and_log"),
            patch("megatron.bridge.training.pretrain._finish_train"),
        ):
            mock_dist.is_initialized.return_value = True
            mock_setup.return_value.state = state

            _pretrain(state, forward_step_func=Mock(), inprocess_call_wrapper=wrapper, store=Mock())

        mock_train.assert_called_once()
        assert mock_train.call_args.kwargs["is_inprocess_restart_retry"] is expected_retry
