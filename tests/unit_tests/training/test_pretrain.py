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

import logging
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

    @patch("megatron.bridge.training.pretrain.dist")
    def test_abort_does_not_wait_for_outstanding_collectives(self, mock_dist):
        """Test failure cleanup aborts process groups without synchronization."""
        mock_dist.is_initialized.return_value = True

        _maybe_destroy_process_group(should_destroy=True, synchronize=False, abort=True)

        mock_dist.distributed_c10d._abort_process_group.assert_called_once_with()
        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()


class TestPretrainProcessGroupOwnership:
    """Test process group ownership across exceptional pretrain exits."""

    def test_framework_owned_failure_is_logged_before_cleanup_and_abort(self, caplog):
        """Test Bridge logs the original failure before cleaning up framework-owned state."""
        state = MagicMock()
        async_queue = MagicMock()
        state._async_calls_queue = async_queue
        original_error = RuntimeError("setup failed after distributed initialization")

        def assert_failure_was_logged():
            failure_records = [
                record for record in caplog.records if record.getMessage().startswith("Pretrain failed")
            ]
            assert len(failure_records) == 1

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.destroy_global_state") as mock_destroy_global_state,
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup", side_effect=original_error),
        ):
            mock_dist.is_initialized.side_effect = [False, True, True]
            mock_dist.get_rank.return_value = 7
            async_queue.close.side_effect = lambda *, abort: assert_failure_was_logged()
            mock_destroy_global_state.side_effect = assert_failure_was_logged
            mock_dist.distributed_c10d._abort_process_group.side_effect = assert_failure_was_logged

            with (
                caplog.at_level(logging.ERROR, logger="megatron.bridge.training.pretrain"),
                pytest.raises(RuntimeError) as exc_info,
            ):
                _pretrain(state, MagicMock())

        assert exc_info.value is original_error
        mock_destroy_global_state.assert_called_once_with()
        mock_dist.barrier.assert_not_called()
        mock_dist.distributed_c10d._abort_process_group.assert_called_once_with()
        mock_dist.destroy_process_group.assert_not_called()
        failure_record = next(record for record in caplog.records if record.getMessage().startswith("Pretrain failed"))
        failure_message = failure_record.getMessage()
        assert "rank=7" in failure_message
        assert "RuntimeError: setup failed after distributed initialization" in failure_message
        assert "aborting framework-owned distributed resources" in failure_message
        assert "peer ranks may be blocked in collectives" in failure_message
        assert failure_record.exc_info is not None
        assert failure_record.exc_info[1] is original_error
        assert "Traceback (most recent call last)" in caplog.text
        assert "RuntimeError: setup failed after distributed initialization" in caplog.text

    def test_cleanup_failure_does_not_replace_original_failure(self, caplog):
        """Test cleanup errors are logged while the original pretrain failure propagates."""
        state = MagicMock()
        original_error = RuntimeError("original setup failure")

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch(
                "megatron.bridge.training.pretrain.destroy_global_state",
                side_effect=RuntimeError("cleanup failure"),
            ),
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup", side_effect=original_error),
        ):
            mock_dist.is_initialized.side_effect = [False, True, True]
            mock_dist.get_rank.return_value = 2

            with (
                caplog.at_level(logging.ERROR, logger="megatron.bridge.training.pretrain"),
                pytest.raises(RuntimeError) as exc_info,
            ):
                _pretrain(state, MagicMock())

        assert exc_info.value is original_error
        messages = [record.getMessage() for record in caplog.records]
        assert messages[0].startswith("Pretrain failed on rank=2 with RuntimeError: original setup failure")
        assert "Failed to destroy Megatron global state after pretrain failure" in messages

    def test_framework_owned_async_worker_is_aborted_before_process_groups(self):
        """Test failed setup aborts its async checkpoint worker before distributed cleanup."""
        state = MagicMock()
        async_queue = MagicMock()
        cleanup_order = []

        def setup_then_fail(*_args, **_kwargs):
            state._async_calls_queue = async_queue
            raise RuntimeError("setup failed after async worker initialization")

        async_queue.close.side_effect = lambda *, abort: cleanup_order.append(("async_queue", abort))

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.destroy_global_state"),
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup", side_effect=setup_then_fail),
        ):
            mock_dist.is_initialized.side_effect = [False, True, True]
            mock_dist.distributed_c10d._abort_process_group.side_effect = lambda: cleanup_order.append(
                ("process_group", "abort")
            )

            with pytest.raises(RuntimeError, match="setup failed after async worker initialization"):
                _pretrain(state, MagicMock())

        assert cleanup_order == [("async_queue", True), ("process_group", "abort")]
        assert state._async_calls_queue is None

    def test_framework_owned_failure_shuts_down_fault_tolerance_before_process_groups(self):
        """Test failed setup disconnects fault-tolerance monitoring before distributed cleanup."""
        state = MagicMock()
        rank_monitor_client = MagicMock()
        rank_monitor_client.is_initialized = True
        cleanup_order = []

        def setup_then_fail(*_args, **_kwargs):
            state.rank_monitor_client = rank_monitor_client
            raise RuntimeError("setup failed after fault-tolerance initialization")

        rank_monitor_client.shutdown_workload_monitoring.side_effect = lambda: cleanup_order.append("fault_tolerance")

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.destroy_global_state"),
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup", side_effect=setup_then_fail),
        ):
            mock_dist.is_initialized.side_effect = [False, True, True]
            mock_dist.distributed_c10d._abort_process_group.side_effect = lambda: cleanup_order.append("process_group")

            with pytest.raises(RuntimeError, match="setup failed after fault-tolerance initialization"):
                _pretrain(state, MagicMock())

        assert cleanup_order == ["fault_tolerance", "process_group"]
        assert state.rank_monitor_client is None

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
            patch("megatron.bridge.training.pretrain.logger.exception") as mock_logger_exception,
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
        mock_logger_exception.assert_not_called()
        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()
