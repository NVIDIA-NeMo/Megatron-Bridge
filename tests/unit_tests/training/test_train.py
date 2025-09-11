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

"""Tests for train module utility functions."""

from unittest.mock import Mock

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from megatron.bridge.training.train import _handle_mxfp8_param_buffer_copy, should_disable_forward_pre_hook


class TestMxfp8ParamBufferCopy:
    """Unit tests for mxfp8 parameter buffer copying functionality."""

    def test_copy_main_params_called_when_both_flags_true(self):
        """Test that _copy_main_params_to_param_buffer is called when both config flags are True."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_other_optimizer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_other_optimizer,
            mock_distributed_optimizer,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_called_once()
        assert (
            not hasattr(mock_other_optimizer, "_copy_main_params_to_param_buffer")
            or not mock_other_optimizer._copy_main_params_to_param_buffer.called
        )

    def test_no_copy_when_reuse_grad_buf_false(self):
        """Test that no copying occurs when reuse_grad_buf_for_mxfp8_param_ag is False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=False, overlap_param_gather=True
        )
        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_no_copy_when_overlap_param_gather_false(self):
        """Test that no copying occurs when overlap_param_gather is False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]
        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=False
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_no_copy_when_both_flags_false(self):
        """Test that no copying occurs when both flags are False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=False, overlap_param_gather=False
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_handles_multiple_distributed_optimizers(self):
        """Test that function calls copy on multiple DistributedOptimizers."""
        mock_distributed_optimizer_1 = Mock(spec=DistributedOptimizer)
        mock_distributed_optimizer_2 = Mock(spec=DistributedOptimizer)
        mock_other_optimizer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_other_optimizer,
            mock_distributed_optimizer_1,
            mock_distributed_optimizer_2,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer_1._copy_main_params_to_param_buffer.assert_called_once()
        mock_distributed_optimizer_2._copy_main_params_to_param_buffer.assert_called_once()

    def test_only_calls_on_distributed_optimizers(self):
        """Test that only DistributedOptimizer instances get the copy call."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_regular_optimizer = Mock()  # Regular optimizer without _copy_main_params_to_param_buffer
        mock_different_optimizer = Mock()

        # Add the method to one non-DistributedOptimizer to ensure it's not called
        mock_different_optimizer._copy_main_params_to_param_buffer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_regular_optimizer,
            mock_different_optimizer,
            mock_distributed_optimizer,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_called_once()
        mock_different_optimizer._copy_main_params_to_param_buffer.assert_not_called()

        assert (
            not hasattr(mock_regular_optimizer, "_copy_main_params_to_param_buffer")
            or not mock_regular_optimizer._copy_main_params_to_param_buffer.called
        )


class TestShouldDisableForwardPreHook:
    """Unit tests for should_disable_forward_pre_hook function."""

    def test_disable_with_distributed_optimizer_and_overlap_no_fsdp(self):
        """Test that pre-hook is disabled when using distributed optimizer + overlap without FSDP."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=True, overlap_param_gather=True
        )
        assert result is True

    def test_keep_enabled_with_megatron_fsdp(self):
        """Test that pre-hook stays enabled when using Megatron FSDP."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=True, use_distributed_optimizer=True, overlap_param_gather=True
        )
        assert result is False

    def test_keep_enabled_without_distributed_optimizer(self):
        """Test that pre-hook stays enabled when not using distributed optimizer."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=False, overlap_param_gather=True
        )
        assert result is False

    def test_keep_enabled_without_overlap_param_gather(self):
        """Test that pre-hook stays enabled when not overlapping parameter gathering."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=True, overlap_param_gather=False
        )
        assert result is False

    def test_keep_enabled_all_false(self):
        """Test that pre-hook stays enabled when all conditions are false."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=False, overlap_param_gather=False
        )
        assert result is False

    def test_keep_enabled_all_true_with_fsdp(self):
        """Test that pre-hook stays enabled when FSDP is used (even with other conditions true)."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=True, use_distributed_optimizer=True, overlap_param_gather=True
        )
        assert result is False
