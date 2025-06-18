# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Tests for megatron.hub.common.decorators.torchrun module."""

from unittest.mock import patch

from megatron.hub.common.decorators.torchrun import torchrun_main


class TestTorchrunMain:
    """Test cases for torchrun_main decorator."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.destroy_process_group")
    def test_successful_execution(self, mock_destroy, mock_is_init):
        """Test successful function execution with distributed cleanup."""
        mock_is_init.return_value = True

        @torchrun_main
        def main_func(x, y):
            return x + y

        result = main_func(5, 3)
        assert result == 8

        # Verify distributed cleanup
        mock_is_init.assert_called_once()
        mock_destroy.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.destroy_process_group")
    def test_no_distributed_init(self, mock_destroy, mock_is_init):
        """Test when distributed is not initialized."""
        mock_is_init.return_value = False

        @torchrun_main
        def main_func():
            return "done"

        result = main_func()
        assert result == "done"

        # Verify destroy_process_group is not called
        mock_is_init.assert_called_once()
        mock_destroy.assert_not_called()

    @patch("os._exit")
    @patch("traceback.print_exc")
    def test_exception_handling(self, mock_print_exc, mock_exit):
        """Test exception handling and hard exit."""

        @torchrun_main
        def failing_func():
            raise RuntimeError("Test error")

        # The function should call os._exit, so we need to catch that
        failing_func()

        # Verify exception handling
        mock_print_exc.assert_called_once()
        mock_exit.assert_called_once_with(1)

    def test_function_metadata_preserved(self):
        """Test that wrapped function preserves metadata."""

        @torchrun_main
        def my_function(a, b, c=10):
            """This is my function."""
            return a + b + c

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my function."

    @patch("megatron.hub.common.decorators.torchrun.record")
    def test_record_decorator_applied(self, mock_record):
        """Test that the record decorator is applied."""
        mock_record.side_effect = lambda fn: fn  # Just return the function

        @torchrun_main
        def test_func():
            return 42

        # Verify record was called
        mock_record.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.destroy_process_group")
    @patch("os._exit")
    @patch("traceback.print_exc")
    def test_exception_with_distributed(self, mock_print_exc, mock_exit, mock_destroy, mock_is_init):
        """Test exception handling when distributed is initialized."""
        mock_is_init.return_value = True

        @torchrun_main
        def failing_func():
            raise ValueError("Test distributed error")

        failing_func()

        # Verify exception handling but no normal cleanup
        mock_print_exc.assert_called_once()
        mock_exit.assert_called_once_with(1)
        mock_destroy.assert_not_called()  # Should not clean up on exception

    def test_decorator_with_args_and_kwargs(self):
        """Test decorator with various argument combinations."""

        @torchrun_main
        def complex_func(a, b, *args, x=1, **kwargs):
            return {"a": a, "b": b, "args": args, "x": x, "kwargs": kwargs}

        result = complex_func(10, 20, 30, 40, x=100, y=200, z=300)

        assert result["a"] == 10
        assert result["b"] == 20
        assert result["args"] == (30, 40)
        assert result["x"] == 100
        assert result["kwargs"] == {"y": 200, "z": 300}
