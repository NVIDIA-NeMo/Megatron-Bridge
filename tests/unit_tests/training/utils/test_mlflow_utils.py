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

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.training.utils.mlflow_utils import (
    _sanitize_mlflow_metrics,
    end_active_mlflow_run,
    install_mlflow_failure_hook,
    on_load_checkpoint_success,
    on_save_checkpoint_success,
)


@pytest.mark.unit
class TestOnSaveCheckpointSuccess:
    """Test cases for on_save_checkpoint_success function."""

    def test_noop_when_mlflow_logger_is_none(self):
        """Test that the function does nothing when mlflow_logger is None."""
        # Should not raise any exception
        on_save_checkpoint_success(
            checkpoint_path="/path/to/checkpoint",
            save_dir="/path/to",
            iteration=100,
            mlflow_logger=None,
        )

    def test_logs_artifacts_with_correct_path(self):
        """Test that log_artifacts is called with correct arguments."""
        mock_mlflow = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()
            save_dir = tmpdir

            on_save_checkpoint_success(
                checkpoint_path=str(checkpoint_path),
                save_dir=save_dir,
                iteration=1000,
                mlflow_logger=mock_mlflow,
            )

            mock_mlflow.log_artifacts.assert_called_once()
            call_args = mock_mlflow.log_artifacts.call_args

            # Verify the checkpoint path is resolved
            assert call_args[0][0] == str(checkpoint_path.resolve())

            # Verify artifact_path format includes iteration
            artifact_path = call_args[1]["artifact_path"]
            base_name = Path(save_dir).name
            assert artifact_path == f"{base_name}/iter_0001000"

    def test_artifact_path_format_with_different_iterations(self):
        """Test that iteration is zero-padded to 7 digits in artifact path."""
        mock_mlflow = MagicMock()

        test_cases = [
            (0, "iter_0000000"),
            (1, "iter_0000001"),
            (999, "iter_0000999"),
            (1234567, "iter_1234567"),
            (9999999, "iter_9999999"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            for iteration, expected_suffix in test_cases:
                mock_mlflow.reset_mock()

                on_save_checkpoint_success(
                    checkpoint_path=str(checkpoint_path),
                    save_dir=tmpdir,
                    iteration=iteration,
                    mlflow_logger=mock_mlflow,
                )

                artifact_path = mock_mlflow.log_artifacts.call_args[1]["artifact_path"]
                assert artifact_path.endswith(expected_suffix), (
                    f"Expected artifact_path to end with {expected_suffix}, got {artifact_path}"
                )

    def test_uses_checkpoints_as_default_base_name(self):
        """Test that 'checkpoints' is used when save_dir has no name."""
        mock_mlflow = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            # Use root-like path that would have empty name
            on_save_checkpoint_success(
                checkpoint_path=str(checkpoint_path),
                save_dir="/",
                iteration=100,
                mlflow_logger=mock_mlflow,
            )

            artifact_path = mock_mlflow.log_artifacts.call_args[1]["artifact_path"]
            assert artifact_path.startswith("checkpoints/")

    def test_handles_exception_gracefully(self):
        """Test that exceptions are caught and logged, not raised."""
        mock_mlflow = MagicMock()
        mock_mlflow.log_artifacts.side_effect = Exception("MLFlow connection error")

        with patch("megatron.bridge.training.utils.mlflow_utils.print_rank_last") as mock_print:
            # Should not raise exception
            on_save_checkpoint_success(
                checkpoint_path="/path/to/checkpoint",
                save_dir="/path/to",
                iteration=100,
                mlflow_logger=mock_mlflow,
            )

            # Should print error message
            mock_print.assert_called_once()
            error_msg = mock_print.call_args[0][0]
            assert "Failed to log checkpoint artifacts to MLFlow" in error_msg
            assert "MLFlow connection error" in error_msg


@pytest.mark.unit
class TestOnLoadCheckpointSuccess:
    """Test cases for on_load_checkpoint_success function."""

    def test_noop_when_mlflow_logger_is_none(self):
        """Test that the function does nothing when mlflow_logger is None."""
        # Should not raise any exception
        on_load_checkpoint_success(
            checkpoint_path="/path/to/checkpoint",
            load_dir="/path/to",
            mlflow_logger=None,
        )

    def test_sets_correct_tags(self):
        """Test that set_tags is called with correct checkpoint information."""
        mock_mlflow = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()
            load_dir = tmpdir

            on_load_checkpoint_success(
                checkpoint_path=str(checkpoint_path),
                load_dir=load_dir,
                mlflow_logger=mock_mlflow,
            )

            mock_mlflow.set_tags.assert_called_once()
            tags = mock_mlflow.set_tags.call_args[0][0]

            assert "last_loaded_checkpoint" in tags
            assert "checkpoint_base_dir" in tags
            assert tags["last_loaded_checkpoint"] == str(checkpoint_path.resolve())
            assert tags["checkpoint_base_dir"] == str(Path(load_dir).resolve())

    def test_resolves_relative_paths(self):
        """Test that relative paths are resolved to absolute paths."""
        mock_mlflow = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            on_load_checkpoint_success(
                checkpoint_path=str(checkpoint_path),
                load_dir=tmpdir,
                mlflow_logger=mock_mlflow,
            )

            tags = mock_mlflow.set_tags.call_args[0][0]

            # Both paths should be absolute (resolved)
            assert Path(tags["last_loaded_checkpoint"]).is_absolute()
            assert Path(tags["checkpoint_base_dir"]).is_absolute()

    def test_handles_exception_gracefully(self):
        """Test that exceptions are caught and logged, not raised."""
        mock_mlflow = MagicMock()
        mock_mlflow.set_tags.side_effect = Exception("MLFlow API error")

        with patch("megatron.bridge.training.utils.mlflow_utils.print_rank_last") as mock_print:
            # Should not raise exception
            on_load_checkpoint_success(
                checkpoint_path="/path/to/checkpoint",
                load_dir="/path/to",
                mlflow_logger=mock_mlflow,
            )

            # Should print error message
            mock_print.assert_called_once()
            error_msg = mock_print.call_args[0][0]
            assert "Failed to record loaded checkpoint information to MLFlow" in error_msg
            assert "MLFlow API error" in error_msg


@pytest.mark.unit
class TestSanitizeMlflowMetrics:
    """Test cases for _sanitize_mlflow_metrics function."""

    def test_handles_multiple_slashes(self):
        """Test that multiple slashes in a key are all replaced."""
        metrics = {
            "train/layer/0/loss": 1.0,
            "model/encoder/attention/weight": 0.5,
        }

        result = _sanitize_mlflow_metrics(metrics)

        assert result == {
            "train/layer_0_loss": 1.0,
            "model/encoder_attention_weight": 0.5,
        }

    def test_preserves_keys_without_slashes(self):
        """Test that keys without slashes are unchanged."""
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95,
            "learning_rate": 0.001,
        }

        result = _sanitize_mlflow_metrics(metrics)

        assert result == metrics

    def test_handles_empty_dict(self):
        """Test that empty dictionary returns empty dictionary."""
        result = _sanitize_mlflow_metrics({})
        assert result == {}

    def test_preserves_values(self):
        """Test that metric values are preserved unchanged."""
        metrics = {
            "train/int_metric": 42,
            "train/float_metric": 3.14159,
            "train/string_metric": "value",
            "train/none_metric": None,
            "train/list_metric": [1, 2, 3],
        }

        result = _sanitize_mlflow_metrics(metrics)

        assert result["train/int_metric"] == 42
        assert result["train/float_metric"] == 3.14159
        assert result["train/string_metric"] == "value"
        assert result["train/none_metric"] is None
        assert result["train/list_metric"] == [1, 2, 3]

    def test_mixed_keys(self):
        """Test dictionary with both slash and non-slash keys."""
        metrics = {
            "train/loss": 0.5,
            "global_step": 1000,
            "eval/accuracy": 0.9,
            "learning_rate": 0.001,
        }

        result = _sanitize_mlflow_metrics(metrics)

        assert result == {
            "train/loss": 0.5,
            "global_step": 1000,
            "eval/accuracy": 0.9,
            "learning_rate": 0.001,
        }


@pytest.mark.unit
class TestEndActiveMlflowRun:
    """Test cases for end_active_mlflow_run function."""

    def test_noop_when_mlflow_not_installed(self):
        """Test that the function does nothing when mlflow cannot be imported."""
        # Setting sys.modules['mlflow'] = None causes `import mlflow` to raise ImportError
        with patch.dict("sys.modules", {"mlflow": None}):
            # Should not raise any exception
            end_active_mlflow_run("FAILED")

    def test_noop_when_no_active_run(self):
        """Test that end_run is not called when no MLFlow run is active."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            end_active_mlflow_run("KILLED")

        mock_mlflow.end_run.assert_not_called()

    def test_calls_end_run_with_status(self):
        """Test that end_run is called with the provided status."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            end_active_mlflow_run("KILLED")

        mock_mlflow.end_run.assert_called_once_with(status="KILLED")

    def test_suppresses_end_run_exception(self):
        """Test that exceptions inside end_run are caught and logged."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()
        mock_mlflow.end_run.side_effect = Exception("MLFlow API error")

        with (
            patch.dict("sys.modules", {"mlflow": mock_mlflow}),
            patch("megatron.bridge.training.utils.mlflow_utils.print_rank_last") as mock_print,
        ):
            # Should not raise exception
            end_active_mlflow_run("FAILED")

        # Should print error message
        mock_print.assert_called_once()
        error_msg = mock_print.call_args[0][0]
        assert "Failed to end MLFlow run with status=FAILED" in error_msg
        assert "MLFlow API error" in error_msg


@pytest.mark.unit
class TestInstallMlflowFailureHook:
    """Test cases for install_mlflow_failure_hook function."""

    @pytest.fixture(autouse=True)
    def restore_excepthook(self):
        """Snapshot and restore sys.excepthook so tests don't leak global state.

        Resets to ``sys.__excepthook__`` before each test so an earlier test
        in the suite that installed our hook (e.g. via ``state.mlflow_logger``)
        does not pollute the starting state here.
        """
        original = sys.excepthook
        sys.excepthook = sys.__excepthook__
        yield
        sys.excepthook = original

    def test_replaces_sys_excepthook_with_marked_wrapper(self):
        """Test that install replaces sys.excepthook with a marked wrapper."""
        original = sys.excepthook
        install_mlflow_failure_hook()

        assert sys.excepthook is not original
        assert getattr(sys.excepthook, "_mlflow_failure_hook", False) is True

    def test_idempotent_second_call_is_noop(self):
        """Test that a second install does not re-wrap the hook."""
        install_mlflow_failure_hook()
        first_hook = sys.excepthook

        install_mlflow_failure_hook()

        assert sys.excepthook is first_hook

    def test_hook_calls_end_active_mlflow_run_with_failed(self):
        """Test that invoking the installed hook ends the active run as FAILED."""
        install_mlflow_failure_hook()

        with patch("megatron.bridge.training.utils.mlflow_utils.end_active_mlflow_run") as mock_end:
            try:
                raise RuntimeError("simulated crash")
            except RuntimeError:
                sys.excepthook(*sys.exc_info())

        mock_end.assert_called_once_with("FAILED")

    def test_hook_chains_to_previous_excepthook(self):
        """Test that the installed hook delegates to the previous excepthook."""
        previous = MagicMock()
        sys.excepthook = previous

        install_mlflow_failure_hook()

        with patch("megatron.bridge.training.utils.mlflow_utils.end_active_mlflow_run"):
            try:
                raise RuntimeError("simulated crash")
            except RuntimeError:
                sys.excepthook(*sys.exc_info())

        # Previous excepthook ran (default traceback printing still happens)
        previous.assert_called_once()
