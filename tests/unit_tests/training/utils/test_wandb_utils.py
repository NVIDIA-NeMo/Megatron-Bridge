# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for wandb_utils.

The module exposes two checkpoint callbacks (``on_save_checkpoint_success`` and
``on_load_checkpoint_success``) plus two private path helpers. None of them
need a live W&B server — the writer can be replaced with a MagicMock. These
tests verify artifact logging, tracker-file round-trip, and error swallowing.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.training.utils.wandb_utils import (
    _get_artifact_name_and_version,
    _get_wandb_artifact_tracker_filename,
    on_load_checkpoint_success,
    on_save_checkpoint_success,
)


def _make_wandb_writer():
    """Build a MagicMock that satisfies the wandb_writer attribute access pattern."""
    writer = MagicMock()
    # Artifact() returns an artifact-like object; both .add_reference and
    # writer.run.log_artifact must be callable. MagicMock chains handle this
    # by default — only explicitly set what we want to inspect.
    writer.run.entity = "my-org"
    writer.run.project = "my-proj"
    return writer


class TestPrivateHelpers:
    def test_get_artifact_name_and_version_uses_stems(self):
        save_dir = Path("/var/checkpoints/run-42")
        ckpt_path = Path("/var/checkpoints/run-42/iter_0000050.ckpt")
        name, version = _get_artifact_name_and_version(save_dir, ckpt_path)
        assert name == "run-42"
        assert version == "iter_0000050"

    def test_tracker_filename_is_under_save_dir(self, tmp_path):
        result = _get_wandb_artifact_tracker_filename(str(tmp_path))
        assert isinstance(result, Path)
        assert result.parent == tmp_path
        assert result.name == "latest_wandb_artifact_path.txt"


class TestOnSaveCheckpointSuccess:
    def test_noop_when_writer_is_none(self, tmp_path):
        # Should not raise and should not create any tracker file.
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"x")
        on_save_checkpoint_success(str(ckpt), str(tmp_path), iteration=5, wandb_writer=None)
        assert not (tmp_path / "latest_wandb_artifact_path.txt").exists()

    def test_logs_artifact_and_writes_tracker_file(self, tmp_path):
        save_dir = tmp_path / "myrun"
        save_dir.mkdir()
        ckpt = save_dir / "iter_0000010.bin"
        ckpt.write_bytes(b"x")

        writer = _make_wandb_writer()
        on_save_checkpoint_success(str(ckpt), str(save_dir), iteration=10, wandb_writer=writer)

        # An Artifact was created with the save dir's stem as name and the
        # iteration metadata.
        writer.Artifact.assert_called_once()
        call_args = writer.Artifact.call_args
        assert call_args.args[0] == "myrun"
        assert call_args.kwargs.get("type") == "model"
        assert call_args.kwargs.get("metadata") == {"iteration": 10}

        # log_artifact was called with the ckpt's stem as alias.
        writer.run.log_artifact.assert_called_once()
        log_call = writer.run.log_artifact.call_args
        assert log_call.kwargs.get("aliases") == ["iter_0000010"]

        # add_reference was called with an absolute file:// URI.
        artifact = writer.Artifact.return_value
        artifact.add_reference.assert_called_once()
        ref_arg = artifact.add_reference.call_args.args[0]
        assert ref_arg.startswith("file://")
        assert "iter_0000010.bin" in ref_arg

        # Tracker file was written with entity/project.
        tracker = save_dir / "latest_wandb_artifact_path.txt"
        assert tracker.exists()
        assert tracker.read_text() == "my-org/my-proj"

    def test_swallows_exceptions_from_wandb(self, tmp_path):
        # If wandb misbehaves, the function should log via print_rank_last
        # and return cleanly (training should not abort on artifact-log failure).
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"x")
        writer = _make_wandb_writer()
        writer.Artifact.side_effect = RuntimeError("wandb is offline")

        with patch("megatron.bridge.training.utils.wandb_utils.print_rank_last") as mock_log:
            # Must not raise.
            on_save_checkpoint_success(str(ckpt), str(tmp_path), iteration=1, wandb_writer=writer)

        mock_log.assert_called_once()
        message = mock_log.call_args.args[0]
        assert "failed to log checkpoint" in message
        assert str(ckpt) in message


class TestOnLoadCheckpointSuccess:
    def test_noop_when_writer_is_none(self, tmp_path):
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"x")
        # No exception, no side effects on disk.
        on_load_checkpoint_success(str(ckpt), str(tmp_path), wandb_writer=None)

    def test_reads_tracker_file_and_calls_use_artifact(self, tmp_path):
        load_dir = tmp_path / "myrun"
        load_dir.mkdir()
        (load_dir / "latest_wandb_artifact_path.txt").write_text("my-org/my-proj")
        ckpt = load_dir / "iter_0000025.bin"
        ckpt.write_bytes(b"x")

        writer = _make_wandb_writer()
        on_load_checkpoint_success(str(ckpt), str(load_dir), wandb_writer=writer)

        writer.run.use_artifact.assert_called_once_with("my-org/my-proj/myrun:iter_0000025")

    def test_use_artifact_called_with_empty_prefix_when_tracker_missing(self, tmp_path):
        load_dir = tmp_path / "myrun"
        load_dir.mkdir()
        ckpt = load_dir / "iter_0000025.bin"
        ckpt.write_bytes(b"x")

        writer = _make_wandb_writer()
        on_load_checkpoint_success(str(ckpt), str(load_dir), wandb_writer=writer)

        # No tracker file → artifact_path remains "" → use_artifact called with
        # just "<name>:<version>".
        writer.run.use_artifact.assert_called_once_with("myrun:iter_0000025")

    def test_swallows_exceptions_from_wandb(self, tmp_path):
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"x")
        writer = _make_wandb_writer()
        writer.run.use_artifact.side_effect = RuntimeError("wandb offline")

        with patch("megatron.bridge.training.utils.wandb_utils.print_rank_last") as mock_log:
            on_load_checkpoint_success(str(ckpt), str(tmp_path), wandb_writer=writer)

        mock_log.assert_called_once()
        message = mock_log.call_args.args[0]
        assert "failed to find checkpoint" in message
        assert str(ckpt) in message
