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

"""Tests for packed Parquet path resolution with shared-filesystem retry (#4207)."""

import pytest

from megatron.bridge.data.packing import paths as paths_mod
from megatron.bridge.data.packing.paths import (
    _refresh_directory_metadata,
    resolve_packed_parquet_paths_with_refresh,
)


class TestResolvePackedParquetPathsWithRefresh:
    def test_resolves_existing_file_without_retry(self, tmp_path, monkeypatch):
        shard = tmp_path / "data.idx.parquet"
        shard.touch()
        sleeps = []
        monkeypatch.setattr(paths_mod.time, "sleep", sleeps.append)

        resolved = resolve_packed_parquet_paths_with_refresh(str(shard))

        assert resolved == [str(shard)]
        assert sleeps == []

    def test_retries_until_files_become_visible(self, tmp_path, monkeypatch):
        """Simulates a shared-filesystem client that sees the file only on a later attempt."""
        shard = tmp_path / "data.idx.parquet"
        attempts = []
        sleeps = []
        monkeypatch.setattr(paths_mod.time, "sleep", sleeps.append)

        real_resolve = paths_mod._resolve_parquet_paths

        def delayed_resolve(spec):
            attempts.append(spec)
            if len(attempts) == 3:
                shard.touch()
            return real_resolve(spec)

        monkeypatch.setattr(paths_mod, "_resolve_parquet_paths", delayed_resolve)

        resolved = resolve_packed_parquet_paths_with_refresh(str(shard), max_attempts=5, backoff_s=1.0)

        assert resolved == [str(shard)]
        assert len(attempts) == 3
        # slept after the two failed attempts, with linear backoff
        assert sleeps == [1.0, 2.0]

    def test_raises_after_exhausting_attempts(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing.idx.parquet"
        sleeps = []
        monkeypatch.setattr(paths_mod.time, "sleep", sleeps.append)

        with pytest.raises(ValueError, match="not found"):
            resolve_packed_parquet_paths_with_refresh(str(missing), max_attempts=3, backoff_s=0.5)

        assert len(sleeps) == 2  # no sleep after the final attempt

    def test_directory_spec_retries_on_empty_directory(self, tmp_path, monkeypatch):
        sleeps = []
        monkeypatch.setattr(paths_mod.time, "sleep", sleeps.append)

        with pytest.raises(ValueError, match="No Parquet files found"):
            resolve_packed_parquet_paths_with_refresh(str(tmp_path), max_attempts=2, backoff_s=0.1)

        assert len(sleeps) == 1


class TestRefreshDirectoryMetadata:
    def test_refresh_existing_directory(self, tmp_path):
        _refresh_directory_metadata(str(tmp_path / "shard_*.parquet"))

    def test_refresh_missing_directory_is_noop(self, tmp_path):
        _refresh_directory_metadata(str(tmp_path / "nope" / "shard_*.parquet"))
