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

import pytest

from megatron.bridge.data.packing.paths import resolve_packed_parquet_paths


@pytest.mark.unit
def test_resolve_character_class_glob_pattern(tmp_path):
    expected = [tmp_path / f"shard_{index}.idx.parquet" for index in range(2)]
    for shard in expected:
        shard.touch()

    resolved = resolve_packed_parquet_paths(tmp_path / "shard_[01].idx.parquet")

    assert resolved == [str(shard) for shard in expected]


@pytest.mark.unit
def test_resolve_existing_literal_bracket_path(tmp_path):
    path = tmp_path / "shard_[01].idx.parquet"
    path.touch()

    assert resolve_packed_parquet_paths(path) == [str(path)]
