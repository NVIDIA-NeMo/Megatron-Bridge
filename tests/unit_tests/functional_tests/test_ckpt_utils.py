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

"""Tests for checkpoint functional-test helpers."""

import pytest

from tests.functional_tests.test_groups.ckpts.utils import ensure_mcore_checkpoint_dir


pytestmark = pytest.mark.unit


def test_ensure_mcore_checkpoint_dir_creates_nested_path(tmp_path):
    """Test that the MCore checkpoint root is created recursively."""
    checkpoint_dir = tmp_path / "llama32_1b" / "mcore"

    ensure_mcore_checkpoint_dir(str(checkpoint_dir))

    assert checkpoint_dir.is_dir()


def test_ensure_mcore_checkpoint_dir_allows_existing_path(tmp_path):
    """Test that an existing MCore checkpoint root is accepted."""
    checkpoint_dir = tmp_path / "qwen3_4b" / "mcore"
    checkpoint_dir.mkdir(parents=True)

    ensure_mcore_checkpoint_dir(str(checkpoint_dir))

    assert checkpoint_dir.is_dir()
