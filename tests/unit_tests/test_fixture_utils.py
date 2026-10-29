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

from tests.functional_tests.fixture_utils import TEST_DATA_ROOT_ENV, resolve_test_data_file


def test_resolve_test_data_file_prefers_staged_file(monkeypatch, tmp_path):
    relative_path = "megatron_bridge/assets/demo.jpeg"
    staged_path = tmp_path / relative_path
    staged_path.parent.mkdir(parents=True)
    staged_path.write_bytes(b"fixture")
    monkeypatch.setenv(TEST_DATA_ROOT_ENV, str(tmp_path))

    assert resolve_test_data_file(relative_path, "https://example.com/demo.jpeg") == str(staged_path)


def test_resolve_test_data_file_uses_fallback_when_staged_file_is_missing(monkeypatch, tmp_path):
    fallback = "https://example.com/demo.jpeg"
    monkeypatch.setenv(TEST_DATA_ROOT_ENV, str(tmp_path))

    assert resolve_test_data_file("megatron_bridge/assets/demo.jpeg", fallback) == fallback
