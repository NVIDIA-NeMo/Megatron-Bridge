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

import os
from pathlib import Path


DEFAULT_TEST_DATA_ROOT = Path("/home/TestData")
TEST_DATA_ROOT_ENV = "NEMO_TEST_DATA_ROOT"


def get_test_data_root() -> Path:
    """Return the configured shared TestData root."""
    return Path(os.environ.get(TEST_DATA_ROOT_ENV) or DEFAULT_TEST_DATA_ROOT)


def resolve_test_data_file(relative_path: str | Path, fallback: str) -> str:
    """Resolve a staged TestData file before using its egress fallback.

    Args:
        relative_path: File path relative to the TestData root.
        fallback: Value to use when the staged file is unavailable.

    Returns:
        The staged file path when it exists, otherwise ``fallback``.
    """
    staged_path = get_test_data_root() / relative_path
    return str(staged_path) if staged_path.is_file() else fallback
