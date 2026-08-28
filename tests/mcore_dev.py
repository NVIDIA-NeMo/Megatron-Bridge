# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Helpers for tests that cover APIs available only on the MCore dev lane."""

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_commit_file(name: str) -> str:
    return (REPO_ROOT / name).read_text().strip()


def _get_submodule_commit() -> str:
    result = subprocess.run(
        ["git", "ls-tree", "HEAD", "3rdparty/Megatron-LM"],
        capture_output=True,
        check=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.split()[2]


MAIN_COMMIT = _read_commit_file(".main.commit")
DEV_COMMIT = _read_commit_file(".dev.commit")
HAS_MCORE_DEV_BRANCH = DEV_COMMIT != MAIN_COMMIT and _get_submodule_commit() == DEV_COMMIT
