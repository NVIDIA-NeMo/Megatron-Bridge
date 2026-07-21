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

"""Detect whether tests run against the unreleased Megatron-Core dev ref.

The nightly Megatron-Core bump opens a dev-ref variant that pins the
``3rdparty/Megatron-LM`` submodule gitlink to ``.dev.commit`` instead of
``.main.commit``. A test break present ONLY on that unreleased dev ref can be
guarded behind :data:`HAS_MCORE_DEV_BRANCH` so it is skipped on the dev-ref lane
while staying active on ``main`` — the flag is OFF by default and ON only when
the submodule points at the dev ref.
"""

import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_commit_file(name: str) -> str:
    path = _REPO_ROOT / name
    if not path.is_file():
        return ""
    return path.read_text().strip()


def _submodule_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "ls-tree", "HEAD", "3rdparty/Megatron-LM"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
    except (OSError, ValueError):
        return ""
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    parts = result.stdout.split()
    return parts[2] if len(parts) >= 3 else ""


def _detect_mcore_dev_branch() -> bool:
    """Return True only when the submodule gitlink points at the dev ref."""
    dev_commit = _read_commit_file(".dev.commit")
    main_commit = _read_commit_file(".main.commit")
    submodule_commit = _submodule_commit()
    if not dev_commit or not submodule_commit:
        return False
    return submodule_commit == dev_commit and dev_commit != main_commit


HAS_MCORE_DEV_BRANCH: bool = _detect_mcore_dev_branch()
