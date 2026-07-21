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

"""Detect whether the pinned Megatron-Core submodule tracks the ``dev`` ref.

The nightly Megatron-Core bump keeps a single ``3rdparty/Megatron-LM`` submodule
with two recorded refs, ``.main.commit`` and ``.dev.commit``. On the dev-ref bump
branch the submodule gitlink is advanced to ``.dev.commit``; on ``main`` (and once
the dev branch reverts its gitlink back to ``main`` in phase B) it tracks
``.main.commit`` instead.

``HAS_MCORE_DEV_BRANCH`` is ``True`` only on that dev-ref state, so the handful of
tests that exercise genuinely-unreleased Megatron-Core ``dev`` features can be
skipped on the dev lane without ever affecting ``main``.
"""

import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMODULE_PATH = "3rdparty/Megatron-LM"


def _read_commit(filename: str) -> str | None:
    """Return the stripped contents of a pinned-ref file, or ``None`` when absent."""
    try:
        return (_REPO_ROOT / filename).read_text().strip() or None
    except OSError:
        return None


def _gitlink_commit() -> str | None:
    """Return the submodule gitlink SHA recorded in the working tree, or ``None``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", f"HEAD:{_SUBMODULE_PATH}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _has_mcore_dev_branch() -> bool:
    """Return ``True`` when the submodule gitlink tracks ``.dev.commit`` and not ``.main.commit``."""
    gitlink = _gitlink_commit()
    dev_commit = _read_commit(".dev.commit")
    main_commit = _read_commit(".main.commit")
    return bool(gitlink and dev_commit and gitlink == dev_commit and dev_commit != main_commit)


HAS_MCORE_DEV_BRANCH = _has_mcore_dev_branch()
