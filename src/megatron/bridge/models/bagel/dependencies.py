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

"""Optional dependency helpers for the official BAGEL runtime."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


_BAGEL_INSTALL_HINT = (
    "Run `uv sync --extra bagel`, clone the pinned ByteDance-Seed/Bagel source, "
    "and pass that checkout through `bagel_repo`."
)


def configure_official_bagel_repo(bagel_repo: str) -> None:
    """Validate and expose an official BAGEL source checkout."""
    repo = Path(bagel_repo).expanduser().resolve()
    if not (repo / "data").is_dir() or not (repo / "modeling").is_dir():
        raise ImportError(f"BAGEL source checkout is invalid: {repo}. {_BAGEL_INSTALL_HINT}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def import_official_bagel_module(module: str) -> ModuleType:
    """Import an official BAGEL module with an actionable error."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(f"Could not import official BAGEL module `{module}`. {_BAGEL_INSTALL_HINT}") from error
