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
"""Consistency checks for the published docs version switcher.

``docs/versions1.json`` is served to readers by ``docs/conf.py`` as the version
dropdown, so every entry in it is a row a reader sees and clicks.
"""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VERSIONS_JSON = REPO_ROOT / "docs" / "versions1.json"

LATEST_MARKER = "(latest)"


def _entries():
    entries = json.loads(VERSIONS_JSON.read_text(encoding="utf-8"))
    assert isinstance(entries, list) and entries, f"{VERSIONS_JSON} must be a non-empty JSON list"
    return entries


def test_exactly_one_entry_is_preferred():
    """Exactly one switcher entry carries ``preferred: true``."""
    preferred = [e["version"] for e in _entries() if e.get("preferred") is True]
    assert len(preferred) == 1, f"expected exactly one preferred entry in {VERSIONS_JSON}, found {preferred}"


def test_only_the_preferred_entry_is_named_latest():
    """The ``(latest)`` name marker sits on the preferred entry and nowhere else."""
    entries = _entries()
    named_latest = [e["version"] for e in entries if LATEST_MARKER in (e.get("name") or "").lower()]
    preferred = [e["version"] for e in entries if e.get("preferred") is True]
    assert named_latest == preferred, (
        f"entries named '{LATEST_MARKER}' are {named_latest} but the preferred entries are {preferred}; "
        f"a promotion must move the name marker and the flag together in {VERSIONS_JSON}"
    )
