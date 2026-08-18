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

Deliberately stdlib-only (no torch / megatron import) so it scans the file
directly and runs anywhere.
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


def test_every_entry_url_names_its_own_version():
    """Each entry's URL contains its own version rather than a moving alias."""
    mismatched = [(e["version"], e["url"]) for e in _entries() if e["version"] not in e["url"]]
    assert not mismatched, (
        f"these {VERSIONS_JSON} entries point at a URL that does not name their own version: {mismatched}"
    )


if __name__ == "__main__":
    # Allow standalone RED-GREEN without pytest/torch:  python3 test_docs_version_picker.py
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
