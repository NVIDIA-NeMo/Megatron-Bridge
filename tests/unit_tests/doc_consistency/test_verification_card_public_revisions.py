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
"""Check that every Bridge revision pinned by a model verification card is publicly retrievable."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS_DIR = REPO_ROOT / "examples" / "model_verification_cards"
PUBLIC_URL = "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
FETCH_TIMEOUT_S = 120
TOTAL_BUDGET_S = 300

# Unpublished pins tracked separately; stale entries are ignored.
KNOWN_UNPUBLISHED = frozenset(
    {
        ("k-exaone-2:items.inference.bridge_commit", "bd56432512613a88ceae9ecb05654b63955f6dcb"),
        (
            "nemotron-3.5-lightning:items.pretrain_performance.GB200.bridge_commit",
            "9b0c2dff42e6175e91d34cd2a504529f803ebde9",
        ),
    }
)


class _FetchUnavailable(Exception):
    """Raised when the public repository cannot be queried."""


def _walk_pins(node: object, path: str) -> list[tuple[str, str]]:
    """Return (dotted path, sha) for every bridge_commit key under node."""
    pins: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if key == "bridge_commit" and value is not None:
                pins.append((child, str(value)))
            else:
                pins.extend(_walk_pins(value, child))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            pins.extend(_walk_pins(value, f"{path}[{index}]"))
    return pins


def _collect_pins(cards_dir: Path = CARDS_DIR) -> dict[str, list[str]]:
    """Map each pinned sha to the card locations that pin it."""
    pins: dict[str, list[str]] = {}
    for card_path in sorted(cards_dir.glob("*/card.yaml")):
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        for location, sha in _walk_pins(card, ""):
            pins.setdefault(sha, []).append(f"{card_path.parent.name}:{location}")
    return pins


def _git_env() -> dict[str, str]:
    """Return a non-interactive git environment that never lazily fetches."""
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"}


def _present_locally(sha: str) -> bool:
    """Return whether the sha is a local commit; a locally present commit is treated as published."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
            env=_git_env(),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _fetch(repo: Path, shas: list[str], deadline: float) -> subprocess.CompletedProcess:
    """Fetch the given shas from the public repository into a bare repo before the deadline."""
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0:
        raise _FetchUnavailable("time budget exhausted")
    try:
        if not repo.exists():
            subprocess.run(["git", "init", "--bare", "-q", str(repo)], check=True, capture_output=True, timeout=30)
        return subprocess.run(
            ["git", "-C", str(repo), "fetch", "--depth=1", "--filter=tree:0", "--no-tags", PUBLIC_URL, *shas],
            capture_output=True,
            text=True,
            env=_git_env(),
            timeout=min(FETCH_TIMEOUT_S, remaining_s),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _FetchUnavailable(str(error)) from error


def _unpublished_shas(shas: list[str], repo: Path) -> set[str]:
    """Return the shas that the public repository does not serve."""
    remaining = sorted(sha for sha in shas if not _present_locally(sha))
    if not remaining:
        return set()
    deadline = time.monotonic() + TOTAL_BUDGET_S
    result = _fetch(repo, remaining, deadline)
    if result.returncode == 0:
        return set()
    if "not our ref" not in result.stderr:
        raise _FetchUnavailable(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "fetch failed")
    unpublished = set()
    for sha in remaining:
        single = _fetch(repo, [sha], deadline)
        if single.returncode == 0:
            continue
        if "not our ref" not in single.stderr:
            raise _FetchUnavailable(single.stderr.strip() or "fetch failed")
        unpublished.add(sha)
    return unpublished


def _new_unpublished_pins(pins: dict[str, list[str]], repo: Path) -> set[tuple[str, str]]:
    """Return unpublished (location, sha) pins that are not in the known baseline."""
    candidates = [
        sha for sha, locations in pins.items() if any((loc, sha) not in KNOWN_UNPUBLISHED for loc in locations)
    ]
    unpublished = _unpublished_shas(candidates, repo)
    found = {(location, sha) for sha in unpublished for location in pins[sha]}
    return found - KNOWN_UNPUBLISHED


def test_card_bridge_commits_are_public(tmp_path: Path) -> None:
    """Every card bridge_commit must be fetchable from the public Bridge repository."""
    pins = _collect_pins()
    assert pins, "no bridge_commit pins found in verification cards"
    try:
        new_pins = _new_unpublished_pins(pins, tmp_path / "probe.git")
    except _FetchUnavailable as error:
        pytest.skip(f"public repository unavailable: {error}")
    assert not new_pins, f"bridge_commit pins not in the public repository: {sorted(new_pins)}"


def test_public_revision_check_rejects_unpublished_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The check flags an unpublished pin, accepts a published one, and never fetches baseline-only pins."""
    published, missing, known = "a" * 40, "b" * 40, "c" * 40
    calls: list[list[str]] = []

    def fake_fetch(repo: Path, shas: list[str], deadline: float) -> subprocess.CompletedProcess:
        calls.append(shas)
        stderr = f"fatal: remote error: upload-pack: not our ref {missing}" if missing in shas else ""
        return subprocess.CompletedProcess([], 1 if stderr else 0, "", stderr)

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "KNOWN_UNPUBLISHED", frozenset({("card:items.k.bridge_commit", known)}))
    monkeypatch.setattr(module, "_present_locally", lambda sha: False)
    monkeypatch.setattr(module, "_fetch", fake_fetch)
    pins = {
        published: ["card:verification_environment.bridge_commit"],
        missing: ["card:items.x.bridge_commit"],
        known: ["card:items.k.bridge_commit"],
    }
    assert _new_unpublished_pins(pins, tmp_path / "probe.git") == {("card:items.x.bridge_commit", missing)}
    assert _new_unpublished_pins({published: pins[published]}, tmp_path / "probe2.git") == set()
    assert calls[0] == [published, missing]
    assert all(known not in shas for shas in calls)
