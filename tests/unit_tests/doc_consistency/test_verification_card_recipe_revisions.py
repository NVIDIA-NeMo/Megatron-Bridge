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
"""Check that the Nemotron 3.5 Lightning card's verified training recipes exist at the revisions it records.

Uses only pytest, PyYAML, git and the standard library (no torch / megatron import).
The revision check needs the recorded commits in the local git history; it skips only
in a shallow checkout and treats a commit missing from a complete history as a failure.
"""

import shlex
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
LIGHTNING_CARD = REPO_ROOT / "examples" / "model_verification_cards" / "nemotron-3.5-lightning" / "card.yaml"


def _verified_recipes(node: object, revision: str, path: str) -> Iterator[tuple[str, str, str]]:
    """Yield (location, recipe, effective bridge_commit) for every verified leaf command naming a recipe."""
    if not isinstance(node, dict):
        return
    revision = node.get("bridge_commit", revision)
    if node.get("status") == "verified":
        commands = node.get("commands") or [node.get("command")]
        for command in commands:
            tokens = shlex.split(command) if isinstance(command, str) else []
            if "--recipe" in tokens:
                yield path, tokens[tokens.index("--recipe") + 1], revision
    for key, value in node.items():
        yield from _verified_recipes(value, revision, f"{path}/{key}")


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a read-only git command in the repository root."""
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def _recipe_defined_at(recipe: str, revision: str) -> bool | None:
    """Return whether ``revision`` defines ``recipe`` under src/, or None when the revision is not available."""
    if _git("cat-file", "-e", f"{revision}^{{commit}}").returncode:
        return None
    return _git("grep", "-q", "-E", f"^def {recipe}\\(", revision, "--", "src/").returncode == 0


def test_verified_recipes_inherit_the_nearest_bridge_commit() -> None:
    items = {
        "pretrain": {"H100": {"status": "verified", "command": "train.sh --recipe a_config"}},
        "fsdp": {
            "GB200": {
                "status": "verified",
                "bridge_commit": "leaf",
                "variants": {
                    "bf16": {"status": "verified", "command": "train.sh --recipe b_config"},
                    "fp8": {"status": "verified", "bridge_commit": "variant", "command": "train.sh --recipe c_config"},
                    "nvfp4": {"status": "unverified", "command": "train.sh --recipe d_config"},
                },
            }
        },
    }
    assert sorted(_verified_recipes(items, "card", "items")) == [
        ("items/fsdp/GB200/variants/bf16", "b_config", "leaf"),
        ("items/fsdp/GB200/variants/fp8", "c_config", "variant"),
        ("items/pretrain/H100", "a_config", "card"),
    ]


def test_lightning_card_recipes_exist_at_their_recorded_revision() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    card = yaml.safe_load(LIGHTNING_CARD.read_text(encoding="utf-8"))
    default = card["verification_environment"]["bridge_commit"]
    leaves = list(_verified_recipes(card["items"], default, "items"))
    assert leaves, "the card names no verified training recipe"
    results = [(leaf, _recipe_defined_at(leaf[1], leaf[2])) for leaf in leaves]
    if (
        any(found is None for _, found in results)
        and _git("rev-parse", "--is-shallow-repository").stdout.strip() != "false"
    ):
        pytest.skip("a recorded bridge_commit is not in this shallow checkout's git history")
    missing = [f"{path}: {recipe} @ {revision[:10]}" for (path, recipe, revision), found in results if not found]
    assert not missing, "verified leaves name a recipe not found at their bridge_commit: " + "; ".join(missing)
