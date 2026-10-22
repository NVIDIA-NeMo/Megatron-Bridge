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
"""Check that verification cards record revisions that exist in the repository history.

The Nemotron 3.5 Lightning card's verified training recipes must exist at the revisions it records,
and each verified leaf must be last verified no earlier than its recorded revision's commit date.
Every verified K-EXAONE-2 item must record a revision that exists in the repository history, and every
GLM-5 / GLM-5.2 verified GB200 training leaf that pins its own revision must pin one that exists there.
Every verified Beam model-level item must record a revision that exists in the repository history.
Every Beam command that pins a Hugging Face model or tokenizer revision must pin the card's model.hf_revision.

Uses only pytest, PyYAML, git and the standard library (no torch / megatron import).
The revision check needs the recorded commits in the local git history; it skips only
in a shallow checkout and treats a commit missing from a complete history as a failure.
"""

import datetime
import re
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
K_EXAONE_2_CARD = REPO_ROOT / "examples" / "model_verification_cards" / "k-exaone-2" / "card.yaml"
# inference is re-pinned separately in #6294; remove it from this set once that lands.
K_EXAONE_2_PENDING = {"items/inference"}
BEAM_CARD = REPO_ROOT / "examples" / "model_verification_cards" / "beam" / "card.yaml"
GLM5_CARDS = [REPO_ROOT / "examples" / "model_verification_cards" / name / "card.yaml" for name in ("glm5", "glm5-2")]
# These items have not reproduced at a public revision yet; remove each one once it is re-verified.
GLM5_PENDING = {
    "glm5": {"items/pretrain/GB200", "items/checkpoint_resume/GB200", "items/sft_long_context/GB200"},
    "glm5-2": {"items/pretrain/GB200", "items/checkpoint_resume/GB200", "items/sft_long_context/GB200"},
}


def _verified_recipes(
    node: object, revision: str, path: str, verified_on: object = None
) -> Iterator[tuple[str, str, str, object]]:
    """Yield (location, recipe, effective bridge_commit, last_verified) for every verified leaf naming a recipe."""
    if not isinstance(node, dict):
        return
    revision = node.get("bridge_commit", revision)
    verified_on = node.get("last_verified", verified_on)
    if node.get("status") == "verified":
        commands = node.get("commands") or [node.get("command")]
        for command in commands:
            tokens = shlex.split(command) if isinstance(command, str) else []
            if "--recipe" in tokens:
                yield path, tokens[tokens.index("--recipe") + 1], revision, verified_on
    for key, value in node.items():
        yield from _verified_recipes(value, revision, f"{path}/{key}", verified_on)


def _verified_revisions(node: object, revision: str, path: str) -> Iterator[tuple[str, str]]:
    """Yield (location, effective bridge_commit) for every verified leaf."""
    if not isinstance(node, dict):
        return
    revision = node.get("bridge_commit", revision)
    if node.get("status") == "verified":
        yield path, revision
    for key, value in node.items():
        yield from _verified_revisions(value, revision, f"{path}/{key}")


def _hf_revisions(node: object, path: str) -> Iterator[tuple[str, str]]:
    """Yield (location, revision) for every Hugging Face model or tokenizer revision pinned in a command."""
    if isinstance(node, list):
        for index, value in enumerate(node):
            yield from _hf_revisions(value, f"{path}/{index}")
    if not isinstance(node, dict):
        return
    commands = node.get("commands") or [node.get("command")]
    for command in commands:
        tokens = shlex.split(command) if isinstance(command, str) else []
        for index, token in enumerate(tokens):
            if token == "--hf-revision" and index + 1 < len(tokens):
                yield path, tokens[index + 1]
            elif token.startswith("--hf-revision="):
                yield path, token.split("=", 1)[1]
            elif token.lstrip("+~").startswith("tokenizer.hf_tokenizer_kwargs"):
                yield from ((path, match) for match in re.findall(r"\b[0-9a-f]{40}\b", token))
    for key, value in node.items():
        yield from _hf_revisions(value, f"{path}/{key}")


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a read-only git command in the repository root."""
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def _commit_date(revision: str) -> datetime.date:
    """Return the committer date of ``revision``."""
    return datetime.date.fromisoformat(_git("log", "-1", "--format=%cs", revision).stdout.strip())


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
                "last_verified": "leaf-date",
                "variants": {
                    "bf16": {"status": "verified", "command": "train.sh --recipe b_config"},
                    "fp8": {"status": "verified", "bridge_commit": "variant", "command": "train.sh --recipe c_config"},
                    "nvfp4": {"status": "unverified", "command": "train.sh --recipe d_config"},
                },
            }
        },
    }
    assert sorted(_verified_recipes(items, "card", "items")) == [
        ("items/fsdp/GB200/variants/bf16", "b_config", "leaf", "leaf-date"),
        ("items/fsdp/GB200/variants/fp8", "c_config", "variant", "leaf-date"),
        ("items/pretrain/H100", "a_config", "card", None),
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
    missing = [f"{path}: {recipe} @ {revision[:10]}" for (path, recipe, revision, _), found in results if not found]
    assert not missing, "verified leaves name a recipe not found at their bridge_commit: " + "; ".join(missing)
    predated = [
        f"{path}: {verified_on} < {revision[:10]}"
        for path, _, revision, verified_on in leaves
        if verified_on is None or verified_on < _commit_date(revision)
    ]
    assert not predated, "verified leaves were last verified before their bridge_commit: " + "; ".join(predated)


def test_verified_revisions_inherit_the_nearest_bridge_commit() -> None:
    items = {
        "inference": {"status": "verified"},
        "manual_forward_pass": {"status": "verified", "bridge_commit": "item"},
        "megatron_to_hf_gpu": {"status": "unverified", "bridge_commit": "other"},
    }
    assert sorted(_verified_revisions(items, "card", "items")) == [
        ("items/inference", "card"),
        ("items/manual_forward_pass", "item"),
    ]


def test_k_exaone_2_verified_items_record_reachable_revisions() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    card = yaml.safe_load(K_EXAONE_2_CARD.read_text(encoding="utf-8"))
    default = card["verification_environment"]["bridge_commit"]
    leaves = [
        leaf for leaf in _verified_revisions(card["items"], default, "items") if leaf[0] not in K_EXAONE_2_PENDING
    ]
    assert leaves, "the card records no verified item"
    missing = [
        f"{path}: {revision[:10]}"
        for path, revision in leaves
        if _git("cat-file", "-e", f"{revision}^{{commit}}").returncode
    ]
    if missing and _git("rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("a recorded bridge_commit is not in this shallow checkout's git history")
    assert not missing, "verified items record a bridge_commit missing from the repository: " + "; ".join(missing)


@pytest.mark.parametrize("card_path", GLM5_CARDS, ids=lambda path: path.parent.name)
def test_glm5_verified_items_pin_reachable_revisions(card_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    # The card-level default and model-level pins come from pull-request refs, so only training-leaf pins are checked.
    pending = GLM5_PENDING[card_path.parent.name]
    pinned = [
        leaf
        for leaf in _verified_revisions(card["items"], None, "items")
        if leaf[0].endswith("/GB200") and leaf[1] is not None
    ]
    assert pending <= {path for path, _ in pinned}, "a pending entry is not a verified GB200 leaf on the card"
    leaves = [leaf for leaf in pinned if leaf[0] not in pending]
    assert leaves, "the card records no verified GB200 training leaf that pins its own revision"
    missing = [
        f"{path}: {revision[:10]}"
        for path, revision in leaves
        if _git("cat-file", "-e", f"{revision}^{{commit}}").returncode
    ]
    if missing and _git("rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("a recorded bridge_commit is not in this shallow checkout's git history")
    assert not missing, "verified items pin a bridge_commit missing from the repository: " + "; ".join(missing)


def test_beam_verified_model_level_items_record_reachable_revisions() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    card = yaml.safe_load(BEAM_CARD.read_text(encoding="utf-8"))
    default = card["verification_environment"]["bridge_commit"]
    leaves = [leaf for leaf in _verified_revisions(card["items"], default, "items") if leaf[0].count("/") == 1]
    assert leaves, "the card records no verified model-level item"
    missing = [
        f"{path}: {revision[:10]}"
        for path, revision in leaves
        if _git("cat-file", "-e", f"{revision}^{{commit}}").returncode
    ]
    if missing and _git("rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("a recorded bridge_commit is not in this shallow checkout's git history")
    assert not missing, "verified items record a bridge_commit missing from the repository: " + "; ".join(missing)


def test_hf_revisions_collects_model_and_tokenizer_pins_only() -> None:
    model, tokenizer, dataset = "a" * 40, "b" * 40, "c" * 40
    items = {
        "conversion": {"command": f"convert.sh --hf-revision {model}"},
        "export": {"H200": {"commands": [f"convert.sh --hf-revision={model}", "generate.sh"]}},
        "sft": {
            "H200": {
                "command": f"train.sh 'dataset.hf_dataset.load_kwargs={{revision:\"{dataset}\"}}' "
                f"'++tokenizer.hf_tokenizer_kwargs.revision=\"{tokenizer}\"'"
            },
            "B200": {"command": f"train.sh 'tokenizer.hf_tokenizer_kwargs={{revision:\"{model}\"}}'"},
        },
    }
    assert sorted(_hf_revisions(items, "items")) == [
        ("items/conversion", model),
        ("items/export/H200", model),
        ("items/sft/B200", model),
        ("items/sft/H200", tokenizer),
    ]


def test_beam_commands_pin_the_card_hf_revision() -> None:
    card = yaml.safe_load(BEAM_CARD.read_text(encoding="utf-8"))
    expected = card["model"]["hf_revision"]
    pins = list(_hf_revisions(card["items"], "items"))
    assert pins, "the card has no command that pins a Hugging Face revision"
    wrong = [f"{path}: {revision[:10]}" for path, revision in pins if revision != expected]
    assert not wrong, f"commands pin a Hugging Face revision other than {expected[:10]}: " + "; ".join(wrong)
