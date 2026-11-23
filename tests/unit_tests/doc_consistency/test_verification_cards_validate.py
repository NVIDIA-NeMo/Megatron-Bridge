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
"""Check that every shipped model verification card passes the bundled card validator."""

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS_DIR = REPO_ROOT / "examples" / "model_verification_cards"
VALIDATOR_PATH = REPO_ROOT / "skills" / "create-model-verification-card" / "scripts" / "validate_card.py"
CARDS = sorted(CARDS_DIR.glob("*/card.yaml"))


def _load_validator():
    spec = importlib.util.spec_from_file_location("shipped_card_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def test_shipped_cards_are_discovered() -> None:
    assert CARDS, f"no model verification cards found under {CARDS_DIR}"


@pytest.mark.parametrize("card_path", CARDS, ids=lambda path: path.parent.name)
def test_shipped_card_passes_validator(card_path: Path) -> None:
    raw = card_path.read_text(encoding="utf-8")
    card, errors = VALIDATOR._load_card(raw)
    if card is not None:
        errors = [*errors, *VALIDATOR._validate_card(card, raw, ())]
    assert not errors, "\n".join(sorted(set(errors)))
