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

"""Check that verified card performance layouts match what their published recipes build."""

import importlib
import re
import shlex
from pathlib import Path

import pytest
import yaml

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

CARDS = Path(__file__).resolve().parents[3] / "examples" / "model_verification_cards"

# (card directory, pretrain_performance hardware leaf, perf recipe module)
PERFORMANCE_LEAVES = [
    ("deepseek-v4-flash", "GB200", "megatron.bridge.perf_recipes.deepseek"),
]


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def _recorded_layout(expected_result: str) -> tuple[int, int, int]:
    """Return the (EP, GBS, MBS) a card's expected_result records."""
    ep = re.search(r"\bEP(\d+)\b", expected_result)
    batch = re.search(r"GBS/MBS (\d+)/(\d+)", expected_result)
    assert ep and batch, f"expected_result does not record EP and GBS/MBS: {expected_result!r}"
    return int(ep.group(1)), int(batch.group(1)), int(batch.group(2))


def test_recorded_layout_parses_the_card_wording() -> None:
    assert _recorded_layout("PP1/EP32/CP1/TP1/DP4 with GBS/MBS 2048/1, FP8-MX") == (32, 2048, 1)


@pytest.mark.parametrize("card,hardware,module", PERFORMANCE_LEAVES)
def test_verified_performance_layout_matches_the_published_recipe(card: str, hardware: str, module: str) -> None:
    leaf = yaml.safe_load((CARDS / card / "card.yaml").read_text())["items"]["pretrain_performance"][hardware]
    assert leaf["status"] == "verified"
    tokens = shlex.split(leaf["command"])
    cfg = getattr(importlib.import_module(module), tokens[tokens.index("--recipe") + 1])()

    built = (cfg.model.expert_model_parallel_size, cfg.train.global_batch_size, cfg.train.micro_batch_size)
    assert _recorded_layout(leaf["expected_result"]) == built
