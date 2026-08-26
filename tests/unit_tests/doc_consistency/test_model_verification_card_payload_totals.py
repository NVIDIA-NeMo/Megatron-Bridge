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
"""Consistency checks for examples/model_verification_cards/*/card.yaml.

The four conversion leaves of a card all convert the one revision-pinned source
checkpoint named by model.hf_id/model.hf_revision, so a payload-byte total is a
property of that source rather than of any single run. Every conversion leaf that
publishes such a total must therefore publish the same one. Leaves outside the
conversion set are excluded: an SFT-export leaf legitimately describes a different
checkpoint.

Deliberately free of torch/megatron imports so it runs without a GPU.
"""

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS_ROOT = REPO_ROOT / "examples" / "model_verification_cards"

CONVERSION_ITEMS = (
    "hf_to_megatron_cpu",
    "hf_to_megatron_gpu",
    "megatron_to_hf_cpu",
    "megatron_to_hf_gpu",
)

# "1,498,715,010,176 payload bytes", also "... tensor-payload bytes".
PAYLOAD_BYTES = re.compile(r"([0-9][0-9,]{6,})\s+(?:[\w-]+[- ])?payload bytes")


def _card_paths() -> list[Path]:
    paths = sorted(CARDS_ROOT.glob("*/card.yaml"))
    assert paths, f"no model verification cards found under {CARDS_ROOT}"
    return paths


def _conversion_payload_totals(card_path: Path) -> dict[str, set[str]]:
    """Map each conversion leaf to the payload-byte totals its result publishes."""
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    items = card.get("items") or {}
    totals: dict[str, set[str]] = {}
    for name in CONVERSION_ITEMS:
        leaf = items.get(name)
        if not isinstance(leaf, dict):
            continue
        result = leaf.get("expected_result")
        if not isinstance(result, str):
            continue
        found = {match.group(1) for match in PAYLOAD_BYTES.finditer(result)}
        if found:
            totals[name] = found
    return totals


def test_conversion_leaves_publish_one_payload_total_per_card():
    """Conversion leaves of a card agree on the pinned source's payload-byte total."""
    for card_path in _card_paths():
        totals = _conversion_payload_totals(card_path)
        distinct = set().union(*totals.values()) if totals else set()
        assert len(distinct) <= 1, (
            f"{card_path.relative_to(REPO_ROOT)} publishes {len(distinct)} different payload-byte "
            f"totals for one immutable source checkpoint: "
            f"{ {leaf: sorted(values) for leaf, values in sorted(totals.items())} }"
        )


def test_k_exaone_2_conversion_leaves_agree():
    """k-exaone-2 import and export leaves publish the same corrected total."""
    totals = _conversion_payload_totals(CARDS_ROOT / "k-exaone-2" / "card.yaml")
    assert set(totals) == {"hf_to_megatron_gpu", "megatron_to_hf_gpu"}, (
        f"expected both k-exaone-2 GPU conversion leaves to publish a payload total, got {sorted(totals)}"
    )
    assert set().union(*totals.values()) == {"1,498,715,010,176"}, (
        f"k-exaone-2 conversion leaves disagree on the source payload total: {totals}"
    )


if __name__ == "__main__":
    # Allow standalone RED-GREEN without pytest:  python3 test_model_verification_card_payload_totals.py
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
