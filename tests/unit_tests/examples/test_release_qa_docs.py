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

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_exaone45_docs_use_current_conversion_cli_and_exact_recipes() -> None:
    example_dir = _REPO_ROOT / "examples" / "models" / "exaone" / "exaone45"
    readme = (example_dir / "README.md").read_text()
    conversion = (example_dir / "conversion.sh").read_text()

    for content in (readme, conversion):
        assert "scripts/conversion/convert.sh import" in content
        assert "examples/conversion/convert_checkpoints.py" not in content

    assert "exaone45_vl_33b_sft_16gpu_h100_bf16_config" in readme
    assert "exaone45_vl_33b_peft_4gpu_h100_bf16_config" in readme
    assert "exaone45_vl_sft_config" not in readme
    assert "exaone45_vl_peft_config" not in readme


def test_gemma4_vl_evaluation_points_to_checked_in_verification_card() -> None:
    readme_path = _REPO_ROOT / "examples" / "models" / "gemma" / "gemma4_vl" / "README.md"
    readme = readme_path.read_text()
    card = _REPO_ROOT / "examples" / "model_verification_cards" / "gemma-4-26b-a4b-it" / "card.yaml"

    assert "../../../model_verification_cards/gemma-4-26b-a4b-it/card.yaml" in readme
    assert "eval_sft_cord_v2.py" not in readme
    assert "slurm_eval_sft.sh" not in readme
    assert card.is_file()


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/models/ernie/ernie45.md",
        "docs/models/exaone/exaone.md",
        "docs/models/hy_v3/hy-v3.md",
    ],
)
def test_registered_family_doc_links_resolve(relative_path: str) -> None:
    model_index = (_REPO_ROOT / "docs" / "models" / "README.md").read_text()
    model_page = _REPO_ROOT / relative_path

    assert model_page.is_file()
    assert str(model_page.relative_to(_REPO_ROOT / "docs" / "models")) in model_index
