# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Check that model doc pages link their examples README."""

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PAGE_EXAMPLES = {
    "docs/models/gemma/diffusiongemma.md": "examples/models/diffusion_gemma",
    "docs/models/glm/glm5.md": "examples/models/glm/glm5",
    "docs/models/glm/glm5-2.md": "examples/models/glm/glm5",
    "docs/models/gpt_oss/gpt-oss-20b.md": "examples/models/gpt_oss",
    "docs/models/gpt_oss/gpt-oss-120b.md": "examples/models/gpt_oss",
    "docs/models/muse_glimmer/muse-glimmer.md": "examples/models/muse_glimmer",
    "docs/models/muse_spark_open/muse-spark-open.md": "examples/models/muse_spark_open",
    "docs/models/qwen/qwen3.6-35b-a3b.md": "examples/models/qwen/qwen35_vl",
    "docs/models/stepfun/step35.md": "examples/models/stepfun/step35",
    "docs/models/stepfun/step37.md": "examples/models/stepfun/step37",
}


@pytest.mark.parametrize(("page", "example_dir"), sorted(MODEL_PAGE_EXAMPLES.items()))
def test_model_page_links_examples_readme(page: str, example_dir: str) -> None:
    readme = f"{example_dir}/README.md"
    assert (REPO_ROOT / readme).is_file(), f"{readme} does not exist"
    page_path = REPO_ROOT / page
    assert page_path.is_file(), f"{page} does not exist"
    assert readme in page_path.read_text(encoding="utf-8"), f"{page} does not link {readme}"
