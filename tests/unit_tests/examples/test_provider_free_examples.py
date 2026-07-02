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

"""Keep builder-backed model construction as the default in examples."""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

EXAMPLES_ROOT = Path(__file__).parents[3] / "examples"
REPO_ROOT = EXAMPLES_ROOT.parent
DOCS_ROOT = REPO_ROOT / "docs"
NIGHTLY_DOCS_ROOT = DOCS_ROOT / "fern" / "versions" / "nightly" / "pages"

# Remove entries as these integrations gain standalone ModelConfig/ModelBuilder
# support or no longer require a persistent provider-based HF loading hook.
TEMPORARY_FALLBACKS = {
    "to_megatron_model": {
        Path("models/llada/llada15/convert_llada15_hf_to_megatron.py"),
    },
    "to_megatron_provider": {
        Path("distillation/llama/distill_llama32_3b-1b.py"),
        Path("models/diffusion/inference_dllm.py"),
    },
    "provider_bridge": {
        Path("models/flux/conversion/convert_checkpoints.py"),
        Path("models/nemotron_labs_diffusion/inference_nemotron_labs_diffusion.py"),
        Path("models/wan/conversion/convert_checkpoints.py"),
    },
    "provide_distributed_model": {
        Path("models/flux/conversion/convert_checkpoints.py"),
        Path("models/wan/conversion/convert_checkpoints.py"),
    },
    "direct_model_provider_import": {
        Path("megatron_mimo/llava/megatron_mimo_training_llava.py"),
        Path("megatron_mimo/llava/megatron_mimo_training_llava_audio.py"),
    },
}


def test_examples_only_use_allowlisted_provider_fallbacks() -> None:
    calls: dict[str, set[Path]] = {name: set() for name in TEMPORARY_FALLBACKS}

    for path in EXAMPLES_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("megatron.bridge.models."):
                if any(alias.name.endswith("Provider") for alias in node.names):
                    calls["direct_model_provider_import"].add(path.relative_to(EXAMPLES_ROOT))
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in calls:
                calls[node.func.attr].add(path.relative_to(EXAMPLES_ROOT))

    assert calls == TEMPORARY_FALLBACKS


def test_examples_do_not_use_removed_model_construction_aliases() -> None:
    forbidden_calls = {"to_megatron_model", "to_megatron_model_config"}
    violations: list[str] = []

    for path in EXAMPLES_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in forbidden_calls:
                relative_path = path.relative_to(EXAMPLES_ROOT)
                if relative_path in TEMPORARY_FALLBACKS.get(node.func.attr, set()):
                    continue
                violations.append(f"{relative_path}:{node.lineno}:{node.func.attr}")

    assert not violations, "Deprecated model construction aliases remain: " + ", ".join(violations)


def test_user_docs_default_to_builder_backed_model_construction() -> None:
    violations: list[str] = []
    paths = [REPO_ROOT / "README.md", *DOCS_ROOT.rglob("*.md"), *NIGHTLY_DOCS_ROOT.rglob("*.mdx")]
    compatibility_references = {
        DOCS_ROOT / "bridge-guide.md",
        NIGHTLY_DOCS_ROOT / "bridge-guide.mdx",
    }
    for path in paths:
        if path in compatibility_references:
            # The API reference intentionally documents deprecated compatibility aliases.
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "to_megatron_provider(" in line or "provide_distributed_model(" in line:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")

    assert not violations, "User docs still default to provider construction: " + ", ".join(violations)
