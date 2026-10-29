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

import ast
import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


G_REPO_ROOT = Path(__file__).resolve().parents[3]
G_MODEL_SIZE_TOKEN_PATTERN = re.compile(r"(?:A\d+B|\d+(?:P\d+|_\d+)?[BM])(?:v\d+)?(?=$|[A-Z_])")
G_PROVIDER_SOURCE_ROOTS = (
    G_REPO_ROOT / "src" / "megatron" / "bridge" / "models",
    G_REPO_ROOT / "src" / "megatron" / "bridge" / "diffusion" / "models",
)


def _encodes_model_size(class_name: str) -> bool:
    return "Provider" in class_name and G_MODEL_SIZE_TOKEN_PATTERN.search(class_name) is not None


@pytest.mark.parametrize(
    "class_name",
    [
        "T5ModelProvider",
        "Gemma3ModelProvider",
        "Gemma4ModelProvider",
        "GLM5ModelProvider",
        "Ministral3ModelProvider",
        "Qwen25VLModelProvider",
    ],
)
def test_provider_class_version_numbers_are_not_model_sizes(class_name: str) -> None:
    assert not _encodes_model_size(class_name)


@pytest.mark.parametrize(
    "class_name",
    [
        "GPTProvider175B",
        "GemmaModelProvider2B",
        "Ministral3ModelProvider14B",
        "NemotronNano12Bv2VLModelProvider",
        "WanModelProvider1_3B",
        "Qwen3A3BModelProvider",
    ],
)
def test_provider_class_explicit_size_tokens_are_model_sizes(class_name: str) -> None:
    assert _encodes_model_size(class_name)


def test_public_provider_classes_do_not_encode_model_size() -> None:
    """Provider classes should not expose hardcoded model-size variants."""
    violations = []
    for source_root in G_PROVIDER_SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(module):
                if isinstance(node, ast.ClassDef) and _encodes_model_size(node.name):
                    violations.append(f"{path.relative_to(G_REPO_ROOT)}:{node.lineno}:{node.name}")

    assert violations == []
