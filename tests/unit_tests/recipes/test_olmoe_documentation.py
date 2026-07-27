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


_REPO_ROOT = Path(__file__).parents[3]
_ALIAS_PATH = _REPO_ROOT / "src/megatron/bridge/recipes/olmoe/olmoe_7b.py"
_IMPLEMENTATION_PATH = _REPO_ROOT / "src/megatron/bridge/recipes/olmoe/h100/olmoe_7b.py"
_DOCUMENTATION_PATHS = (
    _REPO_ROOT / "docs/models/olmoe/olmoe.md",
    _REPO_ROOT / "docs/fern/versions/nightly/pages/models/olmoe/olmoe.mdx",
)
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PUBLIC_RECIPES = {
    "olmoe_7b_pretrain_config",
    "olmoe_7b_sft_config",
    "olmoe_7b_peft_config",
}


@pytest.mark.parametrize("documentation_path", _DOCUMENTATION_PATHS, ids=lambda path: path.suffix)
def test_olmoe_python_examples_match_recipe_signatures(documentation_path: Path) -> None:
    alias_tree = ast.parse(_ALIAS_PATH.read_text())
    alias_targets = {
        alias.asname: alias.name
        for node in alias_tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname is not None
    }

    implementation_tree = ast.parse(_IMPLEMENTATION_PATH.read_text())
    recipe_parameters = {
        node.name: {argument.arg for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        for node in implementation_tree.body
        if isinstance(node, ast.FunctionDef)
    }

    example_calls = set()
    for snippet in _PYTHON_FENCE.findall(documentation_path.read_text()):
        snippet_tree = ast.parse(snippet)
        for node in ast.walk(snippet_tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in _PUBLIC_RECIPES:
                continue

            target = alias_targets[node.func.id]
            unsupported = {keyword.arg for keyword in node.keywords} - recipe_parameters[target]
            assert not unsupported, f"{node.func.id} passes unsupported recipe keywords: {sorted(unsupported)}"
            example_calls.add(node.func.id)

    assert example_calls == _PUBLIC_RECIPES
