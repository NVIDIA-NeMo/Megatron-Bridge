import ast
import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import pytest

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global
from tests.unit_tests.recipes.test_moonlight_recipes import _FakeBridge


_REPO_ROOT = Path(__file__).parents[3]
_DOCUMENTATION_PATHS = (
    _REPO_ROOT / "docs/models/moonlight/moonlight.md",
    _REPO_ROOT / "docs/fern/versions/nightly/pages/models/moonlight/moonlight.mdx",
)
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PUBLIC_RECIPES = {
    "moonlight_16b_pretrain_config",
    "moonlight_16b_sft_config",
    "moonlight_16b_peft_config",
}


def _attribute_path(node: ast.expr) -> list[str]:
    path = []
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        path.append(node.id)
    return list(reversed(path))


@pytest.mark.parametrize("documentation_path", _DOCUMENTATION_PATHS, ids=lambda path: path.suffix)
def test_moonlight_python_examples_match_recipe_contracts(
    documentation_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moonlight_recipes = importlib.import_module("megatron.bridge.recipes.moonlight")
    example_calls = set()

    for snippet in _PYTHON_FENCE.findall(documentation_path.read_text()):
        snippet_tree = ast.parse(snippet)
        for node in ast.walk(snippet_tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in _PUBLIC_RECIPES:
                continue

            recipe_func = getattr(moonlight_recipes, node.func.id)
            recipe_module = importlib.import_module(recipe_func.__module__)
            patch_recipe_module_global(monkeypatch, recipe_module, "AutoBridge", _FakeBridge)
            kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
            inspect.signature(recipe_func).bind(**kwargs)
            cfg = recipe_func(**kwargs)
            example_calls.add(node.func.id)

            for assignment in (item for item in ast.walk(snippet_tree) if isinstance(item, ast.Assign)):
                for target in assignment.targets:
                    attribute_path = _attribute_path(target)
                    if len(attribute_path) != 3 or attribute_path[0] != "cfg":
                        continue
                    container = getattr(cfg, attribute_path[1])
                    declared_fields = {field.name for field in dataclasses.fields(container)}
                    assert attribute_path[2] in declared_fields

    assert example_calls == _PUBLIC_RECIPES
