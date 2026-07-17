import ast
import re
from pathlib import Path

import pytest


_DOC_PATHS = (
    "docs/models/glm/glm45.md",
    "docs/fern/versions/nightly/pages/models/glm/glm45.mdx",
    "docs/fern/versions/0.4.2/pages/models/llm/glm45.mdx",
)
_PYTHON_BLOCK_PATTERN = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
_RECIPE_NAMES = {
    "glm45_355b_peft_config",
    "glm45_355b_pretrain_config",
    "glm45_355b_sft_config",
    "glm45_air_106b_peft_config",
    "glm45_air_106b_pretrain_config",
    "glm45_air_106b_sft_config",
}


def _public_recipe_keyword_parameters(repo_root: Path) -> dict[str, set[str]]:
    alias_path = repo_root / "src/megatron/bridge/recipes/glm/glm45.py"
    canonical_path = repo_root / "src/megatron/bridge/recipes/glm/h100/glm45.py"
    alias_tree = ast.parse(alias_path.read_text())
    canonical_tree = ast.parse(canonical_path.read_text())

    canonical_names = {
        alias.asname: alias.name
        for node in alias_tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname in _RECIPE_NAMES
    }
    canonical_functions = {node.name: node for node in canonical_tree.body if isinstance(node, ast.FunctionDef)}

    parameters = {}
    for public_name in _RECIPE_NAMES:
        function = canonical_functions[canonical_names[public_name]]
        parameters[public_name] = {
            argument.arg for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
        }
    return parameters


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _DOC_PATHS)
def test_glm45_python_recipe_calls_match_public_signatures(relative_path: str) -> None:
    repo_root = Path(__file__).parents[3]
    doc_path = repo_root / relative_path
    recipe_parameters = _public_recipe_keyword_parameters(repo_root)
    doc_text = doc_path.read_text()

    for match in _PYTHON_BLOCK_PATTERN.finditer(doc_text):
        block = match.group(1)
        block_start_line = doc_text[: match.start(1)].count("\n") + 1
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in recipe_parameters:
                continue

            unexpected_keywords = sorted(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None and keyword.arg not in recipe_parameters[node.func.id]
            )
            if unexpected_keywords:
                line_number = block_start_line + node.lineno - 1
                pytest.fail(
                    f"{relative_path}:{line_number} passes unsupported keyword arguments "
                    f"{unexpected_keywords} to public recipe {node.func.id}",
                    pytrace=False,
                )
