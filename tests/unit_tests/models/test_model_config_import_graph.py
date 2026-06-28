# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).parents[3] / "src"
MODULE_FILES = {".".join(path.relative_to(SRC_ROOT).with_suffix("").parts): path for path in SRC_ROOT.rglob("*.py")}


def _top_level_imports(module: str) -> list[str]:
    tree = ast.parse(MODULE_FILES[module].read_text())
    imports = []
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            continue
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    return imports


@pytest.mark.unit
def test_model_config_family_import_graphs_are_provider_neutral():
    for model_config_path in (SRC_ROOT / "megatron/bridge").rglob("model_config.py"):
        module = ".".join(model_config_path.relative_to(SRC_ROOT).with_suffix("").parts)
        family = module.rsplit(".", 1)[0]

        pending = [module]
        package_init = f"{family}.__init__"
        if package_init in MODULE_FILES:
            for imported in _top_level_imports(package_init):
                assert "provider" not in imported, f"{package_init} eagerly imports {imported}"
        visited = set()

        while pending:
            current = pending.pop()
            if current in visited or current not in MODULE_FILES:
                continue
            visited.add(current)
            for imported in _top_level_imports(current):
                if not imported.startswith(family):
                    continue
                assert "provider" not in imported, f"{current} eagerly imports {imported}"
                pending.append(imported)
