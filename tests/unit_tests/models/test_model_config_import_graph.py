# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).parents[3] / "src"
MODULE_FILES = {".".join(path.relative_to(SRC_ROOT).with_suffix("").parts): path for path in SRC_ROOT.rglob("*.py")}
BUILDER_AUDIT = Path(__file__).with_name("test_registered_bridge_builder_audit.py")
MIGRATED_CONFIG_NAMES = {
    "DeepSeekV2ModelConfig",
    "DeepSeekV3ModelConfig",
    "Ernie45ModelConfig",
    "GLM45ModelConfig",
    "GLM47FlashModelConfig",
    "KimiK2ModelConfig",
    "MiniMaxM2ModelConfig",
    "OlMoEModelConfig",
    "SarvamMoEModelConfig",
}


def _registered_model_config_paths() -> set[str]:
    """Read the registry audit's config-target manifest without importing models."""
    tree = ast.parse(BUILDER_AUDIT.read_text(), filename=str(BUILDER_AUDIT))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REGISTERED_BRIDGE_CONFIGS"
        ):
            manifest = ast.literal_eval(node.value)
            return {path for paths in manifest.values() for path in paths}
    raise AssertionError("REGISTERED_BRIDGE_CONFIGS manifest not found")


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
def test_registered_model_config_import_graphs_are_provider_neutral():
    for config_path in sorted(_registered_model_config_paths()):
        module = config_path.rsplit(".", 1)[0]
        assert module in MODULE_FILES, f"Registered config module does not exist: {module}"
        for imported in _top_level_imports(module):
            assert "provider" not in imported, f"{module} eagerly imports {imported}"


@pytest.mark.unit
def test_migrated_model_config_transitive_import_graphs_are_provider_neutral():
    migrated_paths = {
        path for path in _registered_model_config_paths() if path.rsplit(".", 1)[-1] in MIGRATED_CONFIG_NAMES
    }
    assert {path.rsplit(".", 1)[-1] for path in migrated_paths} == MIGRATED_CONFIG_NAMES

    for config_path in sorted(migrated_paths):
        module = config_path.rsplit(".", 1)[0]
        family = module.rsplit(".", 1)[0]
        pending = [module]
        visited = set()

        while pending:
            current = pending.pop()
            if current in visited or current not in MODULE_FILES:
                continue
            visited.add(current)
            for imported in _top_level_imports(current):
                assert "provider" not in imported, f"{current} eagerly imports {imported}"
                if imported.startswith(family):
                    pending.append(imported)
