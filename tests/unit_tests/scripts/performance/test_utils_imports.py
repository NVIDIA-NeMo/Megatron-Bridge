"""Import-boundary tests for performance launcher utilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_utils_import_does_not_eagerly_import_training_stack() -> None:
    """Headnode utilities must not import the rank-local Bridge stack."""
    module_path = Path(__file__).resolve().parents[4] / "scripts" / "performance" / "utils" / "utils.py"
    script = """
import importlib.util
import sys

module_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("isolated_perf_utils", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "recipe_runner" not in sys.modules
assert "megatron.bridge" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script, str(module_path)], check=True)
