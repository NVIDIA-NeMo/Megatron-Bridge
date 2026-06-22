"""Local conftest for nemotron_labs_diffusion unit tests.

Pre-loads megatron.bridge.diffusion.common.{cp_utils,dllm} directly from their
source files before pytest collection so the package's eager __init__.py chain
(which requires megatron.core and many GPU deps) is never triggered.
"""

import importlib.util
import sys
import types
from pathlib import Path


_SRC = Path(__file__).parents[5] / "src"  # → Megatron-Bridge/src


def _load(module_name: str, rel_path: str) -> None:
    """Load a source file directly by path and inject it into sys.modules."""
    file_path = _SRC / rel_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)


def _ensure_stub(name: str) -> None:
    """Ensure a stub package exists in sys.modules without clobbering real ones."""
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        m.__package__ = name
        sys.modules[name] = m
        parts = name.split(".")
        if len(parts) > 1:
            parent = sys.modules.get(".".join(parts[:-1]))
            if parent is not None:
                setattr(parent, parts[-1], m)


# Build the namespace hierarchy as stubs (prevents __init__.py from running)
for _ns in [
    "megatron",
    "megatron.bridge",
    "megatron.bridge.diffusion",
    "megatron.bridge.diffusion.common",
]:
    _ensure_stub(_ns)

# Load the two real modules under test directly from source
_load(
    "megatron.bridge.diffusion.common.cp_utils",
    "megatron/bridge/diffusion/common/cp_utils.py",
)
_load(
    "megatron.bridge.diffusion.common.dllm",
    "megatron/bridge/diffusion/common/dllm.py",
)
