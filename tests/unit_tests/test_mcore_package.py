"""Check namespace isolation without importing GPU or model dependencies."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib


G_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
@pytest.mark.parametrize("bridge_first", [False, True])
@pytest.mark.parametrize("include_bridge", [False, True])
def test_mcore_source_namespace(bridge_first: bool, include_bridge: bool, tmp_path: Path) -> None:
    with (G_REPO_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    mcore_root = G_REPO_ROOT / project["tool"]["uv"]["sources"]["megatron-core"]["path"]
    if not (mcore_root / "megatron/core/package_info.py").is_file():
        pytest.skip("MCore submodule is not initialized")
    source_paths = [str(mcore_root)]
    if include_bridge:
        source_paths.insert(0 if bridge_first else 1, str(G_REPO_ROOT / "src"))
    # -S removes installed packages and editable hooks. Looking up immediate children
    # of the megatron namespace does not execute their heavyweight __init__ modules.
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
import importlib.util
import json
import os
import sys
from pathlib import Path

paths = json.loads(sys.argv[1])
sys.path[:0] = paths
for name in ("core", "training"):
    spec = importlib.util.find_spec("megatron." + name)
    assert spec is not None and spec.origin, name
    assert Path(spec.origin).resolve().is_relative_to(Path(sys.argv[2]).resolve())
bridge = importlib.util.find_spec("megatron.bridge")
if sys.argv[3] == "True":
    assert bridge is not None and bridge.origin
    assert Path(bridge.origin).resolve() == Path(sys.argv[4]).resolve()
else:
    assert bridge is None, bridge
""",
            json.dumps(source_paths),
            str(G_REPO_ROOT / "3rdparty/Megatron-LM"),
            str(include_bridge),
            str(G_REPO_ROOT / "src/megatron/bridge/__init__.py"),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        check=True,
        capture_output=True,
        text=True,
    )
