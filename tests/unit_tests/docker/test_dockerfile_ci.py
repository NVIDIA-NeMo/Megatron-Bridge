from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_dockerfile_ci_keeps_public_launchers_executable():
    dockerfile = Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.ci"
    contents = dockerfile.read_text()
    repository_copy = contents.index("COPY --chmod=644 . /opt/Megatron-Bridge")

    for launcher in ("scripts/conversion/convert.sh", "scripts/training/train.sh"):
        executable_copy = f"COPY --chmod=755 {launcher} /opt/Megatron-Bridge/{launcher}"
        assert contents.index(executable_copy) > repository_copy
