"""CPU-only regressions for PR-controlled functional-test matrix inputs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / ".github/scripts/generate_test_matrix.py"


def _script(root: Path, *, hardware: str, directory: str, name: str, headers: str = "") -> Path:
    path = root / "tests/functional_tests/launch_scripts" / hardware / directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(headers, encoding="utf-8")
    return path


def _generate(
    root: Path, output: Path, *, hardware: str, runner_prefix: str | None = None
) -> subprocess.CompletedProcess:
    command = [sys.executable, str(GENERATOR), "--hardware", hardware]
    if runner_prefix is not None:
        command.extend(("--runner-prefix", runner_prefix))
    elif hardware == "h100":
        command.extend(("--runner-prefix", "nemo-ci-gpu-x8"))
    return subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=False,
    )


def _outputs(path: Path) -> dict:
    return {key: json.loads(value) for key, value in (line.split("=", 1) for line in path.read_text().splitlines())}


@pytest.mark.parametrize("hardware", ["h100", "gb200"])
@pytest.mark.parametrize("directory", ["active", "flaky"])
@pytest.mark.parametrize(
    "name",
    [
        "L0_$(touch proof).sh",
        "L0_`touch proof`.sh",
        "L0_safe;touch proof.sh",
        'L0_safe"quoted.sh',
        "L0_safe'quoted.sh",
        "L0_safe\nmatrix_l0=bad.sh",
        "L0_.sh",
    ],
)
def test_rejects_script_injection_before_emitting_any_output(tmp_path, hardware, directory, name):
    _script(tmp_path, hardware=hardware, directory="active", name="L0_Valid.sh")
    _script(tmp_path, hardware=hardware, directory=directory, name=name)
    output = tmp_path / "outputs"
    output.write_text("previous=value\n")

    result = _generate(tmp_path, output, hardware=hardware)

    assert result.returncode != 0
    assert "Invalid functional-test script identifier" in result.stderr
    assert output.read_text() == "previous=value\n"
    assert not (tmp_path / "proof").exists()


@pytest.mark.parametrize("hardware", ["h100", "gb200"])
@pytest.mark.parametrize("directory", ["active", "flaky"])
@pytest.mark.parametrize(
    "header",
    [
        "# CI_TIMEOUT=",
        "# CI_TIMEOUT=0",
        "# CI_TIMEOUT=-1",
        "# CI_TIMEOUT=01",
        "# CI_TIMEOUT=1.5",
        "# CI_TIMEOUT=2147483648",
        "# CI_TIMEOUT=$(touch proof)",
        '# CI_TIMEOUT=30,"runner":"injected"',
        "# GPU_COUNT=",
        "# GPU_COUNT=0",
        "# GPU_COUNT=x0",
        "# GPU_COUNT=$(touch proof)",
        '# GPU_COUNT=4","script":"injected',
    ],
)
def test_rejects_malformed_metadata(tmp_path, hardware, directory, header):
    _script(tmp_path, hardware=hardware, directory=directory, name="L0_Test.sh", headers=f"{header}\n")
    output = tmp_path / "outputs"

    result = _generate(tmp_path, output, hardware=hardware)

    assert result.returncode != 0
    assert "Invalid" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "proof").exists()


@pytest.mark.parametrize(("hardware", "gpu_count"), [("h100", "x4"), ("gb200", "4")])
def test_gpu_count_uses_hardware_specific_spelling(tmp_path, hardware, gpu_count):
    _script(tmp_path, hardware=hardware, directory="active", name="L0_Test.sh", headers=f"# GPU_COUNT={gpu_count}\n")
    result = _generate(tmp_path, tmp_path / "outputs", hardware=hardware)
    assert result.returncode != 0
    assert "Invalid GPU_COUNT" in result.stderr


@pytest.mark.parametrize(
    ("hardware", "runner_prefix", "gpu_count", "expected_runner"),
    [
        ("h100", "nemo-ci-azure-gpu-x8", "4", "nemo-ci-azure-gpu-x4"),
        ("h100", "nemo-ci-aws-gpu-x2", "4", "nemo-ci-aws-gpu-x4"),
        ("h100", "ephe-v2-gpu-x2-run42", "4", "ephe-v2-gpu-x4-run42"),
        ("h100", "ephe-v2-run42", "4", "ephe-v2-run42"),
        ("gb200", None, "x4", "nemo-ci-gcp-gpu-x4"),
    ],
)
def test_preserves_routing_timeouts_order_and_flaky_jobs(
    tmp_path, hardware, runner_prefix, gpu_count, expected_runner
):
    _script(tmp_path, hardware=hardware, directory="active", name="L0_Z.default-1.sh")
    _script(
        tmp_path,
        hardware=hardware,
        directory="active",
        name="L0_A_override.sh",
        headers=f"# GPU_COUNT={gpu_count}\n# CI_TIMEOUT=60\n",
    )
    _script(tmp_path, hardware=hardware, directory="flaky", name="L2_Flaky.sh", headers="# CI_TIMEOUT=2147483647\n")
    output = tmp_path / "outputs"

    result = _generate(tmp_path, output, hardware=hardware, runner_prefix=runner_prefix)

    assert result.returncode == 0, result.stderr
    matrices = _outputs(output)
    prefix = "matrix_" if hardware == "h100" else "matrix_gb200_"
    default_runner = runner_prefix or "nemo-ci-gcp-gpu-x2"
    assert matrices == {
        f"{prefix}l0": {
            "include": [
                {"script": "L0_A_override", "timeout": 60, "runner": expected_runner},
                {"script": "L0_Z.default-1", "timeout": 30, "runner": default_runner},
            ]
        },
        f"{prefix}l1": {"include": []},
        f"{prefix}l2": {"include": []},
        f"{prefix}flaky": {"include": [{"script": "L2_Flaky", "timeout": 2147483647, "runner": default_runner}]},
    }


@pytest.mark.parametrize("hardware", ["h100", "gb200"])
def test_empty_globs_emit_empty_matrices(tmp_path, hardware):
    output = tmp_path / "outputs"
    result = _generate(tmp_path, output, hardware=hardware)
    assert result.returncode == 0, result.stderr
    assert len(_outputs(output)) == 4
    assert all(matrix == {"include": []} for matrix in _outputs(output).values())


@pytest.mark.parametrize("hardware", ["h100", "gb200"])
def test_existing_inventory_is_valid(tmp_path, hardware):
    output = tmp_path / "outputs"
    result = _generate(REPO_ROOT, output, hardware=hardware)
    assert result.returncode == 0, result.stderr
    prefix = "matrix_" if hardware == "h100" else "matrix_gb200_"
    root = REPO_ROOT / "tests/functional_tests/launch_scripts" / hardware
    for tier in ("L0", "L1", "L2", "flaky"):
        pattern = "flaky/L*.sh" if tier == "flaky" else f"active/{tier}_*.sh"
        expected = sorted(path.stem for path in root.glob(pattern) if path.is_file())
        assert [entry["script"] for entry in _outputs(output)[f"{prefix}{tier.lower()}"]["include"]] == expected
