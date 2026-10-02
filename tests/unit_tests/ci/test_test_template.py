# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise the generated test command across the container's shell boundary."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit
ACTION = Path(__file__).resolve().parents[3] / ".github/actions/test-template/action.yml"


@pytest.mark.parametrize("hardware", ["h100", "gb200"])
@pytest.mark.parametrize("directory", ["active", "flaky"])
@pytest.mark.parametrize(
    ("headers", "expected_timeout", "override_gpu"),
    [
        ("", 30, False),
        ("# CI_TIMEOUT=\n# GPU_COUNT=\n", 30, False),
        ("# CI_TIMEOUT=60\n# GPU_COUNT={count}\n", 60, True),
        ("# CI_TIMEOUT=0\n", None, False),
        ("# CI_TIMEOUT=-1\n", None, False),
        ("# CI_TIMEOUT=01\n", None, False),
        ("# CI_TIMEOUT=1.5\n", None, False),
        ("# CI_TIMEOUT=30=bad\n", None, False),
        ("# CI_TIMEOUT=$(touch proof)\n", None, False),
        ('# CI_TIMEOUT=30,"runner":"injected"\n', None, False),
        ("# GPU_COUNT=0\n", None, False),
        ("# GPU_COUNT={wrong_count}\n", None, False),
        ("# GPU_COUNT={count}=bad\n", None, False),
        ("# GPU_COUNT=$(touch proof)\n", None, False),
        ('# GPU_COUNT={count}","script":"injected\n', None, False),
    ],
)
def test_functional_matrix_validates_numeric_headers(
    tmp_path: Path, hardware: str, directory: str, headers: str, expected_timeout: int | None, override_gpu: bool
) -> None:
    workflow = yaml.safe_load((ACTION.parents[2] / "workflows/cicd-main.yml").read_text())
    job = "generate-test-matrix" if hardware == "h100" else "generate-gb200-test-matrix"
    scan = next(step["run"] for step in workflow["jobs"][job]["steps"] if step.get("id") == "scan")
    root = tmp_path / "tests/functional_tests/launch_scripts" / hardware
    (root / "active").mkdir(parents=True)
    (root / "flaky").mkdir()
    for tier in ("L0", "L1", "L2"):
        (root / "active" / f"{tier}_Default.sh").touch()
    count, wrong_count = ("4", "x4") if hardware == "h100" else ("x4", "4")
    (root / directory / "L0_Header_probe.sh").write_text(
        headers.replace("{count}", count).replace("{wrong_count}", wrong_count)
    )
    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", scan],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(output), "RUNNER_PREFIX": "nemo-ci-gpu-x8"},
        capture_output=True,
        text=True,
    )
    assert not (tmp_path / "proof").exists()
    if expected_timeout is None:
        assert result.returncode != 0
        assert "::error::" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    matrices = [json.loads(line.split("=", 1)[1]) for line in output.read_text().splitlines()]
    entry = next(item for matrix in matrices for item in matrix["include"] if item["script"] == "L0_Header_probe")
    runner = "nemo-ci-gpu-x8" if hardware == "h100" else "nemo-ci-gcp-gpu-x2"
    assert entry == {
        "script": "L0_Header_probe",
        "timeout": expected_timeout,
        "runner": runner[:-1] + "4" if override_gpu else runner,
    }


@pytest.mark.parametrize("runner", ["nemo-ci-gcp-gpu-x2", "nemo-ci-aws-gpu-x2", "nemo-ci-azure-gpu-x2"])
@pytest.mark.parametrize("test_exit_code", [0, 7])
def test_container_startup_cannot_override_gcp_test_environment(
    tmp_path: Path, runner: str, test_exit_code: int
) -> None:
    action = yaml.safe_load(ACTION.read_text())
    create = next(step["run"] for step in action["runs"]["steps"] if step.get("id") == "create")
    expressions = {
        "inputs.is_unit_test": "false",
        "github.run_id": "123",
        "inputs.github-token": "test-token",
        "contains(inputs.runner, 'gcp')": str("gcp" in runner).lower(),
        "inputs.timeout": "1",
        "inputs.is_unit_test == 'true' && 'unit_tests' || format('functional_tests/launch_scripts/{0}', inputs.script_dir)": (
            "functional_tests/launch_scripts/gb200/active"
        ),
        "inputs.script": "probe",
    }
    create = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda match: expressions[match[1]], create)

    startup = tmp_path / "container-startup.sh"
    # PyTorch 26.08 ARM sets BASH_ENV=/etc/bash.bashrc, which sources
    # /etc/shinit_v2 and re-selects GCP plugins in each child shell.
    startup.write_text(
        "export NCCL_ENV_PLUGIN=gcp NCCL_NET_PLUGIN=gcp NCCL_PROFILER_PLUGIN=gcp\nexport STARTUP_WAS_RUN=yes\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\n# Drop exec, -t, and container name.\nshift 3\nexec "$@"\n')
    docker.chmod(0o755)
    launch = tmp_path / "tests/functional_tests/launch_scripts/gb200/active/probe.sh"
    launch.parent.mkdir(parents=True)
    # Probe the actual environment in a child process of the launch script.
    launch.write_text(
        "python3 - <<'PROBE'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "names = ('NCCL_ENV_PLUGIN', 'NCCL_NET_PLUGIN', 'NCCL_PROFILER_PLUGIN', 'NCCL_NET', 'STARTUP_WAS_RUN')\n"
        "Path('observed.json').write_text(json.dumps({key: os.environ.get(key) for key in names}))\n"
        "PROBE\n"
        f"exit {test_exit_code}\n"
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "BASH_ENV": str(startup),
        "GITHUB_OUTPUT": str(tmp_path / "github-output"),
        "NCCL_NET": "inherited-network",
    }
    subprocess.run(["bash", "-c", create], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    result = subprocess.run(["bash", "job.sh"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == test_exit_code, result.stderr
    observed = json.loads((tmp_path / "observed.json").read_text())
    plugin = "none" if "gcp" in runner else "gcp"
    assert observed == {
        "NCCL_ENV_PLUGIN": plugin,
        "NCCL_NET_PLUGIN": plugin,
        "NCCL_PROFILER_PLUGIN": plugin,
        "NCCL_NET": "Socket" if "gcp" in runner else "inherited-network",
        "STARTUP_WAS_RUN": "yes",
    }
