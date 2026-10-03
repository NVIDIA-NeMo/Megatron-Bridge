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


@pytest.mark.parametrize("runner", ["nemo-ci-gcp-gpu-x2", "nemo-ci-aws-gpu-x2", "nemo-ci-azure-gpu-x2"])
@pytest.mark.parametrize("test_exit_code", [0, 7])
@pytest.mark.parametrize("is_unit_test", [False, True])
@pytest.mark.parametrize("script", ["L0_probe", "L0_$(touch proof)", 'L0_";touch proof;#'])
def test_container_startup_cannot_override_gcp_test_environment(
    tmp_path: Path, runner: str, test_exit_code: int, is_unit_test: bool, script: str
) -> None:
    action = yaml.safe_load(ACTION.read_text())
    create = next(step["run"] for step in action["runs"]["steps"] if step.get("id") == "create")
    script_dir = "unit_tests" if is_unit_test else "functional_tests/launch_scripts/gb200/active"
    if is_unit_test and script == "L0_probe":
        script = "Launch_Unit_Tests"
    expressions = {
        "inputs.is_unit_test": str(is_unit_test).lower(),
        "github.run_id": "123",
        "inputs.github-token": "test-token",
        "contains(inputs.runner, 'gcp')": str("gcp" in runner).lower(),
        "inputs.timeout": "1",
        "inputs.is_unit_test == 'true' && 'unit_tests' || format('functional_tests/launch_scripts/{0}', inputs.script_dir)": (
            script_dir
        ),
        "inputs.script": script,
        "inputs.runner": runner,
        "inputs.container-image": "test-image",
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
    launch = tmp_path / "tests" / script_dir / f"{script}.sh"
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
        "SCRIPT": script,
    }
    subprocess.run(["bash", "-c", create], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    step = next(step for step in action["runs"]["steps"] if step.get("id") == "run-main-script")
    assert step["env"]["SCRIPT"] == "${{ inputs.script }}"
    run = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda match: expressions[match[1]], step["run"])
    result = subprocess.run(["bash", "-eu", "-c", run], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == test_exit_code, result.stderr
    assert not (tmp_path / "proof").exists()
    observed = json.loads((tmp_path / "observed.json").read_text())
    plugin = "none" if "gcp" in runner else "gcp"
    assert observed == {
        "NCCL_ENV_PLUGIN": plugin,
        "NCCL_NET_PLUGIN": plugin,
        "NCCL_PROFILER_PLUGIN": plugin,
        "NCCL_NET": "Socket" if "gcp" in runner else "inherited-network",
        "STARTUP_WAS_RUN": "yes",
    }


@pytest.mark.parametrize("script", ["L0_$(touch proof)", "L0_`touch proof`", 'L0_";touch proof;#', "../probe"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_result_logging_keeps_script_names_literal(tmp_path: Path, script: str, exit_code: int) -> None:
    action = yaml.safe_load(ACTION.read_text())
    step = next(step for step in action["runs"]["steps"] if step.get("id") == "check")
    assert step["env"]["SCRIPT"] == "${{ inputs.script }}"
    expressions = {"inputs.script": script, "steps.run-main-script.outputs.exit_code": str(exit_code)}
    run = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda match: expressions.get(match[1], "false"), step["run"])
    for command in ("docker", "uuidgen"):
        stub = tmp_path / command
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "SCRIPT": script}
    env["GITHUB_OUTPUT"] = str(tmp_path / "output")
    result = subprocess.run(["bash", "-e", "-c", run], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == (0 if exit_code == 0 else 1), result.stderr
    assert not (tmp_path / "proof").exists()


@pytest.mark.parametrize("hardware", ["h100", "gb200"])
@pytest.mark.parametrize("directory", ["active", "flaky"])
@pytest.mark.parametrize("script", ["L0_probe", "L0_$(touch proof)", 'L0_"injected'])
def test_matrix_preserves_literal_script_names(tmp_path: Path, hardware: str, directory: str, script: str) -> None:
    workflow = yaml.safe_load((ACTION.parents[2] / "workflows/cicd-main.yml").read_text())
    job = "generate-test-matrix" if hardware == "h100" else "generate-gb200-test-matrix"
    run = next(step["run"] for step in workflow["jobs"][job]["steps"] if step.get("id") == "scan")
    root = tmp_path / "tests/functional_tests/launch_scripts" / hardware
    for category in ("active", "flaky"):
        (root / category).mkdir(parents=True)
    gpu_count = "4" if hardware == "h100" else "x4"
    headers = f"# CI_TIMEOUT=45\n# GPU_COUNT={gpu_count}\n"
    for tier in ("L0", "L1", "L2"):
        (root / "active" / f"{tier}_placeholder.sh").write_text(headers)
    (root / directory / f"{script}.sh").write_text(headers)
    output = tmp_path / "output"
    env = {**os.environ, "GITHUB_OUTPUT": str(output), "RUNNER_PREFIX": "nemo-ci-aws-gpu-x2"}
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", run], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "proof").exists()
    matrices = dict(line.split("=", 1) for line in output.read_text().splitlines())
    prefix = "matrix_" if hardware == "h100" else "matrix_gb200_"
    entries = json.loads(matrices[prefix + ("l0" if directory == "active" else "flaky")])["include"]
    entry = next(entry for entry in entries if entry["script"] == script)
    assert entry["timeout"] == 45
    runner_prefix = "nemo-ci-aws-gpu-x" if hardware == "h100" else "nemo-ci-gcp-gpu-x"
    assert entry["runner"] == runner_prefix + "4"
