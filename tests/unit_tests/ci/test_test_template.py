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
