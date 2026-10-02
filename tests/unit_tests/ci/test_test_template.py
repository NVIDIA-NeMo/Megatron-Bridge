# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise action input handling across the host and container shell boundary."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit
ACTION = Path(__file__).resolve().parents[3] / ".github/actions/test-template/action.yml"
EXPRESSION = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")


@pytest.fixture
def action_context(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    # Simulate Docker's argument boundary and container-only Bash startup. No
    # command from a test can reach a real Docker daemon or coverage executable.
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with Path('docker-args.jsonl').open('a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args[0] != 'exec':\n"
        "    sys.exit(0)\n"
        "args.pop(0)\n"
        "forwarded_env = []\n"
        "while args[0].startswith('-'):\n"
        "    option = args.pop(0)\n"
        "    if option in ('--env', '-e'):\n"
        "        forwarded_env.append(args.pop(0))\n"
        "assert args.pop(0) == 'nemo_container_123'\n"
        "if args[0] != 'bash':\n"
        "    sys.exit(0)\n"
        "env = dict(os.environ)\n"
        "if 'GH_TOKEN' not in forwarded_env:\n"
        "    env.pop('GH_TOKEN', None)\n"
        "if 'STUB_CONTAINER_BASH_ENV' in env:\n"
        "    env['BASH_ENV'] = env['STUB_CONTAINER_BASH_ENV']\n"
        "os.execvpe(args[0], args, env)\n"
    )
    docker.chmod(0o755)
    uuidgen = bin_dir / "uuidgen"
    uuidgen.write_text("#!/bin/sh\nprintf '%s\\n' test-uuid\n")
    uuidgen.chmod(0o755)
    return {
        "inputs.script": "L0_Launch_probe",
        "inputs.script_dir": "gb200/active",
        "inputs.is_unit_test": "false",
        "inputs.timeout": "1",
        "inputs.runner": "nemo-ci-gcp-gpu-x2",
        "inputs.github-token": "test-token",
        "inputs.container-image": "test-image",
        "inputs.test-data-path": "/test-data",
        "inputs.is-optional": "false",
        "inputs.shard-id": "",
        "inputs.num-shards": "",
        "github.run_id": "123",
        "inputs.is_unit_test == 'false'": "true",
    }


def _run_step(
    tmp_path: Path,
    context: dict[str, str],
    step_id: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    action = yaml.safe_load(ACTION.read_text())
    step = next(step for step in action["runs"]["steps"] if step.get("id", step["name"]) == step_id)

    def render(value: str) -> str:
        return EXPRESSION.sub(lambda match: context[match[1]], value)

    output = tmp_path / f"{step_id.replace(' ', '-')}-output"
    environment = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "GITHUB_RUN_ID": "123",
        "GITHUB_OUTPUT": str(output),
        **{key: render(value) for key, value in step.get("env", {}).items()},
        **(env or {}),
    }
    environment.pop("BASH_ENV", None)
    shell = ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail"]
    if "-u" in step["shell"].split():
        shell.append("-u")
    result = subprocess.run(
        [*shell, "-c", render(step["run"])],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if output.exists():
        for line in output.read_text().splitlines():
            key, value = line.split("=", 1)
            context[f"steps.{step_id}.outputs.{key}"] = value
    return result


def _create_probe(tmp_path: Path, context: dict[str, str], *, exit_code: int = 0) -> Path:
    created = _run_step(tmp_path, context, "create")
    assert created.returncode == 0, created.stderr
    launch = tmp_path / context["steps.create.outputs.script_path"]
    launch.parent.mkdir(parents=True)
    launch.write_text(
        "python3 - <<'PROBE'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "names = ('NCCL_ENV_PLUGIN', 'NCCL_NET_PLUGIN', 'NCCL_PROFILER_PLUGIN', 'NCCL_NET', "
        "'STARTUP_WAS_RUN', 'GH_TOKEN')\n"
        "Path('observed.json').write_text(json.dumps({key: os.environ.get(key) for key in names}))\n"
        "PROBE\n"
        f"exit {exit_code}\n"
    )
    return launch


@pytest.mark.parametrize("runner", ["nemo-ci-gcp-gpu-x2", "nemo-ci-aws-gpu-x2", "nemo-ci-azure-gpu-x2"])
@pytest.mark.parametrize("test_exit_code", [0, 7, 124])
def test_container_startup_cannot_override_gcp_test_environment(
    tmp_path: Path, action_context: dict[str, str], runner: str, test_exit_code: int
) -> None:
    action_context["inputs.runner"] = runner
    _create_probe(tmp_path, action_context, exit_code=test_exit_code)
    startup = tmp_path / "container-startup.sh"
    # PyTorch ARM startup selects cloud plugins again in each child Bash.
    startup.write_text(
        "export NCCL_ENV_PLUGIN=gcp NCCL_NET_PLUGIN=gcp NCCL_PROFILER_PLUGIN=gcp\nexport STARTUP_WAS_RUN=yes\n"
    )
    result = _run_step(
        tmp_path,
        action_context,
        "run-main-script",
        env={"STUB_CONTAINER_BASH_ENV": str(startup), "NCCL_NET": "inherited-network"},
    )
    assert result.returncode == test_exit_code, result.stderr
    assert action_context["steps.run-main-script.outputs.exit_code"] == str(test_exit_code)
    observed = json.loads((tmp_path / "observed.json").read_text())
    plugin = "none" if "gcp" in runner else "gcp"
    assert observed == {
        "NCCL_ENV_PLUGIN": plugin,
        "NCCL_NET_PLUGIN": plugin,
        "NCCL_PROFILER_PLUGIN": plugin,
        "NCCL_NET": "Socket" if "gcp" in runner else "inherited-network",
        "STARTUP_WAS_RUN": "yes",
        "GH_TOKEN": "test-token",
    }
    assert ("timed out" in result.stdout) == (test_exit_code == 124)


@pytest.mark.parametrize("script_dir", ["h100/active", "h100/flaky", "gb200/active", "gb200/flaky"])
def test_functional_script_directories(tmp_path: Path, action_context: dict[str, str], script_dir: str) -> None:
    action_context["inputs.script_dir"] = script_dir
    launch = _create_probe(tmp_path, action_context)
    assert (
        launch.relative_to(tmp_path).as_posix()
        == f"tests/functional_tests/launch_scripts/{script_dir}/L0_Launch_probe.sh"
    )
    assert action_context["steps.create.outputs.coverage-prefix"] == "e2e"
    result = _run_step(tmp_path, action_context, "run-main-script")
    assert result.returncode == 0, result.stderr


def test_unit_test_launch_script(tmp_path: Path, action_context: dict[str, str]) -> None:
    action_context.update(
        {
            "inputs.is_unit_test": "true",
            "inputs.script": "Launch_unit_tests",
            "inputs.is_unit_test == 'false'": "false",
        }
    )
    launch = _create_probe(tmp_path, action_context)
    assert launch.relative_to(tmp_path).as_posix() == "tests/unit_tests/Launch_unit_tests.sh"
    assert action_context["steps.create.outputs.coverage-prefix"] == "unit-test"
    started = _run_step(tmp_path, action_context, "Start container")
    assert started.returncode == 0, started.stderr
    calls = [json.loads(line) for line in (tmp_path / "docker-args.jsonl").read_text().splitlines()]
    start = next(args for args in calls if args[0] == "run")
    assert "--volume" not in start
    assert "HF_HOME=/home/ubuntu/.cache/huggingface" in start
    result = _run_step(tmp_path, action_context, "run-main-script")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("input_name", "value"),
    [
        ("script", "L0_$(touch proof)"),
        ("script", "L0_`touch proof`"),
        ("script", 'L0_";touch proof;#'),
        ("script", "L0_probe\ntouch proof"),
        ("script", "../L0_probe"),
        ("script", "L3_probe"),
        ("script", "L0_"),
        ("script_dir", "h100/active/../../.."),
        ("script_dir", "h100/$(touch proof)"),
        ("timeout", "$(touch proof)"),
        ("timeout", "1+1"),
        ("timeout", "0"),
        ("timeout", "01"),
        ("timeout", ""),
        ("timeout", "-1"),
        ("timeout", "1.5"),
        ("timeout", "2147483648"),
        ("timeout", "999999999999999999999"),
        ("is_unit_test", "unexpected"),
    ],
)
def test_invalid_inputs_fail_before_container_startup(
    tmp_path: Path, action_context: dict[str, str], input_name: str, value: str
) -> None:
    action_context[f"inputs.{input_name}"] = value
    result = _run_step(tmp_path, action_context, "create")
    assert result.returncode != 0
    assert not (tmp_path / "proof").exists()
    assert not (tmp_path / "docker-args.jsonl").exists()
    steps = yaml.safe_load(ACTION.read_text())["runs"]["steps"]
    assert next(index for index, step in enumerate(steps) if step.get("id") == "create") == 0


@pytest.mark.parametrize("script", ["Launch_$(touch proof)", "Launch_../probe", "L0_probe"])
def test_invalid_unit_script_names(tmp_path: Path, action_context: dict[str, str], script: str) -> None:
    action_context.update({"inputs.is_unit_test": "true", "inputs.script": script})
    result = _run_step(tmp_path, action_context, "create")
    assert result.returncode != 0
    assert not (tmp_path / "proof").exists()


@pytest.mark.parametrize("script", ["L0_$(touch proof)", "L0_`touch proof`", 'L0_";touch proof;#'])
def test_host_and_container_treat_script_path_as_data(
    tmp_path: Path, action_context: dict[str, str], script: str
) -> None:
    launch = _create_probe(tmp_path, action_context)
    renamed = launch.with_name(f"{script}.sh")
    launch.rename(renamed)
    # Bypass validation to independently verify the consumer's shell boundary.
    action_context.update(
        {"inputs.script": script, "steps.create.outputs.script_path": renamed.relative_to(tmp_path).as_posix()}
    )
    result = _run_step(tmp_path, action_context, "run-main-script")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "observed.json").exists()
    assert not (tmp_path / "proof").exists()


def test_runner_image_mount_and_token_are_literal_data(tmp_path: Path, action_context: dict[str, str]) -> None:
    action_context.update(
        {
            "inputs.runner": "nemo-ci-gcp-$(touch runner-proof)",
            "inputs.container-image": "image; touch image-proof",
            "inputs.test-data-path": "/data with spaces/$(touch mount-proof)",
            "inputs.github-token": "token'$(touch token-proof);\"",
        }
    )
    _create_probe(tmp_path, action_context)
    started = _run_step(tmp_path, action_context, "Start container")
    assert started.returncode == 0, started.stderr
    result = _run_step(tmp_path, action_context, "run-main-script")
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "docker-args.jsonl").read_text().splitlines()]
    start = next(args for args in calls if args[0] == "run")
    assert action_context["inputs.container-image"] in start
    assert f"{action_context['inputs.test-data-path']}:/home/TestData" in start
    assert f"GHA_RUNNER={action_context['inputs.runner']}" in start
    token = action_context["inputs.github-token"]
    assert json.loads((tmp_path / "observed.json").read_text())["GH_TOKEN"] == token
    assert token not in started.stdout + started.stderr + result.stdout + result.stderr
    assert all(token not in arg for args in calls for arg in args)
    assert not list(tmp_path.glob("*-proof"))
    assert not (tmp_path / "job.sh").exists()
    assert not (tmp_path / "retry_job.sh").exists()


@pytest.mark.parametrize(
    ("exit_code", "optional", "expected"),
    [("0", "false", 0), ("7", "false", 1), ("7", "true", 0), ("", "false", 1)],
)
def test_result_logging_does_not_execute_script_input(
    tmp_path: Path, action_context: dict[str, str], exit_code: str, optional: str, expected: int
) -> None:
    # Exercise result logging independently of the preceding validation step.
    action_context.update(
        {
            "inputs.script": "L0_$(touch proof)",
            "inputs.is-optional": optional,
            "steps.run-main-script.outputs.exit_code": exit_code,
            "steps.create.outputs.coverage-prefix": "e2e",
        }
    )
    result = _run_step(tmp_path, action_context, "check")
    assert result.returncode == expected, result.stderr
    assert not (tmp_path / "proof").exists()


def test_action_does_not_embed_inputs_in_shell_source() -> None:
    steps = yaml.safe_load(ACTION.read_text())["runs"]["steps"]
    for step in steps:
        for expression in EXPRESSION.findall(step.get("run", "")):
            assert "inputs." not in expression, step["name"]
