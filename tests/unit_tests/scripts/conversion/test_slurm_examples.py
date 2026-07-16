import os
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER_CASES = [
    (
        "examples/models/glm/glm5/slurm_conversion.sh",
        {
            "nodes": "8",
            "time": "01:00:00",
            "hf_model": "zai-org/GLM-5",
            "configs": [("2", "1", "32", None)],
            "trust_remote_code": False,
        },
    ),
    (
        "examples/models/glm47/slurm_conversion.sh",
        {
            "nodes": "4",
            "time": "04:00:00",
            "hf_model": "zai-org/GLM-4.7",
            "configs": [("1", "1", "32", None)],
            "trust_remote_code": False,
        },
    ),
    (
        "examples/models/kimi/kimi_k25_vl/slurm_conversion.sh",
        {
            "nodes": "12",
            "time": "08:00:00",
            "hf_model": "moonshotai/Kimi-K2.5",
            "configs": [("2", "1", "48", None), ("2", "2", "24", None), ("4", "1", "24", None)],
            "trust_remote_code": True,
        },
    ),
    (
        "examples/models/mimo_v2_flash/slurm_conversion.sh",
        {
            "nodes": "2",
            "time": "04:00:00",
            "hf_model": "XiaomiMiMo/MiMo-V2-Flash",
            "configs": [("2", "1", "8", "2"), ("1", "2", "8", "1"), ("2", "2", "4", "2")],
            "trust_remote_code": True,
        },
    ),
    (
        "examples/models/minimax/minimax_m2/slurm_conversion.sh",
        {
            "nodes": "2",
            "time": "04:00:00",
            "hf_model": "MiniMaxAI/MiniMax-M2",
            "configs": [("2", "1", "8", None), ("1", "2", "8", None), ("2", "2", "4", None)],
            "trust_remote_code": True,
        },
    ),
    (
        "examples/models/minimax/minimax_m3/slurm_conversion.sh",
        {
            "nodes": "4",
            "time": "00:45:00",
            "hf_model": "MiniMaxAI/MiniMax-M3",
            "configs": [("1", "1", "32", "1")],
            "trust_remote_code": True,
        },
    ),
]


def _option_values(arguments, option):
    return [arguments[index + 1] for index, argument in enumerate(arguments[:-1]) if argument == option]


def _single_option(arguments, option):
    values = _option_values(arguments, option)
    assert len(values) == 1, (option, arguments)
    return values[0]


def _parse_stub_calls(output):
    calls = []
    for line in output.splitlines():
        if line == "CALL":
            calls.append([])
        elif line.startswith("ARG="):
            calls[-1].append(line.removeprefix("ARG="))
    return calls


def test_slurm_conversion_examples_use_public_roundtrip_launcher():
    scripts = sorted(REPO_ROOT.glob("examples/models/**/slurm_conversion.sh"))

    assert scripts
    assert {str(script.relative_to(REPO_ROOT)) for script in scripts} == {case[0] for case in WRAPPER_CASES}
    for script in scripts:
        contents = script.read_text()
        assert "scripts/conversion/convert.sh" in contents, script
        assert "roundtrip" in contents, script
        assert "--executor slurm" in contents, script
        assert "--device gpu" in contents, script
        assert "hf_megatron_roundtrip_multi_gpu.py" not in contents, script
        assert "#SBATCH" not in contents, script
        assert not re.search(r"(^|\s)srun\s", contents), script


def test_slurm_conversion_readmes_use_login_node_wrappers():
    scripts = sorted(REPO_ROOT.glob("examples/models/**/slurm_conversion.sh"))

    assert scripts
    for script in scripts:
        readme = script.with_name("README.md")
        contents = readme.read_text()
        relative_script = script.relative_to(REPO_ROOT)
        assert f"bash {relative_script}" in contents, readme
        assert not re.search(r"sbatch[^\n]*slurm_conversion\.sh", contents), readme


@pytest.mark.parametrize(("relative_script", "expected"), WRAPPER_CASES)
def test_slurm_conversion_wrapper_forwards_public_launcher_args(tmp_path, relative_script, expected):
    launcher = tmp_path / "convert.sh"
    launcher.write_text('#!/usr/bin/env bash\nprintf "CALL\\n"\nprintf "ARG=%s\\n" "$@"\n')
    launcher.chmod(0o755)
    env = os.environ.copy()
    for name in ("ETP", "HF_MODEL_ID", "MODEL_NAME", "PP", "TIME_LIMIT", "TP", "EP"):
        env.pop(name, None)
    env.update(
        {
            "CONTAINER_IMAGE": "/shared/container.sqsh",
            "CONTAINER_MOUNTS": "/shared:/opt/shared",
            "CONVERT_SH": str(launcher),
            "HF_HOME": "/shared/hf-cache",
            "HF_TOKEN": "token",
            "SLURM_ACCOUNT": "account",
            "SLURM_PARTITION": "batch",
            "UV_CACHE_DIR": "/shared/uv-cache",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / relative_script),
            "--submission-dry-run",
            "--srun-arg=--mpi=pmix",
            "--srun-arg=--container-writable",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    calls = _parse_stub_calls(result.stdout)
    assert len(calls) == len(expected["configs"])
    for arguments, (tp, pp, ep, etp) in zip(calls, expected["configs"], strict=True):
        assert arguments[0] == "roundtrip"
        assert _single_option(arguments, "--executor") == "slurm"
        assert _single_option(arguments, "--device") == "gpu"
        assert _single_option(arguments, "--nodes") == expected["nodes"]
        assert _single_option(arguments, "--gpus-per-node") == "8"
        assert _single_option(arguments, "--account") == "account"
        assert _single_option(arguments, "--partition") == "batch"
        assert _single_option(arguments, "--time") == expected["time"]
        assert _single_option(arguments, "--container-image") == "/shared/container.sqsh"
        assert _single_option(arguments, "--hf-model") == expected["hf_model"]
        assert _single_option(arguments, "--tp") == tp
        assert _single_option(arguments, "--pp") == pp
        assert _single_option(arguments, "--ep") == ep
        assert _option_values(arguments, "--etp") == ([etp] if etp is not None else [])
        assert _option_values(arguments, "--mount") == [
            f"{REPO_ROOT}:/opt/Megatron-Bridge",
            "/shared:/opt/shared",
        ]
        assert set(_option_values(arguments, "--env")) >= {
            "HF_HOME",
            "HF_TOKEN",
            "NCCL_NVLS_ENABLE",
            "TORCH_NCCL_AVOID_RECORD_STREAMS",
            "UV_CACHE_DIR",
        }
        assert ("--trust-remote-code" in arguments) is expected["trust_remote_code"]
        assert not {
            "--megatron-path",
            "--megatron-load-path",
            "--megatron-save-path",
            "--output-dir",
            "--not-strict",
            "--skip-save",
            "--overwrite",
        } & set(arguments)
        assert "--submission-dry-run" in arguments
        assert [argument for argument in arguments if argument.startswith("--srun-arg=")] == [
            "--srun-arg=--mpi=pmix",
            "--srun-arg=--container-writable",
        ]


@pytest.mark.parametrize(
    ("relative_script", "overrides", "expected_model", "expected_topology"),
    [
        (
            "examples/models/glm47/slurm_conversion.sh",
            {"MODEL_NAME": "GLM-4.7-Flash"},
            "zai-org/GLM-4.7-Flash",
            ("1", "1", "32", None),
        ),
        (
            "examples/models/minimax/minimax_m3/slurm_conversion.sh",
            {"TP": "2", "PP": "2", "EP": "8", "ETP": "1"},
            "MiniMaxAI/MiniMax-M3",
            ("2", "2", "8", "1"),
        ),
    ],
)
def test_slurm_conversion_wrapper_preserves_environment_overrides(
    tmp_path,
    relative_script,
    overrides,
    expected_model,
    expected_topology,
):
    launcher = tmp_path / "convert.sh"
    launcher.write_text('#!/usr/bin/env bash\nprintf "CALL\\n"\nprintf "ARG=%s\\n" "$@"\n')
    launcher.chmod(0o755)
    env = os.environ.copy()
    for name in ("ETP", "HF_MODEL_ID", "MODEL_NAME", "PP", "TP", "EP"):
        env.pop(name, None)
    env.update(
        {
            "CONTAINER_IMAGE": "/shared/container.sqsh",
            "CONTAINER_MOUNTS": "/shared:/shared",
            "CONVERT_SH": str(launcher),
            "HF_HOME": "/shared/hf-cache",
            "SLURM_ACCOUNT": "account",
            **overrides,
        }
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / relative_script), "--submission-dry-run"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    (arguments,) = _parse_stub_calls(result.stdout)
    tp, pp, ep, etp = expected_topology
    assert _single_option(arguments, "--hf-model") == expected_model
    assert _single_option(arguments, "--tp") == tp
    assert _single_option(arguments, "--pp") == pp
    assert _single_option(arguments, "--ep") == ep
    assert _option_values(arguments, "--etp") == ([etp] if etp is not None else [])
