# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only checks for examples/models/deepseek_v4/inference.sh.

The launcher (``uv``) and the Slurm tools (``srun``, ``scontrol``) are replaced by stubs, so the
tests only exercise the shell logic: default parallelism per variant, the multi-node torchrun
arguments, the divisibility guard and the Slurm command documented in the README.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = "examples/models/deepseek_v4/inference.sh"
README = REPO_ROOT / "examples/models/deepseek_v4/README.md"
ENTRYPOINT = "examples/conversion/hf_to_megatron_generate_text.py"
MULTINODE_ARGS = {"--nproc_per_node=4", "--nnodes=8", "--node_rank=3", "--master_addr=node0"}


def _write_stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def launcher_env(tmp_path):
    """Environment whose uv/srun/scontrol are stubs; every uv call appends its arguments to UV_LOG."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_stub(fake_bin / "uv", 'printf "%s\\n" "$*" >> "$UV_LOG"\n')
    # srun: drop the srun options and run the payload once, as a single task would.
    _write_stub(fake_bin / "srun", 'while [[ "${1:-}" == --* ]]; do shift; done\nexec "$@"\n')
    _write_stub(fake_bin / "scontrol", 'printf "node0\\nnode1\\n"\n')

    env = os.environ.copy()
    for key in ("NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "GPUS_PER_NODE", "EP", "PP", "MODEL_VARIANT"):
        env.pop(key, None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["UV_LOG"] = str(tmp_path / "uv.log")
    env["WORKSPACE"] = str(tmp_path)
    return env


def _run(args, env):
    return subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def _uv_calls(env):
    log = Path(env["UV_LOG"])
    return [line.split() for line in log.read_text().splitlines()] if log.exists() else []


def _documented_slurm_command() -> str:
    match = re.search(r"`(srun --ntasks-per-node=1 bash -c '[^`]*')`", README.read_text())
    assert match, "README no longer documents the multi-node srun command"
    return match.group(1)


@pytest.mark.parametrize(("variant", "nproc"), [("DeepSeek-V4-Flash", 4), ("DeepSeek-V4-Pro", 32)])
def test_single_node_defaults(launcher_env, variant, nproc):
    result = _run(["bash", SCRIPT], {**launcher_env, "MODEL_VARIANT": variant})

    assert result.returncode == 0, result.stderr
    calls = _uv_calls(launcher_env)
    assert len(calls) == 1, "no imported checkpoint, so only the HF path should launch"
    args = calls[0]
    assert args[:4] == ["run", "python", "-m", "torch.distributed.run"]
    assert f"--nproc_per_node={nproc}" in args
    assert not any(a.startswith(("--nnodes", "--node_rank", "--master_addr", "--master_port")) for a in args)
    assert ENTRYPOINT in args
    assert (REPO_ROOT / ENTRYPOINT).is_file()


def test_multinode_both_loading_paths(launcher_env, tmp_path):
    (tmp_path / "models" / "DeepSeek-V4-Pro" / "iter_0000000").mkdir(parents=True)
    env = {**launcher_env, "MODEL_VARIANT": "DeepSeek-V4-Pro", "NNODES": "8", "NODE_RANK": "3", "MASTER_ADDR": "node0"}

    result = _run(["bash", SCRIPT], env)

    assert result.returncode == 0, result.stderr
    calls = _uv_calls(env)
    assert len(calls) == 2, "HF path and imported-checkpoint path must both launch"
    for args in calls:
        assert MULTINODE_ARGS | {"--master_port=29500", "--legacy-full-prefix", ENTRYPOINT} <= set(args)
    assert "--megatron_model_path" not in calls[0]
    assert "--megatron_model_path" in calls[1]


def test_nondivisible_nodes_exit_before_launch(launcher_env):
    result = _run(["bash", SCRIPT], {**launcher_env, "PP": "4", "EP": "8", "NNODES": "3"})

    assert result.returncode != 0
    assert "not divisible by NNODES=3" in result.stderr
    assert _uv_calls(launcher_env) == []


def test_documented_slurm_command_runs_from_repo_root(launcher_env):
    command = _documented_slurm_command()
    assert SCRIPT in command, "the documented command must use the root-relative script path"
    assert "./inference.sh" not in (REPO_ROOT / SCRIPT).read_text(), "header example must match the README"
    env = {
        **launcher_env,
        "MODEL_VARIANT": "DeepSeek-V4-Pro",
        "SLURM_JOB_NUM_NODES": "8",
        "SLURM_NODEID": "3",
        "SLURM_JOB_NODELIST": "node[0-7]",
    }

    result = _run(["bash", "-c", command], env)

    assert result.returncode == 0, result.stderr
    calls = _uv_calls(env)
    assert len(calls) == 1
    assert MULTINODE_ARGS | {ENTRYPOINT} <= set(calls[0])
