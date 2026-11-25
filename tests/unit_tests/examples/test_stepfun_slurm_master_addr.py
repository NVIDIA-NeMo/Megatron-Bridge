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

import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

SCRIPTS = [
    "examples/models/stepfun/step35/slurm_pretrain.sh",
    "examples/models/stepfun/step37/slurm_pretrain.sh",
]

FAKE_SRUN = """#!/bin/bash
touch "$SRUN_MARKER"
cmd="${@: -1}"
exec env -i PATH="$INNER_PATH" SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 \\
    SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST" MASTER_ADDR_FILE="$MASTER_ADDR_FILE" bash -c "$cmd"
"""


def _write_exe(path, content):
    path.write_text(content)
    path.chmod(0o755)


def _run_launcher(tmp_path, script, hostnames):
    outer_bin = tmp_path / "outer_bin"
    inner_bin = tmp_path / "inner_bin"
    outer_bin.mkdir()
    inner_bin.mkdir()
    _write_exe(outer_bin / "scontrol", f"#!/bin/bash\nprintf '{hostnames}'\n")
    _write_exe(outer_bin / "srun", FAKE_SRUN)
    _write_exe(inner_bin / "uv", '#!/bin/bash\nprintf "%s" "$MASTER_ADDR" > "$MASTER_ADDR_FILE"\n')
    (inner_bin / "bash").symlink_to(shutil.which("bash"))
    assert shutil.which("scontrol", path=str(inner_bin)) is None

    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "MASTER_"))}
    env.update(
        {
            "PATH": f"{outer_bin}:{env['PATH']}",
            "INNER_PATH": str(inner_bin),
            "CONTAINER_IMAGE": "test.sqsh",
            "WORKSPACE": str(tmp_path),
            "MEGATRON_BRIDGE_PATH": str(tmp_path),
            "SLURM_JOB_ID": "1",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_GPUS_PER_NODE": "1",
            "SLURM_JOB_NODELIST": "node-[a-b]",
            "MASTER_ADDR_FILE": str(tmp_path / "master_addr"),
            "SRUN_MARKER": str(tmp_path / "srun_called"),
        }
    )
    return subprocess.run(["bash", str(REPO_ROOT / script)], cwd=tmp_path, capture_output=True, env=env, text=True)


@pytest.mark.parametrize("script", SCRIPTS)
def test_master_addr_resolved_outside_container(tmp_path, script):
    """MASTER_ADDR reaches the rank even though scontrol is absent inside the container."""
    result = _run_launcher(tmp_path, script, r"node-a\nnode-b\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "master_addr").read_text() == "node-a"


@pytest.mark.parametrize("script", SCRIPTS)
def test_empty_master_addr_fails_before_srun(tmp_path, script):
    """The launcher exits before srun when MASTER_ADDR cannot be resolved."""
    result = _run_launcher(tmp_path, script, "")

    assert result.returncode != 0
    assert not (tmp_path / "srun_called").exists()
    assert "MASTER_ADDR" in result.stdout + result.stderr
