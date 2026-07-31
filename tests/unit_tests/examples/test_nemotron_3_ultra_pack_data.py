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
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PACK_DATA_SCRIPT = REPO_ROOT / "examples/models/nemotron/nemotron_3/ultra/pack_data_job.sh"


def test_nemotron_3_ultra_pack_data_uses_recipe_defaults(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        """#!/bin/bash
while [[ "$1" == --* ]]; do
    shift
done
exec "$@"
"""
    )
    fake_srun.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/bash
printf '%s\\n' "$@" > "$UV_ARGS_FILE"
"""
    )
    fake_uv.chmod(0o755)

    uv_args_file = tmp_path / "uv-args"
    env = os.environ.copy()
    env.update(
        {
            "CONTAINER_IMAGE": "test.sqsh",
            "PATH": f"{fake_bin}:{env['PATH']}",
            "UV_ARGS_FILE": str(uv_args_file),
            "WORKDIR": str(tmp_path),
        }
    )

    result = subprocess.run(
        ["bash", str(PACK_DATA_SCRIPT)],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PACK_DATA_DONE"
    assert uv_args_file.read_text().splitlines() == [
        "run",
        "--no-sync",
        "python",
        "scripts/training/prepare_gpt_sft_packed_data.py",
        "--recipe",
        "nemotron_3_ultra_sft_openmathinstruct2_packed_config",
    ]
