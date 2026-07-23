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


def test_gpt_oss_pretrain_reports_any_failed_config(tmp_path):
    source = REPO_ROOT / "examples/models/gpt_oss/slurm_pretrain.sh"
    configured_script = source.read_text().replace('CONTAINER_IMAGE=""', 'CONTAINER_IMAGE="test.sqsh"', 1)
    launcher = tmp_path / source.name
    launcher.write_text(configured_script)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        """#!/bin/bash
count=$(cat "$SRUN_COUNT_FILE" 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > "$SRUN_COUNT_FILE"
if [ "$count" -eq 1 ]; then
    exit 17
fi
exit 0
"""
    )
    fake_srun.chmod(0o755)

    count_file = tmp_path / "srun-count"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SRUN_COUNT_FILE": str(count_file),
        }
    )

    result = subprocess.run(["bash", str(launcher)], cwd=tmp_path, capture_output=True, env=env, text=True)

    assert count_file.read_text().strip() == "3"
    assert result.returncode == 17
    assert "failed with exit code 17" in result.stdout
    assert "Job completed" not in result.stdout
