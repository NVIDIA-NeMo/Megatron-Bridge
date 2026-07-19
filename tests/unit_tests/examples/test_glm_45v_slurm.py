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


REPO_ROOT = Path(__file__).resolve().parents[3]
GLM_45V_EXAMPLES = REPO_ROOT / "examples" / "models" / "glm" / "glm_45v"


@pytest.mark.unit
@pytest.mark.parametrize("script_name", ["slurm_sft.sh", "slurm_peft.sh"])
def test_glm_45v_slurm_wrappers_propagate_training_failure(tmp_path: Path, script_name: str) -> None:
    """The batch script must return a failed synchronous training step."""
    source_script = GLM_45V_EXAMPLES / script_name
    configured_script = tmp_path / script_name
    source = source_script.read_text()
    configured = source.replace('CONTAINER_IMAGE=""', 'CONTAINER_IMAGE="fake.sqsh"', 1)
    assert configured != source
    configured_script.write_text(configured)

    srun = tmp_path / "srun"
    srun.write_text("#!/bin/bash\nexit 42\n")
    srun.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "SLURM_GPUS_PER_NODE": "8",
            "SLURM_JOB_ID": "1234",
            "SLURM_JOB_NUM_NODES": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(configured_script)],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode == 42
