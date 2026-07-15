import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
GEMMA4_VL_EXAMPLES = REPO_ROOT / "examples" / "models" / "gemma" / "gemma4_vl"


@pytest.mark.unit
@pytest.mark.parametrize("script_name", ["slurm_sft.sh", "slurm_peft.sh"])
def test_gemma4_vl_slurm_wrappers_propagate_training_failure(tmp_path: Path, script_name: str) -> None:
    """The batch script must report and return a failed training step."""
    scontrol = tmp_path / "scontrol"
    scontrol.write_text("#!/bin/bash\necho localhost\n")
    scontrol.chmod(0o755)

    srun = tmp_path / "srun"
    srun.write_text("#!/bin/bash\nexit 42\n")
    srun.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CONTAINER_IMAGE": "fake.sqsh",
            "PATH": f"{tmp_path}:{env['PATH']}",
            "SLURM_JOB_ID": "1234",
            "SLURM_JOB_NODELIST": "localhost",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "8",
        }
    )

    result = subprocess.run(
        ["bash", str(GEMMA4_VL_EXAMPLES / script_name)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert "Training job finished. EXIT=42" in result.stdout
    assert result.returncode == 42
