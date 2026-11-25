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
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples/models/stepfun/step35/slurm_pretrain.sh"


def _sbatch_value(text, option):
    return int(re.search(rf"^#SBATCH --{option}=(\d+)$", text, re.MULTILINE).group(1))


def _launch(tmp_path):
    """Run the launcher with a fake srun and return (header nodes, gpus per node, srun argv text)."""
    text = SCRIPT.read_text()
    nodes = _sbatch_value(text, "nodes")
    gpus_per_node = _sbatch_value(text, "gpus-per-node")
    launcher = tmp_path / SCRIPT.name
    launcher.write_text(text)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SRUN_ARGS_FILE"\nexit 0\n')
    fake_srun.chmod(0o755)
    fake_scontrol = fake_bin / "scontrol"
    fake_scontrol.write_text('#!/bin/bash\nprintf "%s\\n" "node-0"\n')
    fake_scontrol.chmod(0o755)

    args_file = tmp_path / "srun-args"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SRUN_ARGS_FILE": str(args_file),
            "CONTAINER_IMAGE": "test.sqsh",
            "WORKSPACE": str(tmp_path),
            "SLURM_JOB_ID": "1",
            "SLURM_JOB_NODELIST": "node-0",
            "SLURM_JOB_NUM_NODES": str(nodes),
            "SLURM_GPUS_PER_NODE": str(gpus_per_node),
        }
    )
    result = subprocess.run(["bash", str(launcher)], cwd=tmp_path, capture_output=True, env=env, text=True)
    assert result.returncode == 0, result.stderr
    return nodes, gpus_per_node, args_file.read_text()


def _override(argv, key):
    return int(re.search(rf"\b{re.escape(key)}=(\d+)", argv).group(1))


def test_step35_pretrain_layout_fits_sbatch_allocation(tmp_path):
    """The parallel layout in the overrides can be placed on the header's node allocation."""
    nodes, gpus_per_node, argv = _launch(tmp_path)
    world_size = nodes * gpus_per_node
    tp = _override(argv, "model.tensor_model_parallel_size")
    pp = _override(argv, "model.pipeline_model_parallel_size")
    cp = _override(argv, "model.context_parallel_size")
    ep = _override(argv, "model.expert_model_parallel_size")
    etp = _override(argv, "model.expert_tensor_parallel_size")

    assert world_size % (tp * pp * cp) == 0
    assert world_size % (pp * ep * etp) == 0
    assert f"--nodes={nodes}" in argv.splitlines()


def test_step35_pretrain_offloads_optimizer_with_precision_aware_optimizer(tmp_path):
    """Optimizer CPU offload is enabled together with the precision-aware optimizer it requires."""
    _, _, argv = _launch(tmp_path)

    assert "optimizer.optimizer_cpu_offload=true" in argv
    assert "optimizer.optimizer_offload_fraction=1.0" in argv
    assert "optimizer.use_precision_aware_optimizer=true" in argv
