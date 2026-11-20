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

"""Qwen4-Exp (Qwen3.8-Flash-Next) HF <-> Megatron conversion and forward-parity tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.functional_tests.test_groups.models.qwen.qwen4_exp_parity import build_toy_checkpoint


class TestQwen4ExpConversion:
    """Toy Qwen4-Exp model: HF -> Megatron logits parity (bshd + THD) and Megatron -> HF export."""

    @pytest.fixture(scope="class")
    def qwen4_exp_toy_model_path(self, tmp_path_factory):
        model_dir = tmp_path_factory.mktemp("qwen4_exp_toy_model") / "qwen4_exp_toy"
        build_toy_checkpoint(str(model_dir))
        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("tp,force_sparse", [(1, False), (2, False), (2, True)])
    def test_qwen4_exp_parity(self, qwen4_exp_toy_model_path, tmp_path, tp, force_sparse):
        script = Path(__file__).with_name("qwen4_exp_parity.py")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={tp}",
            "--nnodes=1",
            str(script),
            "--hf-model-path",
            qwen4_exp_toy_model_path,
            "--tp",
            str(tp),
            "--export-dir",
            str(tmp_path / f"export_tp{tp}"),
        ]
        if force_sparse:
            cmd.append("--force-sparse")
        env = dict(os.environ)
        # NVLink SHARP multicast is not available in every container / fabric configuration.
        env.setdefault("NCCL_NVLS_ENABLE", "0")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0, (
            f"Qwen4-Exp parity (tp={tp}, force_sparse={force_sparse}) failed\n"
            f"STDOUT: {result.stdout[-4000:]}\nSTDERR: {result.stderr[-4000:]}"
        )
        assert "PARITY OK" in result.stdout
