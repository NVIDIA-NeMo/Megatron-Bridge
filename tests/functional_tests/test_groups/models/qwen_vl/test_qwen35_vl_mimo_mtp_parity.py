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
"""Functional test: MegatronMIMO computes Qwen3.5-VL MTP exactly like the standard path.

Runs examples/megatron_mimo/qwen35_vl/verify_mtp_parity.py on a toy checkpoint with MTP weights:
the standard model on one GPU, MegatronMIMO on two, then the comparison. Requires 2 GPUs.

Run:
    uv run python -m pytest tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_vl_mimo_mtp_parity.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _TRANSFORMERS_HAS_QWEN3_5
from tests.functional_tests.test_groups.megatron_mimo.qwen35_vl_toy import save_hf_toy_checkpoint_with_mtp


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have Qwen3.5 support")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCRIPT = _REPO_ROOT / "examples" / "megatron_mimo" / "qwen35_vl" / "verify_mtp_parity.py"


def _run(cmd: list[str], *, visible_gpus: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_gpus},
    )


@pytest.mark.run_only_on("GPU")
def test_qwen35_vl_mimo_mtp_matches_standard_path(tmp_path):
    """Weights, LM and MTP losses, and every gradient match between the two paths."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2 GPUs")

    hf_path = tmp_path / "qwen35_vl_mtp_toy"
    save_hf_toy_checkpoint_with_mtp(hf_path)
    output_dir = tmp_path / "parity"

    for step, nproc, visible_gpus in (("standard", 1, "0"), ("mimo", 2, "0,1")):
        result = _run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                "--nnodes=1",
                "-m",
                "coverage",
                "run",
                f"--data-file={_REPO_ROOT / '.coverage'}",
                f"--source={_REPO_ROOT}",
                "--parallel-mode",
                str(_SCRIPT),
                step,
                "--hf-model",
                str(hf_path),
                "--output-dir",
                str(output_dir),
            ],
            visible_gpus=visible_gpus,
        )
        assert result.returncode == 0, f"{step} step failed:\n{result.stdout[-5000:]}\n{result.stderr[-5000:]}"

    result = _run([sys.executable, str(_SCRIPT), "compare", "--output-dir", str(output_dir)], visible_gpus="")
    report = result.stdout + result.stderr
    assert result.returncode == 0, f"MegatronMIMO and the standard path differ:\n{report[-5000:]}"
    assert "PASS: MegatronMIMO matches the standard path" in report
