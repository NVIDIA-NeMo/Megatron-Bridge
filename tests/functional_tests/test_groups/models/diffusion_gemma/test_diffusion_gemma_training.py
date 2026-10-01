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

"""Functional tests for DiffusionGemma parity and checkpoint resume."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


pytest.importorskip("transformers.models.gemma4")

REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLES_DIR = REPO_ROOT / "examples" / "models" / "diffusion_gemma"


def _env() -> dict[str, str]:
    python_path = os.pathsep.join(filter(None, [str(EXAMPLES_DIR), os.environ.get("PYTHONPATH")]))
    return {
        **os.environ,
        "PYTHONPATH": python_path,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
    }


def _torchrun(nproc: int, script: str, *args: str) -> None:
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        str(EXAMPLES_DIR / script),
        *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=_env())
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout[-20000:]}")
        print(f"STDERR: {result.stderr[-20000:]}")
    assert result.returncode == 0, f"{script} {' '.join(args)} failed with return code {result.returncode}"


class TestDiffusionGemmaParity:
    """Tiny Hugging Face ↔ Megatron loss and gradient parity."""

    @pytest.mark.run_only_on("GPU")
    def test_combined_objective_parity_and_overfit(self, tmp_path):
        """Check decoder + clean-encoder parity, long prompts, and a short overfit."""
        report_path = tmp_path / "parity.json"
        _torchrun(
            1,
            "toy_training_parity.py",
            "--variant",
            "base-sft",
            "--encoder-loss-weight",
            "1",
            "--output",
            str(report_path),
        )
        report = json.loads(report_path.read_text())
        assert report["passed"]
        assert "text_long_prompt" in report["cases"]

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("tp,ep", [(2, 1), (1, 2)], ids=["TP2", "EP2"])
    def test_model_parallel_parity(self, tmp_path, tp, ep):
        """Check TP/EP gradient reduction against the unsharded reference."""
        report_path = tmp_path / f"parity_tp{tp}_ep{ep}.json"
        _torchrun(
            2,
            "toy_training_parity.py",
            "--tp",
            str(tp),
            "--ep",
            str(ep),
            "--overfit-steps",
            "2",
            "--output",
            str(report_path),
        )
        report = json.loads(report_path.read_text())
        assert report["passed"]
        assert (report["tp"], report["ep"]) == (tp, ep)


class TestDiffusionGemmaTrainingResume:
    """Bridge ``pretrain()`` checkpoint/resume determinism."""

    @pytest.mark.run_only_on("GPU")
    def test_distributed_optimizer_resume_matches_uninterrupted_run(self, tmp_path):
        """Save at step three, resume in fresh processes, and compare logged steps."""
        common = ("--distributed-optimizer", "--eval-interval", "2", "--eval-iters", "1")
        launches = (
            ("reference", tmp_path / "reference", ()),
            ("prefix", tmp_path / "resumed", ("--stop-after", "3")),
            ("resume", tmp_path / "resumed", ("--resume",)),
        )
        for name, work_dir, extra in launches:
            _torchrun(
                2,
                "training_loop_smoke.py",
                "--work-dir",
                str(work_dir),
                "--report",
                str(tmp_path / f"{name}.json"),
                *common,
                *extra,
            )

        reference = json.loads((tmp_path / "reference.json").read_text())
        prefix = json.loads((tmp_path / "prefix.json").read_text())
        resumed = json.loads((tmp_path / "resume.json").read_text())
        assert reference["passed"] and resumed["passed"]
        assert (prefix["end_step"], resumed["start_step"], resumed["end_step"]) == (3, 3, 6)
        assert prefix["steps"] + resumed["steps"] == reference["steps"]
        assert resumed["consumed_train_samples"] == reference["consumed_train_samples"]
