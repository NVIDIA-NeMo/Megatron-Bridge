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

"""Functional smoke tests for WAN checkpoint conversion (HF <-> Megatron)."""

import shutil
import subprocess
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download


WAN_HF_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

BASE_DIR = "/workspace/test_ckpts/wan_conversion"
MEGATRON_CKPT_DIR = f"{BASE_DIR}/megatron"
HF_EXPORT_DIR = f"{BASE_DIR}/hf_export"

CONVERT_SCRIPT = "examples/diffusion/recipes/wan/conversion/convert_checkpoints.py"
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


def _get_hf_ckpt_dir() -> str:
    """Download the WAN HF checkpoint if not already cached and return the local path."""
    return snapshot_download(WAN_HF_MODEL_ID)


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    return result


class TestWanCkptConversion:
    """Functional tests for WAN checkpoint conversion (HF <-> Megatron)."""

    @pytest.mark.run_only_on("GPU")
    def test_hf_to_megatron(self):
        """Import a WAN HF checkpoint into Megatron format."""
        hf_ckpt_dir = _get_hf_ckpt_dir()

        Path(MEGATRON_CKPT_DIR).mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            CONVERT_SCRIPT,
            "import",
            "--hf-model",
            hf_ckpt_dir,
            "--megatron-path",
            MEGATRON_CKPT_DIR,
        ]

        result = _run_cmd(cmd)
        assert result.returncode == 0, f"HF->Megatron conversion failed (rc={result.returncode})"
        assert "Successfully imported model to:" in result.stdout, (
            f"Success message not found in output:\n{result.stdout}"
        )

        # Verify checkpoint directory contains at least one iter_* directory or files
        ckpt_path = Path(MEGATRON_CKPT_DIR)
        assert ckpt_path.exists(), f"Megatron checkpoint directory not created: {ckpt_path}"
        contents = list(ckpt_path.iterdir())
        assert len(contents) > 0, f"Megatron checkpoint directory is empty: {ckpt_path}"

        print(f"Megatron checkpoint saved at: {ckpt_path}")
        print(f"Contents: {[item.name for item in contents]}")

    @pytest.mark.run_only_on("GPU")
    def test_megatron_to_hf(self):
        """Export the previously imported Megatron checkpoint back to HF format."""
        hf_ckpt_dir = _get_hf_ckpt_dir()

        assert Path(MEGATRON_CKPT_DIR).exists(), (
            f"Megatron checkpoint not found at {MEGATRON_CKPT_DIR}. "
            "Run test_hf_to_megatron first."
        )

        # Locate the iter_* directory produced by the import step
        ckpt_path = Path(MEGATRON_CKPT_DIR)
        iter_dirs = sorted(
            [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("iter_")],
            key=lambda d: int(d.name.replace("iter_", "")),
        )
        assert len(iter_dirs) > 0, (
            f"No iter_* directory found in {ckpt_path}. "
            "Ensure test_hf_to_megatron completed successfully."
        )
        megatron_iter_path = iter_dirs[-1]

        Path(HF_EXPORT_DIR).mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            CONVERT_SCRIPT,
            "export",
            "--hf-model",
            hf_ckpt_dir,
            "--megatron-path",
            str(megatron_iter_path),
            "--hf-path",
            HF_EXPORT_DIR,
            "--no-progress",
        ]

        result = _run_cmd(cmd)
        assert result.returncode == 0, f"Megatron->HF conversion failed (rc={result.returncode})"
        assert "Successfully exported model to:" in result.stdout, (
            f"Success message not found in output:\n{result.stdout}"
        )

        # Verify exported transformer/ directory and config.json exist
        export_path = Path(HF_EXPORT_DIR)
        assert export_path.exists(), f"HF export directory not created: {export_path}"

        transformer_dir = export_path / "transformer"
        assert transformer_dir.exists(), f"transformer/ subdirectory not found in {export_path}"

        config_file = transformer_dir / "config.json"
        assert config_file.exists(), f"config.json not found in {transformer_dir}"

        # At least one safetensors shard should be present
        safetensors_files = list(transformer_dir.glob("*.safetensors"))
        assert len(safetensors_files) > 0, f"No safetensors files found in {transformer_dir}"

        print(f"HF export saved at: {export_path}")
        print(f"Transformer contents: {[item.name for item in transformer_dir.iterdir()]}")

    def test_remove_artifacts(self):
        """Remove artifacts created by the conversion tests."""
        shutil.rmtree(BASE_DIR, ignore_errors=True)
        assert not Path(BASE_DIR).exists(), f"Failed to remove {BASE_DIR}"
