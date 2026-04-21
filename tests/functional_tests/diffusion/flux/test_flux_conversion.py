# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Functional tests for FLUX HF <-> Megatron checkpoint conversion.

Uses a tiny randomly initialized diffusers FluxTransformer2DModel saved under a local
``transformer/`` layout (same as FLUX.1-dev) and drives
``examples/diffusion/models/flux/conversion/convert_checkpoints.py``.

"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch


diffusers = pytest.importorskip("diffusers")
FluxTransformer2DModel = diffusers.FluxTransformer2DModel

# Repo root: tests/functional_tests/diffusion/flux -> five parents
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
CONVERT_SCRIPT = "examples/diffusion/models/flux/conversion/convert_checkpoints.py"


def _build_toy_flux_hf_root(base: Path) -> Path:
    """
    Create ``base`` as a FLUX-style HF root with weights under ``base/transformer/``.

    Dimensions are chosen so ``sum(axes_dims_rope) == attention_head_dim`` (128), matching
    real FLUX layouts, while keeping layers minimal for speed.
    """
    hf_root = base / "flux_toy_hf"
    transformer_dir = hf_root / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        num_layers=1,
        num_single_layers=2,
        attention_head_dim=128,
        num_attention_heads=8,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=False,
        axes_dims_rope=(16, 56, 56),
    )
    model = model.to(dtype=torch.bfloat16)
    model.save_pretrained(transformer_dir, safe_serialization=True)
    return hf_root


class TestFluxCheckpointConversion:
    """HF FLUX transformer (diffusers) <-> Megatron via convert_checkpoints.py."""

    @pytest.fixture(scope="class")
    def flux_toy_hf_root(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        base = tmp_path_factory.mktemp("flux_conv_toy")
        return _build_toy_flux_hf_root(Path(base))

    @pytest.fixture(scope="class")
    def flux_megatron_ckpt_dir(self, flux_toy_hf_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """
        Run ``convert_checkpoints.py import`` once for the class.

        Export test no longer repeats this (~10–30s+ per run depending on GPU).
        """
        megatron_out = Path(tmp_path_factory.mktemp("flux_megatron_shared"))
        megatron_out.mkdir(parents=True, exist_ok=True)
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
            str(flux_toy_hf_root),
            "--megatron-path",
            str(megatron_out),
            "--torch-dtype",
            "bfloat16",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            pytest.fail(f"FLUX import (class fixture) failed with exit code {result.returncode}")
        assert "Successfully imported model to:" in result.stdout
        assert any(megatron_out.iterdir()), f"Megatron output empty: {megatron_out}"
        return megatron_out

    def test_toy_flux_layout(self, flux_toy_hf_root: Path) -> None:
        tdir = flux_toy_hf_root / "transformer"
        assert tdir.is_dir()
        assert (tdir / "config.json").is_file()
        assert (tdir / "diffusion_pytorch_model.safetensors").is_file() or (
            tdir / "diffusion_pytorch_model.bin"
        ).is_file()

    def test_import_hf_to_megatron(self, flux_megatron_ckpt_dir: Path) -> None:
        """Asserts the shared class import produced a non-empty Megatron checkpoint."""
        assert flux_megatron_ckpt_dir.is_dir()
        assert any(flux_megatron_ckpt_dir.iterdir())

    def test_export_megatron_to_hf_roundtrip(
        self, flux_toy_hf_root: Path, flux_megatron_ckpt_dir: Path, tmp_path: Path
    ) -> None:
        hf_export = tmp_path / "hf_export"
        hf_export.mkdir(parents=True, exist_ok=True)

        export_cmd = [
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
            str(flux_toy_hf_root),
            "--megatron-path",
            str(flux_megatron_ckpt_dir),
            "--hf-path",
            str(hf_export),
            "--no-progress",
        ]
        er = subprocess.run(export_cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if er.returncode != 0:
            print("EXPORT STDOUT:", er.stdout)
            print("EXPORT STDERR:", er.stderr)
            pytest.fail(f"FLUX export failed with exit code {er.returncode}")

        assert "Successfully exported model to:" in er.stdout
        transformer_out = hf_export / "transformer"
        assert transformer_out.is_dir(), f"Missing {transformer_out}"
        assert (transformer_out / "config.json").is_file()
        st = transformer_out / "diffusion_pytorch_model.safetensors"
        pt = transformer_out / "diffusion_pytorch_model.bin"
        assert st.is_file() or pt.is_file(), f"No weights under {transformer_out}"
