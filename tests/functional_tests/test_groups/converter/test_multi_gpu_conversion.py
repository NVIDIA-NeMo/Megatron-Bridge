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

import subprocess
from pathlib import Path

import pytest


_HF_MODEL = "meta-llama/Llama-3.2-1B"


def _assert_hf_weights_match(export_path: Path) -> None:
    """Compare an exported checkpoint with the original Hugging Face weights."""
    import torch

    from megatron.bridge import AutoBridge

    original_bridge = AutoBridge.from_hf_pretrained(_HF_MODEL, torch_dtype=torch.bfloat16)
    exported_bridge = AutoBridge.from_hf_pretrained(str(export_path), torch_dtype=torch.bfloat16)
    original_state = original_bridge.hf_pretrained.state
    exported_state = exported_bridge.hf_pretrained.state
    assert set(exported_state.keys()) == set(original_state.keys())
    for name in original_state:
        original = original_state[name]
        exported = exported_state[name]
        assert exported.dtype == original.dtype, f"Exported dtype mismatch for {name}"
        assert torch.equal(exported, original), f"Exported weight mismatch for {name}"


@pytest.fixture(scope="class")
def llama_megatron_checkpoint(tmp_path_factory):
    """Create a Megatron checkpoint for the distributed GPU export test."""
    repo_root = Path(__file__).resolve().parents[4]
    checkpoint_path = tmp_path_factory.mktemp("gpu-export") / "megatron_checkpoint"
    result = subprocess.run(
        [
            "python",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "scripts/conversion/run_conversion.py",
            "import",
            "--device",
            "cpu",
            "--hf-model",
            _HF_MODEL,
            "--megatron-path",
            str(checkpoint_path),
            "--torch-dtype",
            "bfloat16",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, (
        f"GPU export fixture import failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return checkpoint_path


class TestMultiGPUConversion:
    """
    Test multi-GPU conversion from HuggingFace models with different parallelism configurations.
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_conversion_parallelism(self, tp, pp, test_name):
        """
        Test model conversion with different parallelism configurations.

        Args:
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """
        repo_root = Path(__file__).resolve().parents[4]
        # Run the scripts/conversion roundtrip worker through the public launcher.
        cmd = [
            "bash",
            str(repo_root / "scripts/conversion/convert.sh"),
            "roundtrip",
            "--executor",
            "local",
            "--device",
            "gpu",
            "--gpus-per-node",
            "2",
            "--hf-model-id",
            _HF_MODEL,
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)

        assert result.returncode == 0, (
            f"{test_name} round-trip validation failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "GPU round-trip validation complete" in result.stdout

    @pytest.mark.run_only_on("GPU")
    def test_gpu_checkpoint_export(self, tmp_path, llama_megatron_checkpoint):
        """Test distributed checkpoint loading and Hugging Face saving on two GPUs."""
        repo_root = Path(__file__).resolve().parents[4]
        hf_export_path = tmp_path / "hf_export"
        cmd = [
            "bash",
            str(repo_root / "scripts/conversion/convert.sh"),
            "export",
            "--executor",
            "local",
            "--device",
            "gpu",
            "--gpus-per-node",
            "2",
            "--hf-model",
            _HF_MODEL,
            "--megatron-path",
            str(llama_megatron_checkpoint),
            "--hf-path",
            str(hf_export_path),
            "--tp",
            "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)

        assert result.returncode == 0, (
            f"Distributed GPU export failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "GPU export complete:" in result.stdout
        assert (hf_export_path / "config.json").exists()
        assert list(hf_export_path.glob("*.safetensors"))
        _assert_hf_weights_match(hf_export_path)
