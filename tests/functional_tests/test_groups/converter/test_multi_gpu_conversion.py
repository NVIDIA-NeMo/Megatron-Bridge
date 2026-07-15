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


@pytest.fixture(scope="class")
def llama_megatron_checkpoint(tmp_path_factory):
    """Create one Megatron checkpoint for the round-trip load-path cases."""
    repo_root = Path(__file__).resolve().parents[4]
    checkpoint_path = tmp_path_factory.mktemp("roundtrip") / "megatron_checkpoint"
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
            "meta-llama/Llama-3.2-1B",
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
        f"Round-trip fixture import failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return checkpoint_path


class TestMultiGPUConversion:
    """
    Test multi-GPU conversion from HuggingFace models with different parallelism configurations.
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name,use_load_path",
        [
            (2, 1, "TP", True),
            (1, 2, "PP", False),
        ],
    )
    def test_conversion_parallelism(
        self,
        tmp_path,
        tp,
        pp,
        test_name,
        use_load_path,
        llama_megatron_checkpoint,
    ):
        """
        Test model conversion with different parallelism configurations.

        Args:
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory
        test_output_dir = tmp_path / test_name
        test_output_dir.mkdir(exist_ok=True)

        repo_root = Path(__file__).resolve().parents[4]
        # Run hf_megatron_roundtrip_multi_gpu.py through the public launcher.
        cmd = [
            str(repo_root / "scripts/conversion/convert.sh"),
            "roundtrip",
            "--executor",
            "local",
            "--device",
            "gpu",
            "--gpus-per-node",
            "2",
            "--hf-model-id",
            "meta-llama/Llama-3.2-1B",
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]
        if use_load_path:
            cmd.extend(["--megatron-load-path", str(llama_megatron_checkpoint)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            converted_model_dir = test_output_dir / "Llama-3.2-1B"
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in converted model at {converted_model_dir}"
            )

            print(f"SUCCESS: {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during {test_name} conversion test: {e}")
            raise
