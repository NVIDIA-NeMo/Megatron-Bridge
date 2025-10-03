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


class TestCheckpointConversion:
    """
    Test checkpoint conversion between HuggingFace and Megatron formats.
    """

    @pytest.mark.run_only_on("GPU")
    def test_import_hf_to_megatron(self, tmp_path):
        """
        Test importing a HuggingFace model to Megatron checkpoint format.

        Args:
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary output directory for Megatron checkpoint
        megatron_output_dir = tmp_path / "megatron_checkpoint"
        megatron_output_dir.mkdir(exist_ok=True)
        # Run convert_checkpoints.py import command
        cmd = [
            "python",
            "-m",
            "coverage",
            "run",
            "--data-file=/workspace/.coverage",
            "--source=/workspace/",
            "--parallel-mode",
            "examples/conversion/convert_checkpoints.py",
            "import",
            "--hf-model",
            "meta-llama/Llama-3.2-1B",
            "--megatron-path",
            str(megatron_output_dir),
            "--torch-dtype",
            "bfloat16",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
            )

            # Check that the import completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Import failed with return code {result.returncode}"

            # Verify that the import succeeded based on output messages
            assert "Successfully imported model to:" in result.stdout, (
                f"Import success message not found in output. Output: {result.stdout}"
            )

            # Verify that the checkpoint directory structure was created
            assert megatron_output_dir.exists(), f"Megatron checkpoint directory not found at {megatron_output_dir}"

            # Check for expected checkpoint files/directories
            checkpoint_contents = list(megatron_output_dir.iterdir())
            assert len(checkpoint_contents) > 0, f"Megatron checkpoint directory is empty: {megatron_output_dir}"

            print("SUCCESS: HF to Megatron import test completed successfully")
            print(f"Megatron checkpoint saved at: {megatron_output_dir}")
            print(f"Checkpoint contents: {[item.name for item in checkpoint_contents]}")

        except Exception as e:
            print(f"Error during HF to Megatron import test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    def test_export_megatron_to_hf(self, tmp_path):
        """
        Test exporting a Megatron checkpoint to HuggingFace format.

        This test first imports a model to create a Megatron checkpoint, then exports it back.

        Args:
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary directories
        megatron_checkpoint_dir = tmp_path / "megatron_checkpoint"
        hf_export_dir = tmp_path / "hf_export"
        megatron_checkpoint_dir.mkdir(exist_ok=True)
        hf_export_dir.mkdir(exist_ok=True)

        try:
            # First, import a HF model to create a Megatron checkpoint
            import_cmd = [
                "python",
                "-m",
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "examples/conversion/convert_checkpoints.py",
                "import",
                "--hf-model",
                "meta-llama/Llama-3.2-1B",
                "--megatron-path",
                str(megatron_checkpoint_dir),
                "--torch-dtype",
                "bfloat16",
            ]

            import_result = subprocess.run(
                import_cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
            )

            if import_result.returncode != 0:
                print(f"Import STDOUT: {import_result.stdout}")
                print(f"Import STDERR: {import_result.stderr}")
                assert False, f"Import step failed with return code {import_result.returncode}"

            # Now export the Megatron checkpoint back to HF format
            export_cmd = [
                "python",
                "-m",
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "examples/conversion/convert_checkpoints.py",
                "export",
                "--hf-model",
                "meta-llama/Llama-3.2-1B",
                "--megatron-path",
                str(megatron_checkpoint_dir),
                "--hf-path",
                str(hf_export_dir),
                "--no-progress",
            ]

            export_result = subprocess.run(
                export_cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
            )

            # Check that the export completed successfully
            if export_result.returncode != 0:
                print(f"Export STDOUT: {export_result.stdout}")
                print(f"Export STDERR: {export_result.stderr}")
                assert False, f"Export failed with return code {export_result.returncode}"

            # Verify that the export succeeded based on output messages
            assert "Successfully exported model to:" in export_result.stdout, (
                f"Export success message not found in output. Output: {export_result.stdout}"
            )

            # Verify that the HF export directory has expected files
            assert hf_export_dir.exists(), f"HF export directory not found at {hf_export_dir}"

            # Check for essential HuggingFace model files
            config_file = hf_export_dir / "config.json"
            assert config_file.exists(), f"config.json not found in exported model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = hf_export_dir / "model.safetensors"
            weights_file_pytorch = hf_export_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in exported model at {hf_export_dir}"
            )

            print("SUCCESS: Megatron to HF export test completed successfully")
            print(f"HF model exported at: {hf_export_dir}")
            print(f"Export contents: {[item.name for item in hf_export_dir.iterdir()]}")

        except Exception as e:
            print(f"Error during Megatron to HF export test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "torch_dtype,device_map,test_name",
        [
            ("float16", "auto", "float16_auto"),
            ("bfloat16", None, "bfloat16_cpu"),
        ],
    )
    def test_import_with_different_settings(self, tmp_path, torch_dtype, device_map, test_name):
        """
        Test importing with different torch_dtype and device_map settings.

        Args:
            tmp_path: Pytest temporary path fixture
            torch_dtype: Model precision to test
            device_map: Device placement strategy to test
            test_name: Name of the test for identification
        """
        # Create temporary output directory
        megatron_output_dir = tmp_path / f"megatron_checkpoint_{test_name}"
        megatron_output_dir.mkdir(exist_ok=True)

        try:
            # Build command with different settings
            cmd = [
                "python",
                "-m",
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "examples/conversion/convert_checkpoints.py",
                "import",
                "--hf-model",
                "meta-llama/Llama-3.2-1B",
                "--megatron-path",
                str(megatron_output_dir),
                "--torch-dtype",
                torch_dtype,
            ]

            # Add device_map if specified
            if device_map:
                cmd.extend(["--device-map", device_map])

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
            )

            # Check that the import completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Import with {test_name} settings failed with return code {result.returncode}"

            # Verify expected settings were used
            assert f"Using torch_dtype: {torch_dtype}" in result.stdout, (
                f"torch_dtype setting not found in {test_name} output. Output: {result.stdout}"
            )

            if device_map:
                assert f"Using device_map: {device_map}" in result.stdout, (
                    f"device_map setting not found in {test_name} output. Output: {result.stdout}"
                )

            # Verify successful completion
            assert "Successfully imported model to:" in result.stdout, (
                f"Import success message not found in {test_name} output. Output: {result.stdout}"
            )

            print(f"SUCCESS: {test_name} settings test completed successfully")

        except Exception as e:
            print(f"Error during {test_name} settings test: {e}")
            raise
