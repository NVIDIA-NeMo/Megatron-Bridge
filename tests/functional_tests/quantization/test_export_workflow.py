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

import json
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors import safe_open


class TestExportWorkflow:
    """
    Test complete export workflow: quantize HuggingFace models to Megatron format,
    then export back to HuggingFace format with different parallelism settings,
    and validate the exported state dict matches expectations.
    """

    def _load_checkpoint(self, load_path: Path, prefix: str = None):
        """
        Load checkpoint from various formats (pytorch_model.bin, safetensors).

        Args:
            load_path: Path to checkpoint directory
            prefix: Optional prefix to filter keys

        Returns:
            dict: State dictionary
        """
        pytorch_model_path = load_path / "pytorch_model.bin"
        safetensors_model_path = load_path / "model.safetensors"
        safetensors_index_path = load_path / "model.safetensors.index.json"

        state_dict = {}

        if pytorch_model_path.exists():
            state_dict = torch.load(pytorch_model_path, weights_only=True, map_location="cpu")
        elif safetensors_model_path.exists():
            with safe_open(safetensors_model_path, framework="pt") as f:
                for k in f.keys():
                    if prefix is None or prefix in k:
                        state_dict[k] = f.get_tensor(k)
        elif safetensors_index_path.exists():
            with open(safetensors_index_path, "r") as f:
                weight_map = json.load(f)["weight_map"]
            for k, filename in weight_map.items():
                if prefix is not None and prefix not in k:
                    continue
                with safe_open(load_path / filename, framework="pt") as f:
                    state_dict[k] = f.get_tensor(k)
        else:
            raise ValueError(f"No supported checkpoint format found at {load_path}")

        return state_dict

    def _check_hf_export_files(self, export_dir: Path):
        """
        Validate that essential HuggingFace model files exist in the export directory.
        This mimics the validation approach from test_checkpoint_conversion.py

        Args:
            export_dir: Path to HF export directory

        Raises:
            AssertionError: If required files are missing
        """
        # Check for config.json
        config_file = export_dir / "config.json"
        assert config_file.exists(), f"config.json not found in exported model at {config_file}"

        # Check for model weights file (could be either safetensors or pytorch_model.bin)
        weights_file_safetensors = export_dir / "model.safetensors"
        weights_file_pytorch = export_dir / "pytorch_model.bin"
        safetensors_index = export_dir / "model.safetensors.index.json"

        assert weights_file_safetensors.exists() or weights_file_pytorch.exists() or safetensors_index.exists(), (
            f"Model weights file not found in exported model at {export_dir}"
        )

        # Check for tokenizer files (optional but good to have)
        tokenizer_config = export_dir / "tokenizer_config.json"
        if tokenizer_config.exists():
            print("  ✓ Tokenizer config found")

    def _check_state_dict_structure(self, state_dict):
        """
        Validate that the state dict has expected structure and non-empty tensors.

        Args:
            state_dict: State dictionary to validate

        Returns:
            tuple: (num_tensors, total_params)
        """
        num_tensors = 0
        total_params = 0

        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                num_tensors += 1
                total_params += v.numel()
                # Ensure tensors are not all zeros
                assert v.numel() > 0, f"Tensor {k} is empty"

        return num_tensors, total_params

    def _run_quantization(self, base_dir, hf_model_id, quant_cfg="fp8", tp=1, pp=1):
        """
        Helper method to run quantization step.

        Args:
            base_dir: Base directory to save the quantized checkpoint
            hf_model_id: HuggingFace model ID to quantize
            quant_cfg: Quantization configuration to use
            tp: Tensor parallelism size
            pp: Pipeline parallelism size

        Returns:
            tuple: (subprocess.CompletedProcess, actual_output_path)
        """
        # Create descriptive checkpoint name including configuration
        checkpoint_name = f"llama32_quantized_{quant_cfg}_tp{tp}_pp{pp}"
        output_dir = base_dir / checkpoint_name
        output_dir.mkdir(exist_ok=True)

        # Calculate total number of processes needed
        total_procs = max(tp * pp, 1)

        import sys

        python_executable = sys.executable

        # Base command following the user's format
        cmd = [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={total_procs}",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/quantization/quantize.py",
            "--hf-model-id",
            hf_model_id,
            "--export-quant-cfg",
            quant_cfg,
            "--megatron-save-path",
            str(output_dir),
            "--disable-hf-datasets-file-lock",
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)
        return result, output_dir

    def _run_export(self, checkpoint_dir, export_dir, hf_model_id, tp=1, pp=1, ep=1, etp=1):
        """
        Helper method to run export step.

        Args:
            checkpoint_dir: Directory containing the quantized checkpoint
            export_dir: Directory to export HuggingFace model to
            hf_model_id: HuggingFace model ID for tokenizer and model structure
            tp: Tensor parallelism size for export
            pp: Pipeline parallelism size for export
            ep: Expert parallelism size for export
            etp: Expert tensor parallelism size for export

        Returns:
            subprocess.CompletedProcess: Result of export process
        """
        # Calculate total number of processes needed
        total_procs = max(tp * pp * ep, 1)

        import sys

        python_executable = sys.executable

        # Base command following the user's format
        cmd = [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={total_procs}",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/quantization/export.py",
            "--hf-model-id",
            hf_model_id,
            "--megatron-load-path",
            str(checkpoint_dir),
            "--export-dir",
            str(export_dir),
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])
        if ep > 1:
            cmd.extend(["--ep", str(ep)])
        if etp > 1:
            cmd.extend(["--etp", str(etp)])

        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)

    @pytest.mark.run_only_on("GPU")
    def test_export_single_gpu(self, tmp_path):
        """
        Test complete workflow on single GPU: quantize, then export to HuggingFace format.

        Args:
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary directories
        base_dir = tmp_path / "checkpoints_export_single"
        base_dir.mkdir(exist_ok=True)
        export_dir = tmp_path / "hf_export_single"
        export_dir.mkdir(exist_ok=True)

        # Use the model ID from the user's request
        hf_model_id = "meta-llama/Llama-3.2-1B"

        try:
            print("=== STEP 1: Quantizing model on single GPU ===")
            # Step 1: Quantize the model
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                base_dir, hf_model_id, quant_cfg="fp8", tp=1, pp=1
            )

            if quantize_result.returncode != 0:
                print(f"Quantization STDOUT: {quantize_result.stdout}")
                print(f"Quantization STDERR: {quantize_result.stderr}")
                assert False, f"Quantization step failed with return code {quantize_result.returncode}"

            # Verify quantization succeeded
            assert "Quantizing the model with fp8 configuration" in quantize_result.stdout, (
                f"Quantization start message not found. Output: {quantize_result.stdout}"
            )
            assert quantized_checkpoint_dir.exists(), (
                f"Quantized checkpoint directory not found at {quantized_checkpoint_dir}"
            )

            checkpoint_contents = list(quantized_checkpoint_dir.iterdir())
            assert len(checkpoint_contents) > 0, f"Quantized checkpoint directory is empty: {quantized_checkpoint_dir}"

            print("✓ Quantization completed successfully")
            print(f"  Checkpoint saved at: {quantized_checkpoint_dir}")

            print("=== STEP 2: Exporting quantized model to HuggingFace format ===")
            # Step 2: Export the quantized model to HuggingFace format
            export_result = self._run_export(quantized_checkpoint_dir, export_dir, hf_model_id, tp=1, pp=1)

            if export_result.returncode != 0:
                print(f"Export STDOUT: {export_result.stdout}")
                print(f"Export STDERR: {export_result.stderr}")
                assert False, f"Export step failed with return code {export_result.returncode}"

            # Verify export succeeded
            # Note: stdout may have line wrapping, so we normalize it by removing newlines within the output
            stdout_normalized = export_result.stdout.replace("\n", "")
            assert (
                "Loaded quantized model from:" in export_result.stdout
                and str(quantized_checkpoint_dir) in stdout_normalized
            ), f"Checkpoint loading message not found. Output: {export_result.stdout}"
            assert "Exporting to HuggingFace format" in export_result.stdout, (
                f"Export message not found. Output: {export_result.stdout}"
            )
            assert "Export completed successfully!" in export_result.stdout, (
                f"Export completion message not found. Output: {export_result.stdout}"
            )

            print("✓ Export completed successfully")

            print("=== STEP 3: Validating exported HF checkpoint ===")
            # Step 3a: Check for essential HF files (mimicking test_checkpoint_conversion.py)
            self._check_hf_export_files(export_dir)
            print("✓ Essential HF files validation passed")

            # Step 3b: Load and validate the exported state dict (deeper validation)
            exported_state_dict = self._load_checkpoint(export_dir)
            num_tensors, total_params = self._check_state_dict_structure(exported_state_dict)

            assert num_tensors > 0, "Exported state dict contains no tensors"
            assert total_params > 0, "Exported state dict contains no parameters"

            print("✓ State dict validation passed")
            print(f"  Number of tensors: {num_tensors}")
            print(f"  Total parameters: {total_params:,}")
            print("SUCCESS: Complete single GPU export workflow test passed")

        except Exception as e:
            print(f"Error during single GPU export workflow test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    def test_export_single_to_pp2(self, tmp_path):
        """
        Test export with different parallelism: quantize on single GPU, export with PP=2.

        Args:
            tmp_path: Pytest temporary path fixture
        """
        quant_tp, quant_pp = 1, 1
        export_tp, export_pp = 1, 2
        test_name = "Single_to_PP2"

        # Create temporary directories
        base_dir = tmp_path / f"checkpoints_export_{test_name.lower()}"
        base_dir.mkdir(exist_ok=True)
        export_dir = tmp_path / f"hf_export_{test_name.lower()}"
        export_dir.mkdir(exist_ok=True)

        # Use the model ID from the user's request
        hf_model_id = "meta-llama/Llama-3.2-1B"

        try:
            print(f"=== STEP 1: Quantizing model with TP={quant_tp}, PP={quant_pp} ===")
            # Step 1: Quantize the model with specified parallelism
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                base_dir, hf_model_id, quant_cfg="fp8", tp=quant_tp, pp=quant_pp
            )

            if quantize_result.returncode != 0:
                print(f"Quantization STDOUT: {quantize_result.stdout}")
                print(f"Quantization STDERR: {quantize_result.stderr}")
                assert False, f"Quantization step for {test_name} failed with return code {quantize_result.returncode}"

            # Verify quantization succeeded with correct parallelism
            assert "Quantizing the model with fp8 configuration" in quantize_result.stdout, (
                f"Quantization start message not found in {test_name}. Output: {quantize_result.stdout}"
            )
            assert f"Tensor parallel size: {quant_tp}" in quantize_result.stdout, (
                f"Quantization TP setting not found in {test_name}. Output: {quantize_result.stdout}"
            )
            assert f"Pipeline parallel size: {quant_pp}" in quantize_result.stdout, (
                f"Quantization PP setting not found in {test_name}. Output: {quantize_result.stdout}"
            )

            assert quantized_checkpoint_dir.exists(), (
                f"Quantized checkpoint directory not found at {quantized_checkpoint_dir}"
            )
            checkpoint_contents = list(quantized_checkpoint_dir.iterdir())
            assert len(checkpoint_contents) > 0, f"Quantized checkpoint directory is empty: {quantized_checkpoint_dir}"

            print(f"✓ Quantization completed with TP={quant_tp}, PP={quant_pp}")

            print(f"=== STEP 2: Exporting with TP={export_tp}, PP={export_pp} ===")
            # Step 2: Export with different parallelism configuration
            export_result = self._run_export(
                quantized_checkpoint_dir, export_dir, hf_model_id, tp=export_tp, pp=export_pp
            )

            if export_result.returncode != 0:
                print(f"Export STDOUT: {export_result.stdout}")
                print(f"Export STDERR: {export_result.stderr}")
                assert False, f"Export step for {test_name} failed with return code {export_result.returncode}"

            # Verify export succeeded with correct parallelism
            # Note: stdout may have line wrapping, so we normalize it by removing newlines within the output
            stdout_normalized = export_result.stdout.replace("\n", "")
            assert (
                "Loaded quantized model from:" in export_result.stdout
                and str(quantized_checkpoint_dir) in stdout_normalized
            ), f"Checkpoint loading message not found in {test_name}. Output: {export_result.stdout}"
            assert f"Tensor parallel size: {export_tp}" in export_result.stdout, (
                f"Export TP setting not found in {test_name}. Output: {export_result.stdout}"
            )
            assert f"Pipeline parallel size: {export_pp}" in export_result.stdout, (
                f"Export PP setting not found in {test_name}. Output: {export_result.stdout}"
            )
            assert "Exporting to HuggingFace format" in export_result.stdout, (
                f"Export message not found in {test_name}. Output: {export_result.stdout}"
            )
            assert "Export completed successfully!" in export_result.stdout, (
                f"Export completion message not found in {test_name}. Output: {export_result.stdout}"
            )

            print(f"✓ Export completed with TP={export_tp}, PP={export_pp}")

            print("=== STEP 3: Validating exported HF checkpoint ===")
            # Step 3a: Check for essential HF files (mimicking test_checkpoint_conversion.py)
            self._check_hf_export_files(export_dir)
            print(f"✓ Essential HF files validation passed for {test_name}")

            # Step 3b: Load and validate the exported state dict (deeper validation)
            exported_state_dict = self._load_checkpoint(export_dir)
            num_tensors, total_params = self._check_state_dict_structure(exported_state_dict)

            assert num_tensors > 0, f"Exported state dict contains no tensors for {test_name}"
            assert total_params > 0, f"Exported state dict contains no parameters for {test_name}"

            print(f"✓ State dict validation passed for {test_name}")
            print(f"  Number of tensors: {num_tensors}")
            print(f"  Total parameters: {total_params:,}")
            print(f"SUCCESS: {test_name} export workflow test passed")

        except Exception as e:
            print(f"Error during {test_name} export workflow test: {e}")
            raise
