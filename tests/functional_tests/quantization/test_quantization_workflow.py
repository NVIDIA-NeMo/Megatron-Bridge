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


class TestQuantizationWorkflow:
    """
    Test complete quantization workflow: quantize HuggingFace models to Megatron format,
    then test text generation from the quantized checkpoints.
    """

    def _run_quantization(self, base_dir, quant_cfg="fp8", tp=1, pp=1):
        """
        Helper method to run quantization step.

        Args:
            base_dir: Base directory to save the quantized checkpoint
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

        # Base command following the user's format
        cmd = [
            "torchrun",
            f"--nproc_per_node={total_procs}",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/quantization/quantize.py",
            "--hf-model-id",
            "meta-llama/Llama-3.2-1B",
            "--export-quant-cfg",
            quant_cfg,
            "--megatron-save-path",
            str(output_dir),
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)
        return result, output_dir

    def _run_generation(self, checkpoint_dir, tp=1, pp=1):
        """
        Helper method to run generation step.

        Args:
            checkpoint_dir: Directory containing the quantized checkpoint
            tp: Tensor parallelism size for generation
            pp: Pipeline parallelism size for generation

        Returns:
            subprocess.CompletedProcess: Result of generation process
        """
        # Calculate total number of processes needed
        total_procs = max(tp * pp, 1)

        # Base command following the user's format
        cmd = [
            "torchrun",
            f"--nproc_per_node={total_procs}",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/quantization/ptq_generate.py",
            "--hf-model-id",
            "meta-llama/Llama-3.2-1B",
            "--megatron-load-path",
            str(checkpoint_dir),
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])

        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)

    @pytest.mark.run_only_on("GPU")
    def test_quantization_and_generation_single_gpu(self, tmp_path):
        """
        Test complete workflow: quantize on single GPU, then generate from quantized checkpoint.

        Args:
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / "checkpoints_single_gpu"
        base_dir.mkdir(exist_ok=True)

        try:
            print("=== STEP 1: Quantizing model on single GPU ===")
            # Step 1: Quantize the model
            quantize_result, quantized_checkpoint_dir = self._run_quantization(base_dir, quant_cfg="fp8", tp=1, pp=1)

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
            print(f"  Checkpoint contents: {[item.name for item in checkpoint_contents]}")

            print("=== STEP 2: Testing generation from quantized checkpoint ===")
            # Step 2: Test generation from the quantized checkpoint
            generation_result = self._run_generation(quantized_checkpoint_dir, tp=1, pp=1)

            if generation_result.returncode != 0:
                print(f"Generation STDOUT: {generation_result.stdout}")
                print(f"Generation STDERR: {generation_result.stderr}")
                assert False, f"Generation step failed with return code {generation_result.returncode}"

            # Verify generation succeeded
            assert f"Loaded quantized model from: {quantized_checkpoint_dir}" in generation_result.stdout, (
                f"Checkpoint loading message not found. Output: {generation_result.stdout}"
            )
            assert "Testing quantized model with custom prompts" in generation_result.stdout, (
                f"Generation test message not found. Output: {generation_result.stdout}"
            )
            assert "Generation completed successfully!" in generation_result.stdout, (
                f"Generation completion message not found. Output: {generation_result.stdout}"
            )

            print("✓ Generation completed successfully")
            print("SUCCESS: Complete single GPU quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during single GPU quantization workflow test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "quant_tp,quant_pp,gen_tp,gen_pp,test_name",
        [
            (2, 1, 1, 1, "TP2_to_Single"),  # quantize with tp=2, generate with tp=1
            (1, 1, 1, 2, "PP1_to_PP2"),  # quantize with pp=1, generate with pp=2
            (1, 2, 1, 1, "PP2_to_Single"),  # additional: quantize pp=2, generate single
        ],
    )
    def test_quantization_and_generation_parallelism(self, tmp_path, quant_tp, quant_pp, gen_tp, gen_pp, test_name):
        """
        Test quantization and generation with different parallelism configurations.

        Args:
            tmp_path: Pytest temporary path fixture
            quant_tp: Tensor parallelism size for quantization
            quant_pp: Pipeline parallelism size for quantization
            gen_tp: Tensor parallelism size for generation
            gen_pp: Pipeline parallelism size for generation
            test_name: Name of the test for identification
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / f"checkpoints_{test_name.lower()}"
        base_dir.mkdir(exist_ok=True)

        try:
            print(f"=== STEP 1: Quantizing model with TP={quant_tp}, PP={quant_pp} ===")
            # Step 1: Quantize the model with specified parallelism
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                base_dir, quant_cfg="fp8", tp=quant_tp, pp=quant_pp
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

            print(f"=== STEP 2: Testing generation with TP={gen_tp}, PP={gen_pp} ===")
            # Step 2: Test generation with different parallelism configuration
            generation_result = self._run_generation(quantized_checkpoint_dir, tp=gen_tp, pp=gen_pp)

            if generation_result.returncode != 0:
                print(f"Generation STDOUT: {generation_result.stdout}")
                print(f"Generation STDERR: {generation_result.stderr}")
                assert False, f"Generation step for {test_name} failed with return code {generation_result.returncode}"

            # Verify generation succeeded with correct parallelism
            assert f"Loaded quantized model from: {quantized_checkpoint_dir}" in generation_result.stdout, (
                f"Checkpoint loading message not found in {test_name}. Output: {generation_result.stdout}"
            )
            assert f"Tensor parallel size: {gen_tp}" in generation_result.stdout, (
                f"Generation TP setting not found in {test_name}. Output: {generation_result.stdout}"
            )
            assert f"Pipeline parallel size: {gen_pp}" in generation_result.stdout, (
                f"Generation PP setting not found in {test_name}. Output: {generation_result.stdout}"
            )
            assert "Testing quantized model with custom prompts" in generation_result.stdout, (
                f"Generation test message not found in {test_name}. Output: {generation_result.stdout}"
            )
            assert "Generation completed successfully!" in generation_result.stdout, (
                f"Generation completion message not found in {test_name}. Output: {generation_result.stdout}"
            )

            print(f"✓ Generation completed with TP={gen_tp}, PP={gen_pp}")
            print(f"SUCCESS: {test_name} quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during {test_name} quantization workflow test: {e}")
            raise
