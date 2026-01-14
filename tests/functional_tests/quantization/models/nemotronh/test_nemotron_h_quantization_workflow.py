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
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


# Overrides for 8B size (same as test_nemotron_h_conversion.py)
HF_NEMOTRONH_TOY_MODEL_OVERRIDES = {
    "attention_head_dim": 48,
    "chunk_size": 48,
    "expand": 2,
    "hidden_size": 768,
    "hybrid_override_pattern": "M*M-",
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 64,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 24,
    "max_position_embeddings": 8192,
    "n_groups": 8,
    "num_attention_heads": 16,
    "num_hidden_layers": 4,
    "num_key_value_heads": 8,
    "ssm_state_size": 128,
    "vocab_size": 131072,
}


class TestNemotronHQuantizationWorkflow:
    """
    Test complete Nemotron-H quantization workflow: quantize HuggingFace Nemotron-H models
    to Megatron format with different parallelism, then test text generation from the
    quantized checkpoints.
    """

    @pytest.fixture(scope="class")
    def nemotron_h_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Nemotron-H toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("nemotron_h_toy_model")
        model_dir = temp_dir / "nemotron_h_toy"

        # Create Nemotron-H toy model config by starting with 8B and applying overrides
        config = AutoConfig.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        for k, v in HF_NEMOTRONH_TOY_MODEL_OVERRIDES.items():
            setattr(config, k, v)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path="nvidia/Nemotron-H-8B-Base-8K",
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id="nvidia/Nemotron-H-8B-Base-8K",
        )
        model = model_class(config)
        model = model.bfloat16()  # Use .bfloat16() method

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Download and save tokenizer from a reference Nemotron-H model
        tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        # Save model, config, and modeling code to directory
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = model.config.to_dict()
        config_to_save["torch_dtype"] = "bfloat16"
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def _run_quantization(self, model_path, base_dir, quant_cfg="fp8", tp=1, pp=1):
        """
        Helper method to run quantization step for Nemotron-H models.

        Args:
            model_path: Path to the HuggingFace model directory
            base_dir: Base directory to save the quantized checkpoint
            quant_cfg: Quantization configuration to use
            tp: Tensor parallelism size
            pp: Pipeline parallelism size

        Returns:
            tuple: (subprocess.CompletedProcess, actual_output_path)
        """
        # Create descriptive checkpoint name including configuration
        checkpoint_name = f"nemotron_h_quantized_{quant_cfg}_tp{tp}_pp{pp}"
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
            model_path,  # Use local toy model path instead of downloading
            "--export-quant-cfg",
            quant_cfg,
            "--megatron-save-path",
            str(output_dir),
            "--trust-remote-code",
            "--disable-hf-datasets-file-lock",
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )
        return result, output_dir

    def _run_generation(self, model_path, checkpoint_dir, tp=1, pp=1):
        """
        Helper method to run generation step for Nemotron-H models.

        Args:
            model_path: Path to the HuggingFace model directory
            checkpoint_dir: Directory containing the quantized checkpoint
            tp: Tensor parallelism size for generation
            pp: Pipeline parallelism size for generation

        Returns:
            subprocess.CompletedProcess: Result of generation process
        """
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
            "examples/quantization/ptq_generate.py",
            "--hf-model-id",
            model_path,  # Use local toy model path instead of downloading
            "--megatron-load-path",
            str(checkpoint_dir),
            "--trust-remote-code",
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])

        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )

    @pytest.mark.run_only_on("GPU")
    def test_nemotron_h_quantization_and_generation_with_parallelism(self, nemotron_h_toy_model_path, tmp_path):
        """
        Test complete Nemotron-H workflow: quantize with tensor parallelism (tp=2),
        then generate with pipeline parallelism (pp=2).

        This test uses a toy Nemotron-H model with:
        - 4 layers, hybrid pattern M*M-, hidden_size 768
        - Quantization: torchrun --nproc_per_node 2 with --tp 2 (2*1=2 processes)
        - Generation: torchrun --nproc_per_node 2 with --pp 2 (1*2=2 processes)

        Args:
            nemotron_h_toy_model_path: Path to the toy Nemotron-H model (from fixture)
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / "checkpoints_nemotron_h_parallel"
        base_dir.mkdir(exist_ok=True)

        try:
            print("=== STEP 1: Quantizing Nemotron-H toy model with TP=2, PP=1 ===")
            # Step 1: Quantize the model with tensor parallelism
            # tp=2, pp=1 gives 2 total processes (2*1=2)
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                nemotron_h_toy_model_path, base_dir, quant_cfg="fp8", tp=2, pp=1
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
            print(f"  Checkpoint contents: {[item.name for item in checkpoint_contents]}")

            print("=== STEP 2: Testing generation from quantized Nemotron-H checkpoint with TP=1, PP=2 ===")
            # Step 2: Test generation from the quantized checkpoint with different parallelism
            # tp=1, pp=2 gives 2 total processes (1*2=2)
            generation_result = self._run_generation(nemotron_h_toy_model_path, quantized_checkpoint_dir, tp=1, pp=2)

            if generation_result.returncode != 0:
                print(f"Generation STDOUT: {generation_result.stdout}")
                print(f"Generation STDERR: {generation_result.stderr}")
                assert False, f"Generation step failed with return code {generation_result.returncode}"

            # Verify generation succeeded
            # Note: stdout may have line wrapping, so we normalize it by removing newlines within the output
            stdout_normalized = generation_result.stdout.replace("\n", "")
            assert (
                "Loaded quantized model from:" in generation_result.stdout
                and str(quantized_checkpoint_dir) in stdout_normalized
            ), f"Checkpoint loading message not found. Output: {generation_result.stdout}"
            assert "Testing quantized model with custom prompts" in generation_result.stdout, (
                f"Generation test message not found. Output: {generation_result.stdout}"
            )
            assert "Generation completed successfully!" in generation_result.stdout, (
                f"Generation completion message not found. Output: {generation_result.stdout}"
            )

            print("✓ Generation completed successfully")
            print("SUCCESS: Complete Nemotron-H quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during Nemotron-H quantization workflow test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "quant_tp,quant_pp,gen_tp,gen_pp,test_name",
        [
            (2, 1, 1, 2, "TP2_to_PP2"),  # quantize with tp=2,pp=1, generate with tp=1,pp=2
        ],
    )
    def test_nemotron_h_quantization_and_generation_parallelism(
        self, nemotron_h_toy_model_path, tmp_path, quant_tp, quant_pp, gen_tp, gen_pp, test_name
    ):
        """
        Test Nemotron-H quantization and generation with different parallelism configurations.

        Args:
            nemotron_h_toy_model_path: Path to the toy Nemotron-H model (from fixture)
            tmp_path: Pytest temporary path fixture
            quant_tp: Tensor parallelism size for quantization
            quant_pp: Pipeline parallelism size for quantization
            gen_tp: Tensor parallelism size for generation
            gen_pp: Pipeline parallelism size for generation
            test_name: Name of the test for identification
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / f"checkpoints_nemotron_h_{test_name.lower()}"
        base_dir.mkdir(exist_ok=True)

        try:
            print(f"=== STEP 1: Quantizing Nemotron-H toy model with TP={quant_tp}, PP={quant_pp} ===")
            # Step 1: Quantize the model with specified parallelism
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                nemotron_h_toy_model_path, base_dir, quant_cfg="fp8", tp=quant_tp, pp=quant_pp
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
            generation_result = self._run_generation(
                nemotron_h_toy_model_path, quantized_checkpoint_dir, tp=gen_tp, pp=gen_pp
            )

            if generation_result.returncode != 0:
                print(f"Generation STDOUT: {generation_result.stdout}")
                print(f"Generation STDERR: {generation_result.stderr}")
                assert False, f"Generation step for {test_name} failed with return code {generation_result.returncode}"

            # Verify generation succeeded with correct parallelism
            # Note: stdout may have line wrapping, so we normalize it by removing newlines within the output
            stdout_normalized = generation_result.stdout.replace("\n", "")
            assert (
                "Loaded quantized model from:" in generation_result.stdout
                and str(quantized_checkpoint_dir) in stdout_normalized
            ), f"Checkpoint loading message not found in {test_name}. Output: {generation_result.stdout}"
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
            print(f"SUCCESS: {test_name} Nemotron-H quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during {test_name} Nemotron-H quantization workflow test: {e}")
            raise
