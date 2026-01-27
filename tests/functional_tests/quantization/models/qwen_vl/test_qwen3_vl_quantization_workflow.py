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
Functional tests for Qwen3 VL (Vision-Language) quantization workflow.

This module tests the complete quantization workflow for dense Qwen3 VL models:
1. Create a tiny toy model for fast testing
2. Run quantization using quantize_vlm.py
3. Run generation using ptq_generate_vlm.py

Example run commands:
    # Run all quantization workflow tests
    pytest tests/functional_tests/quantization/models/qwen_vl/test_qwen3_vl_quantization_workflow.py

    # Run specific test
    pytest tests/functional_tests/quantization/models/qwen_vl/test_qwen3_vl_quantization_workflow.py::TestQwen3VLQuantizationWorkflow::test_qwen3_vl_quantization_and_generation

Note: These tests use small toy models for fast testing.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import Qwen3VLConfig


HF_QWEN3_VL_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "vision_token_id": 151654,
    "image_token_id": 151655,
    "video_token_id": 151656,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_vl",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vision_config": {
        "depth": 27,
        "hidden_size": 1152,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 4304,
        "in_channels": 3,
        "num_heads": 16,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 256,
        "deepstack_visual_indexes": [1, 2, 3],
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "text_config": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 152064,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 152064,
}


class TestQwen3VLQuantizationWorkflow:
    """
    Test complete Qwen3 VL (dense) quantization workflow: quantize HuggingFace Qwen3 VL models
    to Megatron format with tensor parallelism, then test image+text generation from the
    quantized checkpoints.
    """

    @pytest.fixture(scope="class")
    def qwen3_vl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3 VL toy model to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen3_vl_quantization_toy_model")
        model_dir = temp_dir / "qwen3_vl_toy"

        # Create Qwen3 VL config from the toy model config
        config = Qwen3VLConfig(**HF_QWEN3_VL_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Set rope_scaling on text_config
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}

        # Create model with random weights and convert to bfloat16
        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        # Download and save tokenizer and processor from a reference Qwen3 VL model
        try:
            from transformers import AutoProcessor

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            tokenizer.save_pretrained(model_dir)

            # Also save the image processor
            processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            processor.save_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: Could not download tokenizer/processor, creating minimal files: {e}")
            # Create minimal tokenizer files if download fails
            tokenizer_config = {
                "tokenizer_class": "Qwen2Tokenizer",
                "vocab_size": 152064,
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            }
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            preprocessor_config = {
                "image_processor_type": "Qwen2VLImageProcessor",
                "do_resize": True,
                "size": {"height": 224, "width": 224},
                "do_normalize": True,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "do_convert_rgb": True,
            }
            with open(model_dir / "preprocessor_config.json", "w") as f:
                json.dump(preprocessor_config, f, indent=2)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN3_VL_TOY_MODEL_CONFIG, f, indent=2)

        print(f"Created toy model at: {model_dir}")
        return str(model_dir)

    def _run_quantization(self, model_path, base_dir, quant_cfg="fp8", tp=1, pp=1):
        """
        Helper method to run quantization step for Qwen3 VL models.

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
        checkpoint_name = f"qwen3_vl_quantized_{quant_cfg}_tp{tp}_pp{pp}"
        output_dir = base_dir / checkpoint_name
        output_dir.mkdir(exist_ok=True)

        # Calculate total number of processes needed
        total_procs = max(tp * pp, 1)

        import sys

        python_executable = sys.executable

        # Base command for VLM quantization
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
            "examples/quantization/quantize_vlm.py",
            "--hf-model-id",
            model_path,
            "--export-quant-cfg",
            quant_cfg,
            "--megatron-save-path",
            str(output_dir),
            "--calib-size",
            "8",  # Use small calib size for testing
            "--test-image-path",
            "https://picsum.photos/id/237/400/300",  # Reliable placeholder image service
            "--use-random-calib",  # Use random images for offline CI environments
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
        Helper method to run generation step for Qwen3 VL models.

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

        # Base command for VLM generation
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
            "examples/quantization/ptq_generate_vlm.py",
            "--hf-model-id",
            model_path,
            "--megatron-load-path",
            str(checkpoint_dir),
            "--image-path",
            "https://picsum.photos/id/237/400/300",  # Reliable placeholder image service
            "--prompts",
            "Describe this image.",
            "--osl",
            "16",  # Short output for testing
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
    def test_qwen3_vl_quantization_and_generation(self, qwen3_vl_toy_model_path, tmp_path):
        """
        Test complete Qwen3 VL workflow: quantize with tensor parallelism (tp=2),
        then generate with tensor parallelism (tp=2).

        This test uses a toy Qwen3 VL model with:
        - 4 layers, hidden_size 256
        - Quantization: torchrun --nproc_per_node 2 with --tp 2
        - Generation: torchrun --nproc_per_node 2 with --tp 2

        Note: PP (pipeline parallelism) is not used because the toy model's
        deepstack_visual_indexes [1, 2, 3] requires all visual embeds to be
        on the first pipeline stage, which is incompatible with PP > 1
        when there are only 4 layers.

        Args:
            qwen3_vl_toy_model_path: Path to the toy Qwen3 VL model (from fixture)
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / "checkpoints_qwen3_vl"
        base_dir.mkdir(exist_ok=True)

        try:
            print("=== STEP 1: Quantizing Qwen3 VL toy model with TP=2, PP=1 ===")
            # Step 1: Quantize the model with tensor parallelism
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                qwen3_vl_toy_model_path, base_dir, quant_cfg="fp8", tp=2, pp=1
            )

            if quantize_result.returncode != 0:
                print(f"Quantization STDOUT: {quantize_result.stdout}")
                print(f"Quantization STDERR: {quantize_result.stderr}")
                pytest.fail(f"Quantization step failed with return code {quantize_result.returncode}")

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

            print("=== STEP 2: Testing generation from quantized Qwen3 VL checkpoint with TP=2, PP=1 ===")
            # Step 2: Test generation from the quantized checkpoint with same parallelism
            generation_result = self._run_generation(qwen3_vl_toy_model_path, quantized_checkpoint_dir, tp=2, pp=1)

            if generation_result.returncode != 0:
                print(f"Generation STDOUT: {generation_result.stdout}")
                print(f"Generation STDERR: {generation_result.stderr}")
                pytest.fail(f"Generation step failed with return code {generation_result.returncode}")

            # Verify generation succeeded
            stdout_normalized = generation_result.stdout.replace("\n", "")
            assert (
                "Loaded quantized model from:" in generation_result.stdout
                and str(quantized_checkpoint_dir) in stdout_normalized
            ), f"Checkpoint loading message not found. Output: {generation_result.stdout}"
            assert "Testing quantized VLM model with image and prompt" in generation_result.stdout, (
                f"Generation test message not found. Output: {generation_result.stdout}"
            )
            assert "Generation completed successfully!" in generation_result.stdout, (
                f"Generation completion message not found. Output: {generation_result.stdout}"
            )

            print("✓ Generation completed successfully")
            print("SUCCESS: Complete Qwen3 VL quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during Qwen3 VL quantization workflow test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "quant_tp,quant_pp,gen_tp,gen_pp,test_name",
        [
            (2, 1, 1, 1, "TP2_to_TP1"),  # quantize with tp=2,pp=1, generate with tp=1,pp=1
        ],
    )
    def test_qwen3_vl_quantization_and_generation_parallelism(
        self, qwen3_vl_toy_model_path, tmp_path, quant_tp, quant_pp, gen_tp, gen_pp, test_name
    ):
        """
        Test Qwen3 VL quantization and generation with different parallelism configurations.

        Args:
            qwen3_vl_toy_model_path: Path to the toy Qwen3 VL model (from fixture)
            tmp_path: Pytest temporary path fixture
            quant_tp: Tensor parallelism size for quantization
            quant_pp: Pipeline parallelism size for quantization
            gen_tp: Tensor parallelism size for generation
            gen_pp: Pipeline parallelism size for generation
            test_name: Name of the test for identification
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / f"checkpoints_qwen3_vl_{test_name.lower()}"
        base_dir.mkdir(exist_ok=True)

        try:
            print(f"=== STEP 1: Quantizing Qwen3 VL toy model with TP={quant_tp}, PP={quant_pp} ===")
            # Step 1: Quantize the model with specified parallelism
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                qwen3_vl_toy_model_path, base_dir, quant_cfg="fp8", tp=quant_tp, pp=quant_pp
            )

            if quantize_result.returncode != 0:
                print(f"Quantization STDOUT: {quantize_result.stdout}")
                print(f"Quantization STDERR: {quantize_result.stderr}")
                pytest.fail(f"Quantization step for {test_name} failed with return code {quantize_result.returncode}")

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
                qwen3_vl_toy_model_path, quantized_checkpoint_dir, tp=gen_tp, pp=gen_pp
            )

            if generation_result.returncode != 0:
                print(f"Generation STDOUT: {generation_result.stdout}")
                print(f"Generation STDERR: {generation_result.stderr}")
                pytest.fail(f"Generation step for {test_name} failed with return code {generation_result.returncode}")

            # Verify generation succeeded with correct parallelism
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
            assert "Testing quantized VLM model with image and prompt" in generation_result.stdout, (
                f"Generation test message not found in {test_name}. Output: {generation_result.stdout}"
            )
            assert "Generation completed successfully!" in generation_result.stdout, (
                f"Generation completion message not found in {test_name}. Output: {generation_result.stdout}"
            )

            print(f"✓ Generation completed with TP={gen_tp}, PP={gen_pp}")
            print(f"SUCCESS: {test_name} Qwen3 VL quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during {test_name} Qwen3 VL quantization workflow test: {e}")
            raise
