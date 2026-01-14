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
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM


HF_QWEN3_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 262144,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_experts": 4,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "output_router_logits": False,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "router_aux_loss_coef": 0.001,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


class TestQwen3MoeQuantizationWorkflow:
    """
    Test complete Qwen3 MoE quantization workflow: quantize HuggingFace Qwen3 MoE models
    to Megatron format with expert parallelism, then test text generation from the
    quantized checkpoints.
    """

    @pytest.fixture(scope="class")
    def qwen3_moe_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3 MoE toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen3_moe_toy_model")
        model_dir = temp_dir / "qwen3_moe_toy"

        # Create Qwen3 MoE config from the toy model config
        config = Qwen3MoeConfig(**HF_QWEN3_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        model = Qwen3MoeForCausalLM(config)
        model = model.bfloat16()  # Use .bfloat16() method instead of .to()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Download and save tokenizer from a reference Qwen model
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        tokenizer.save_pretrained(model_dir)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_QWEN3_MOE_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def _run_quantization(self, model_path, base_dir, quant_cfg="fp8", tp=1, pp=1, etp=1):
        """
        Helper method to run quantization step for Qwen3 MoE models.

        Args:
            model_path: Path to the HuggingFace model directory
            base_dir: Base directory to save the quantized checkpoint
            quant_cfg: Quantization configuration to use
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            etp: Expert tensor parallelism size

        Returns:
            tuple: (subprocess.CompletedProcess, actual_output_path)
        """
        # Create descriptive checkpoint name including configuration
        checkpoint_name = f"qwen3_moe_quantized_{quant_cfg}_tp{tp}_pp{pp}_etp{etp}"
        output_dir = base_dir / checkpoint_name
        output_dir.mkdir(exist_ok=True)
        # Calculate total number of processes needed
        # For MoE models: etp is a special case of tp (experts split across same TP GPUs)
        # So total_procs = tp * pp (etp doesn't multiply)
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
            "--disable-hf-datasets-file-lock",
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])
        if etp > 1:
            cmd.extend(["--etp", str(etp)])

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )
        return result, output_dir

    def _run_generation(self, model_path, checkpoint_dir, tp=1, pp=1, etp=1):
        """
        Helper method to run generation step for Qwen3 MoE models.

        Args:
            model_path: Path to the HuggingFace model directory
            checkpoint_dir: Directory containing the quantized checkpoint
            tp: Tensor parallelism size for generation
            pp: Pipeline parallelism size for generation
            etp: Expert tensor parallelism size for generation

        Returns:
            subprocess.CompletedProcess: Result of generation process
        """
        # Calculate total number of processes needed
        # For MoE models: etp is a special case of tp (experts split across same TP GPUs)
        # So total_procs = tp * pp (etp doesn't multiply)
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
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])
        if etp > 1:
            cmd.extend(["--etp", str(etp)])

        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_moe_quantization_and_generation_with_expert_parallelism(self, qwen3_moe_toy_model_path, tmp_path):
        """
        Test complete Qwen3 MoE workflow: quantize with expert tensor parallelism (tp=2, etp=2),
        then generate with pipeline parallelism (pp=2).

        This test uses a toy Qwen3 MoE model with:
        - 2 layers, 4 experts, hidden_size 2048
        - Quantization: torchrun --nproc_per_node 2 with --tp 2 --etp 2 (2*1=2 processes, etp uses same TP GPUs)
        - Generation: torchrun --nproc_per_node 2 with --pp 2 (1*2=2 processes)

        Args:
            qwen3_moe_toy_model_path: Path to the toy Qwen3 MoE model (from fixture)
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / "checkpoints_qwen3_moe_expert_parallel"
        base_dir.mkdir(exist_ok=True)

        try:
            print("=== STEP 1: Quantizing Qwen3 MoE toy model with TP=2, ETP=2, PP=1 ===")
            # Step 1: Quantize the model with expert tensor parallelism
            # tp=2, etp=2, pp=1 gives 2 total processes (2*1=2, etp shares TP GPUs)
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                qwen3_moe_toy_model_path, base_dir, quant_cfg="fp8", tp=2, pp=1, etp=2
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

            print("=== STEP 2: Testing generation from quantized Qwen3 MoE checkpoint with TP=1, ETP=1, PP=2 ===")
            # Step 2: Test generation from the quantized checkpoint with different parallelism
            # tp=1, pp=2 gives 2 total processes (1*2=2)
            generation_result = self._run_generation(
                qwen3_moe_toy_model_path, quantized_checkpoint_dir, tp=1, pp=2, etp=1
            )

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
            print("SUCCESS: Complete Qwen3 MoE quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during Qwen3 MoE quantization workflow test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "quant_tp,quant_pp,quant_etp,gen_tp,gen_pp,gen_etp,test_name",
        [
            (2, 1, 2, 1, 2, 1, "TP2_ETP2_to_PP2"),  # quantize with tp=2,etp=2,pp=1, generate with tp=1,pp=2,etp=1
        ],
    )
    def test_qwen3_moe_quantization_and_generation_parallelism(
        self, qwen3_moe_toy_model_path, tmp_path, quant_tp, quant_pp, quant_etp, gen_tp, gen_pp, gen_etp, test_name
    ):
        """
        Test Qwen3 MoE quantization and generation with different parallelism configurations.

        Args:
            qwen3_moe_toy_model_path: Path to the toy Qwen3 MoE model (from fixture)
            tmp_path: Pytest temporary path fixture
            quant_tp: Tensor parallelism size for quantization
            quant_pp: Pipeline parallelism size for quantization
            quant_etp: Expert tensor parallelism size for quantization
            gen_tp: Tensor parallelism size for generation
            gen_pp: Pipeline parallelism size for generation
            gen_etp: Expert tensor parallelism size for generation
            test_name: Name of the test for identification
        """
        # Create temporary base directory for quantized checkpoint
        base_dir = tmp_path / f"checkpoints_qwen3_moe_{test_name.lower()}"
        base_dir.mkdir(exist_ok=True)

        try:
            print(f"=== STEP 1: Quantizing Qwen3 MoE toy model with TP={quant_tp}, PP={quant_pp}, ETP={quant_etp} ===")
            # Step 1: Quantize the model with specified parallelism
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                qwen3_moe_toy_model_path, base_dir, quant_cfg="fp8", tp=quant_tp, pp=quant_pp, etp=quant_etp
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

            print(f"✓ Quantization completed with TP={quant_tp}, PP={quant_pp}, ETP={quant_etp}")

            print(f"=== STEP 2: Testing generation with TP={gen_tp}, PP={gen_pp}, ETP={gen_etp} ===")
            # Step 2: Test generation with different parallelism configuration
            generation_result = self._run_generation(
                qwen3_moe_toy_model_path, quantized_checkpoint_dir, tp=gen_tp, pp=gen_pp, etp=gen_etp
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

            print(f"✓ Generation completed with TP={gen_tp}, PP={gen_pp}, ETP={gen_etp}")
            print(f"SUCCESS: {test_name} Qwen3 MoE quantization and generation workflow test passed")

        except Exception as e:
            print(f"Error during {test_name} Qwen3 MoE quantization workflow test: {e}")
            raise
