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
Functional tests for Qwen3 VL HF to Megatron generation.

Example run commands:
    # Run all generation tests
    pytest tests/functional_tests/models/qwen_vl/test_qwen3_vl_generation.py

    # Run specific test (dense model)
    pytest tests/functional_tests/models/qwen_vl/test_qwen3_vl_generation.py::TestQwen3VLGeneration::test_qwen3_vl_8b_image_generation

    # Run specific test (MOE model)
    pytest tests/functional_tests/models/qwen_vl/test_qwen3_vl_generation.py::TestQwen3VLGeneration::test_qwen3_vl_30b_a3b_moe_image_generation

    # Run with verbose output
    pytest -v -s tests/functional_tests/models/qwen_vl/test_qwen3_vl_generation.py
"""

import subprocess
from pathlib import Path

import pytest


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
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_vl",
    "num_attention_heads": 32,
    "num_hidden_layers": 2,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vision_config": {
        "depth": 2,
        "embed_dim": 1280,
        "hidden_size": 1280,
        "hidden_act": "silu",
        "in_channels": 3,
        "mlp_ratio": 2.6718749999999997,
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "text_config": {
        "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 152064,
}


HF_QWEN3_VL_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3VLMoeForConditionalGeneration"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "vision_token_id": 151654,
    "image_token_id": 151655,
    "video_token_id": 151656,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5632,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_vl_moe",
    "num_attention_heads": 16,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "attention_bias": True,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 2816,
    "decoder_sparse_step": 1,
    "vision_config": {
        "depth": 2,
        "embed_dim": 1280,
        "hidden_size": 1280,
        "hidden_act": "silu",
        "in_channels": 3,
        "mlp_ratio": 2.6718749999999997,
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "text_config": {
        "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 152064,
}


class TestQwen3VLGeneration:
    """
    Test Qwen3 VL model generation using HF to Megatron conversion with vision inputs.
    """

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_vl_8b_image_generation(self):
        """
        Test Qwen3 VL 8B Instruct model with real image generation.
        """
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/conversion/hf_to_megatron_generate_vlm.py",
            "--hf_model_path=Qwen/Qwen3-VL-8B-Instruct",
            "--image_path=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "--prompt=Describe this image.",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for model download
                cwd=Path(__file__).parent.parent.parent.parent.parent,
            )

            # Print output for debugging
            print("\n" + "=" * 80)
            print("STDOUT:")
            print(result.stdout)
            print("\n" + "=" * 80)
            print("STDERR:")
            print(result.stderr)
            print("=" * 80 + "\n")

            # Check that the generation completed successfully
            if result.returncode != 0:
                assert False, f"Qwen3-VL-8B image generation failed with return code {result.returncode}"

            # Verify output contains expected elements
            assert "GENERATED TEXT OUTPUT" in result.stdout, "Expected generation output not found"

            print("SUCCESS: Qwen3-VL-8B image generation test completed successfully")

        except subprocess.TimeoutExpired:
            assert False, "Qwen3-VL-8B image generation test timed out after 10 minutes"
        except Exception as e:
            print(f"Error during Qwen3-VL-8B image generation test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_vl_30b_a3b_moe_image_generation(self):
        """
        Test Qwen3 VL 30B-A3B (MOE) Instruct model with real image generation and EP=2.
        """
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/conversion/hf_to_megatron_generate_vlm.py",
            "--hf_model_path=Qwen/Qwen3-VL-30B-A3B-Instruct",
            "--image_path=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "--prompt=Describe this image.",
            "--ep=2",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for model download
                cwd=Path(__file__).parent.parent.parent.parent.parent,
            )

            # Print output for debugging
            print("\n" + "=" * 80)
            print("STDOUT:")
            print(result.stdout)
            print("\n" + "=" * 80)
            print("STDERR:")
            print(result.stderr)
            print("=" * 80 + "\n")

            # Check that the generation completed successfully
            if result.returncode != 0:
                assert False, f"Qwen3-VL-30B-A3B MOE image generation failed with return code {result.returncode}"

            # Verify output contains expected elements
            assert "GENERATED TEXT OUTPUT" in result.stdout, "Expected generation output not found"

            print("SUCCESS: Qwen3-VL-30B-A3B MOE image generation test completed successfully")

        except subprocess.TimeoutExpired:
            assert False, "Qwen3-VL-30B-A3B MOE image generation test timed out after 10 minutes"
        except Exception as e:
            print(f"Error during Qwen3-VL-30B-A3B MOE image generation test: {e}")
            raise
