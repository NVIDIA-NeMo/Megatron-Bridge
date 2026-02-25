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

Note: These tests use small proxy/toy models for fast generation testing.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import (
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
)
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
        "deepstack_visual_indexes": [1],
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
        "rope_scaling": {"rope_type": "default", "mrope_section": [16, 24, 24]},
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
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_vl_moe",
    "num_attention_heads": 16,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "attention_bias": True,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 384,
    "decoder_sparse_step": 1,
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
        "out_hidden_size": 2048,
        "deepstack_visual_indexes": [1],
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "text_config": {
        "hidden_size": 2048,
        "intermediate_size": 3072,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 152064,  # Keep full vocab to match tokenizer
        "max_position_embeddings": 32768,
        "rope_theta": 5000000.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 384,
        "decoder_sparse_step": 1,
        "rope_scaling": {"rope_type": "default", "mrope_section": [16, 24, 24]},
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 152064,  # Keep full vocab to match tokenizer (embeddings are cheap)
}


class TestQwen3VLGeneration:
    """
    Test Qwen3 VL model generation using HF to Megatron conversion with vision inputs.
    Uses small proxy/toy models for fast generation testing.
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
        temp_dir = tmp_path_factory.mktemp("qwen3_vl_generation_toy_model")
        model_dir = temp_dir / "qwen3_vl_toy"

        # Create Qwen3 VL config from the toy model config
        config = Qwen3VLConfig(**HF_QWEN3_VL_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Set rope_parameters on text_config (transformers 5.0+ uses rope_parameters with rope_type)
        # rope_type must be "default" - MRoPE is indicated by the mrope_section parameter
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_parameters = {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
                "rope_theta": 1000000.0,
            }

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

    @pytest.fixture(scope="class")
    def qwen3_vl_moe_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3 VL MoE toy model to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace MoE model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen3_vl_moe_generation_toy_model")
        model_dir = temp_dir / "qwen3_vl_moe_toy"

        # Create Qwen3 VL MoE config from the toy model config
        config = Qwen3VLMoeConfig(**HF_QWEN3_VL_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Set rope_parameters on text_config (transformers 5.0+ uses rope_parameters with rope_type)
        # rope_type must be "default" - MRoPE is indicated by the mrope_section parameter
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_parameters = {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
                "rope_theta": 5000000.0,
            }

        # Create model with random weights and convert to bfloat16
        model = Qwen3VLMoeForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        try:
            from transformers import AutoProcessor

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            tokenizer.save_pretrained(model_dir)

            # Also save the image processor
            processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            processor.save_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: Could not download tokenizer/processor, creating minimal files: {e}")
            tokenizer_config = {
                "tokenizer_class": "Qwen2Tokenizer",
                "vocab_size": 152064,
            }
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

            # Create minimal preprocessor config for image processing
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

        model.save_pretrained(model_dir, safe_serialization=True)
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN3_VL_MOE_TOY_MODEL_CONFIG, f, indent=2)

        print(f"Created MoE toy model at: {model_dir}")
        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_vl_8b_image_generation(self, qwen3_vl_toy_model_path):
        """
        Test Qwen3 VL toy model with image generation.
        Uses a small proxy model instead of the full 8B model for fast testing.
        Uses real image to test vision-language pipeline with corrected vision config.

        Args:
            qwen3_vl_toy_model_path: Path to the toy Qwen3 VL model (from fixture)
        """
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/conversion/hf_to_megatron_generate_vlm.py",
            f"--hf_model_path={qwen3_vl_toy_model_path}",
            "--image_path=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "--prompt=Describe this image.",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
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

            if result.returncode != 0:
                assert False, f"Qwen3-VL toy model generation failed with return code {result.returncode}"

            print("SUCCESS: Qwen3-VL toy model generation test completed successfully")

        except subprocess.TimeoutExpired:
            assert False, "Qwen3-VL toy model generation test timed out after 5 minutes"
        except Exception as e:
            print(f"Error during Qwen3-VL toy model generation test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_vl_30b_a3b_moe_image_generation(self, qwen3_vl_moe_toy_model_path):
        """
        Test Qwen3 VL MoE toy model with image generation and EP=2.
        Uses a small proxy MoE model instead of the full 30B model for fast testing.
        Uses real image to test vision-language pipeline with corrected vision config.

        Args:
            qwen3_vl_moe_toy_model_path: Path to the toy Qwen3 VL MoE model (from fixture)
        """
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/conversion/hf_to_megatron_generate_vlm.py",
            f"--hf_model_path={qwen3_vl_moe_toy_model_path}",
            "--prompt=Describe this image.",
            "--ep=2",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
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

            if result.returncode != 0:
                assert False, f"Qwen3-VL MoE toy model generation failed with return code {result.returncode}"

            print("SUCCESS: Qwen3-VL MoE toy model generation test completed successfully")

        except subprocess.TimeoutExpired:
            assert False, "Qwen3-VL MoE toy model generation test timed out after 5 minutes"
        except Exception as e:
            print(f"Error during Qwen3-VL MoE toy model generation test: {e}")
            raise
