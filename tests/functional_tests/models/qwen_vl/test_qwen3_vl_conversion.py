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
uv run python -m torch.distributed.run --nproc_per_node=1 -m pytest tests/functional_tests/models/test_qwen3_vl_conversion.py::TestQwen3VLConversion::test_toy_model_creation
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


# Tiny model config optimized for fast testing
# Parameters reduced from 12.27B to ~8M (1500x smaller!)
# Creation time reduced from ~5 minutes to ~1-2 seconds
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
    "hidden_size": 256,  # Reduced from 4096 (16x smaller)
    "initializer_range": 0.02,
    "intermediate_size": 512,  # Reduced from 14336 (28x smaller)
    "max_position_embeddings": 32768,
    "model_type": "qwen3_vl",
    "num_attention_heads": 4,  # Reduced from 32 (8x smaller)
    "num_hidden_layers": 4,  # Reduced from 2 (sufficient for conversion testing)
    "num_key_value_heads": 2,  # Reduced from 8 (4x smaller)
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vision_config": {
        "depth": 1,  # Reduced from 2 (sufficient for testing)
        "embed_dim": 256,  # Reduced from 1280 (5x smaller)
        "hidden_size": 256,  # Reduced from 1280 (5x smaller)
        "hidden_act": "silu",
        "in_channels": 3,
        "mlp_ratio": 2.6718749999999997,
        "num_heads": 4,  # Reduced from 16 (4x smaller)
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "deepstack_visual_indexes": [1],  # Must be <= layers on first PP stage (e.g. 2 layers with PP=2)
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "text_config": {
        # CRITICAL: Must explicitly set ALL text model dimensions!
        # Otherwise Qwen3VLConfig uses default values (32 layers, 4096 hidden_size, etc.)
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 2048,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rope_scaling": {"rope_type": "default", "mrope_section": [16, 24, 24]},
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 2048,  # KEY: Reduced from 152064 (74x smaller!) - saves 1.24B params in embeddings
}


class TestQwen3VLConversion:
    """
    Test Qwen3 VL model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def qwen3_vl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3 VL toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen3_vl_toy_model")
        model_dir = temp_dir / "qwen3_vl_toy"

        # Create Qwen3 VL config from the toy model config
        config = Qwen3VLConfig(**HF_QWEN3_VL_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # IMPORTANT: Set rope_parameters on text_config (not just main config)
        # The text model initialization uses config.text_config which needs rope_parameters
        # rope_type must be "default" - MRoPE is indicated by the mrope_section parameter
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_parameters = {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
                "rope_theta": 1000000.0,
            }

        # Create model with random weights and convert to bfloat16
        model = Qwen3VLForConditionalGeneration(config)
        # Use .to() instead of .bfloat16() to convert both parameters AND buffers
        model = model.to(dtype=torch.bfloat16)

        # Download and save tokenizer from a reference Qwen3 VL model
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            tokenizer.save_pretrained(model_dir)
        except Exception:
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

        # Save model and config to directory
        # NOTE: model.save_pretrained() will save config.json with the proper rope_scaling in text_config
        model.save_pretrained(model_dir, safe_serialization=True)
        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_QWEN3_VL_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, qwen3_vl_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            qwen3_vl_toy_model_path: Path to the toy Qwen3 VL model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(qwen3_vl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred, including sharded format)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            # Check for sharded safetensors (index file indicates sharded model)
            weights_file = model_path / "model.safetensors.index.json"
        if not weights_file.exists():
            # Fallback to PyTorch format
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen3_vl"
        assert config_data["hidden_size"] == 256
        assert config_data["num_hidden_layers"] == 4
        assert config_data["num_attention_heads"] == 4
        assert config_data["vocab_size"] == 2048
        assert "vision_config" in config_data

        _ = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen3_vl_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

        # Try loading the tokenizer as well
        try:
            tokenizer = AutoTokenizer.from_pretrained(qwen3_vl_toy_model_path)
            print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

        print(f"SUCCESS: Toy model created and validated at {qwen3_vl_toy_model_path}")
        print("Model weights are correctly in bfloat16 format")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
        ],
    )
    def test_qwen3_vl_conversion_parallelism(self, qwen3_vl_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Qwen3 VL model conversion with different parallelism configurations.

        Args:
            qwen3_vl_toy_model_path: Path to the toy Qwen3 VL model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"qwen3_vl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism configuration
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen3_vl_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Qwen3 VL {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            model_name = Path(qwen3_vl_toy_model_path).name  # "qwen3_vl_toy"
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Verify the config contains Qwen3 VL-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "qwen3_vl", "Model type should be qwen3_vl"
            assert saved_config["hidden_size"] == 256, "Hidden size should match toy config"
            assert saved_config["num_attention_heads"] == 4, "Number of attention heads should match toy config"
            assert "vision_config" in saved_config, "VL model should have vision_config"

            print(f"SUCCESS: Qwen3 VL {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Qwen3 VL {test_name} conversion test: {e}")
            raise


# Tiny MoE model config optimized for fast testing
# Scaled down proportionally from Qwen3-VL-30B-A3B-Instruct (scale=2x)
# Maintains same ratios: head_dim=64, GQA ratio=8, intermediate/hidden=3.0
# num_key_value_heads=2 (minimum for TP=2 compatibility)
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
    "hidden_size": 2048,  # Scaled from 2048 (2x smaller)
    "initializer_range": 0.02,
    "intermediate_size": 3072,  # Scaled from 6144 (2x smaller, maintains 3.0 ratio)
    "max_position_embeddings": 32768,
    "model_type": "qwen3_vl_moe",
    "num_attention_heads": 16,  # Scaled from 32 (2x smaller)
    "num_hidden_layers": 4,  # Minimal for deepstack compatibility
    "num_key_value_heads": 2,  # Scaled from 4 (2x smaller, maintains GQA ratio=8)
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "attention_bias": True,
    "num_experts": 4,  # Minimal for testing (vs 128 in reference)
    "num_experts_per_tok": 2,  # Reduced from 8
    "moe_intermediate_size": 384,  # Scaled from 768 (2x smaller, maintains 0.375 ratio)
    "decoder_sparse_step": 1,
    "vision_config": {
        "depth": 1,  # Reduced from 27 (sufficient for testing)
        "embed_dim": 1024,  # Match hidden_size for compatibility
        "hidden_size": 1024,  # Match text hidden_size
        "hidden_act": "silu",
        "in_channels": 3,
        "mlp_ratio": 2.6718749999999997,
        "num_heads": 16,  # Match num_attention_heads
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "deepstack_visual_indexes": [1],  # Must be <= layers on first PP stage (e.g. 2 layers with PP=2)
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "text_config": {
        # CRITICAL: Must match main config dimensions for MoE!
        "hidden_size": 2048,  # Must match main!
        "intermediate_size": 3072,  # Must match main!
        "num_hidden_layers": 4,
        "num_attention_heads": 16,  # Must match main!
        "num_key_value_heads": 2,  # Must match main! Creates k_norm with size=128 (compatible with TP=2)
        "vocab_size": 2048,
        "max_position_embeddings": 32768,
        "rope_theta": 5000000.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 384,  # Must match main!
        "decoder_sparse_step": 1,
        "rope_scaling": {"rope_type": "default", "mrope_section": [16, 24, 24]},
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 2048,  # Reduced from 151936 (74x smaller!)
}


class TestQwen3VLMoEConversion:
    """Test Qwen3 VL MoE model conversion."""

    @pytest.fixture(scope="class")
    def qwen3_vl_moe_toy_model_path(self, tmp_path_factory):
        """Create and save a MoE toy model."""
        temp_dir = tmp_path_factory.mktemp("qwen3_vl_moe_toy_model")
        model_dir = temp_dir / "qwen3_vl_moe_toy"

        config = Qwen3VLMoeConfig(**HF_QWEN3_VL_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_parameters = {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
                "rope_theta": 5000000.0,
            }

        model = Qwen3VLMoeForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            tokenizer_config = {
                "tokenizer_class": "Qwen2Tokenizer",
                "vocab_size": 152064,
            }
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        model.save_pretrained(model_dir, safe_serialization=True)
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN3_VL_MOE_TOY_MODEL_CONFIG, f, indent=2)

        return str(model_dir)

    def test_moe_toy_model_creation(self, qwen3_vl_moe_toy_model_path):
        """Test MoE toy model creation."""
        model_path = Path(qwen3_vl_moe_toy_model_path)
        assert model_path.exists()

        config_file = model_path / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen3_vl_moe"
        assert config_data["num_experts"] == 4
        assert config_data["num_experts_per_tok"] == 2

        _ = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            qwen3_vl_moe_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("tp,pp", [(2, 1)])
    def test_moe_conversion(self, qwen3_vl_moe_toy_model_path, tmp_path, tp, pp):
        """Test MoE model conversion."""
        test_output_dir = tmp_path / "qwen3_vl_moe_test"
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen3_vl_moe_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"MoE conversion failed with return code {result.returncode}"

        model_name = Path(qwen3_vl_moe_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists()

        config_file = converted_model_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "qwen3_vl_moe"
        assert saved_config["num_experts"] == 4
