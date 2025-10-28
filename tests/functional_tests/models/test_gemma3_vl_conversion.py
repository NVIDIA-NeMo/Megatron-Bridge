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
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, GemmaTokenizer


# Gemma3 VL toy model configuration based on typical Gemma3 VL structure
HF_GEMMA3_VL_TOY_MODEL_CONFIG = {
    "architectures": ["Gemma3ForConditionalGeneration"],
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
    "eos_token_id": [1, 106],
    "image_token_index": 262144,
    "initializer_range": 0.02,
    "mm_tokens_per_image": 256,
    "model_type": "gemma3",
    "text_config": {
        "hidden_size": 512,
        "intermediate_size": 10240,
        "model_type": "gemma3_text",
        "num_hidden_layers": 8,
        "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
        "sliding_window": 1024,
        "num_attention_heads": 16,
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "4.50.0.dev0",
    "vision_config": {
        "hidden_size": 256,
        "image_size": 896,
        "intermediate_size": 1024,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 8,
        "patch_size": 14,
        "vision_use_head": False,
    },
}


class TestGemma3VLConversion:
    """
    Test Gemma3 VL model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def gemma3_vl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Gemma3 VL toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("gemma3_vl_toy_model")
        model_dir = temp_dir / "gemma3_vl_toy"

        # Create Gemma3 VL config from the toy model config
        # Note: We need to create a mock config since Gemma3ForConditionalGeneration might not be available
        # This is a simplified approach for testing purposes
        config_dict = HF_GEMMA3_VL_TOY_MODEL_CONFIG.copy()

        # Try to create the model if Gemma3ForConditionalGeneration is available

        # Create config object
        config = Gemma3Config(**config_dict)
        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        model = Gemma3ForConditionalGeneration(config)
        model = model.bfloat16()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Download and save tokenizer from a reference Gemma model
        # We use a Gemma model for tokenizer artifacts since they should be compatible
        try:
            tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b")
            tokenizer.save_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: Could not download tokenizer, creating minimal tokenizer files: {e}")
            # Create minimal tokenizer files if download fails
            tokenizer_config = {
                "tokenizer_class": "GemmaTokenizer",
                "vocab_size": 262144,
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }

            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Save config.json explicitly to ensure compatibility
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create minimal model weights file if none exists
        weights_file = model_dir / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_dir / "pytorch_model.bin"
            if not weights_file.exists():
                # Create a minimal weights file for testing
                minimal_weights = {
                    "language_model.model.embed_tokens.weight": torch.randn(262144, 2560, dtype=torch.bfloat16),
                    "language_model.model.norm.weight": torch.randn(2560, dtype=torch.bfloat16),
                }
                torch.save(minimal_weights, weights_file)

        return str(model_dir)

    def test_toy_model_creation(self, gemma3_vl_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            gemma3_vl_toy_model_path: Path to the toy Gemma3 VL model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(gemma3_vl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "gemma3"
        assert "text_config" in config_data
        assert "vision_config" in config_data
        assert config_data["text_config"]["hidden_size"] == 512
        assert config_data["text_config"]["num_hidden_layers"] == 8
        assert config_data["text_config"]["num_attention_heads"] == 16
        assert config_data["vision_config"]["hidden_size"] == 256

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_gemma3_vl_conversion_parallelism(self, gemma3_vl_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Gemma3 VL model conversion with different parallelism configurations.

        Args:
            gemma3_vl_toy_model_path: Path to the toy Gemma3 VL model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"gemma3_vl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism configuration on our toy model
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/workspace/.coverage",
            "--source=/workspace/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            gemma3_vl_toy_model_path,  # Use our local toy model instead of downloading
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)
        print(cmd)

        # Check that the conversion completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Gemma3 VL {test_name} conversion failed with return code {result.returncode}"

        # Verify that the converted model was saved
        # The output directory should be named after the last part of the model path
        model_name = Path(gemma3_vl_toy_model_path).name  # "gemma3_vl_toy"
        converted_model_dir = test_output_dir / model_name
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

        # Verify the config contains Gemma3 VL-specific parameters
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "gemma3", "Model type should be gemma3"
        assert "text_config" in saved_config, "VL model should have text_config"
        assert "vision_config" in saved_config, "VL model should have vision_config"
        assert saved_config["text_config"]["hidden_size"] == 512, "Hidden size should match toy config"
        assert saved_config["text_config"]["num_attention_heads"] == 16, (
            "Number of attention heads should match toy config"
        )
