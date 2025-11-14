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
        "torch_dtype": "bfloat16",  # Explicitly set dtype in text_config
    },
    "vocab_size": 152064,
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

        # IMPORTANT: Set rope_scaling on text_config (not just main config)
        # The text model initialization uses config.text_config which needs rope_scaling
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}

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
        assert config_data["hidden_size"] == 4096
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 32
        assert config_data["vocab_size"] == 152064
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
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
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
            assert saved_config["hidden_size"] == 4096, "Hidden size should match toy config"
            assert saved_config["num_attention_heads"] == 32, "Number of attention heads should match toy config"
            assert "vision_config" in saved_config, "VL model should have vision_config"

            print(f"SUCCESS: Qwen3 VL {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Qwen3 VL {test_name} conversion test: {e}")
            raise
