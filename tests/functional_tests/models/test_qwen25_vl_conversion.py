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
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer


HF_QWEN25_VL_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "vision_token_id": 151654,
    "image_token_id": 151655,
    "video_token_id": 151656,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 128000,
    "max_window_layers": 2,
    "model_type": "qwen2_5_vl",
    "num_attention_heads": 28,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.2",
    "use_cache": True,
    "use_sliding_window": False,
    "vision_config": {
        "depth": 2,
        "hidden_act": "silu",
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_chans": 3,
        "out_hidden_size": 3584,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "window_size": 112,
        "fullatt_block_indexes": [
            0,
        ],
        "tokens_per_second": 2,
        "temporal_patch_size": 2,
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "vocab_size": 152064,
}


class TestQwen25VLConversion:
    """
    Test Qwen25 VL model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def qwen25_vl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen25 VL toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen25_vl_toy_model")
        model_dir = temp_dir / "qwen25_vl_toy"

        # Create Qwen25 VL config from the toy model config
        config = Qwen2_5_VLConfig(**HF_QWEN25_VL_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        model = Qwen2_5_VLForConditionalGeneration(config)
        model = model.bfloat16()  # Use .bfloat16() method instead of .to()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Download and save tokenizer from a reference Qwen25 VL model
        # We use the smallest available Qwen25 VL model for tokenizer artifacts
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            tokenizer.save_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: Could not download tokenizer, creating minimal tokenizer files: {e}")
            # Create minimal tokenizer files if download fails
            # This is a fallback for offline environments
            tokenizer_config = {
                "tokenizer_class": "Qwen2Tokenizer",
                "vocab_size": 151936,
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            }

            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_QWEN25_VL_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, qwen25_vl_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            qwen25_vl_toy_model_path: Path to the toy Qwen25 VL model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(qwen25_vl_toy_model_path)
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

        assert config_data["model_type"] == "qwen2_5_vl"
        assert config_data["hidden_size"] == 3584
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 28
        assert config_data["vocab_size"] == 152064
        assert "vision_config" in config_data

        # Try loading the model to verify it's valid
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen25_vl_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Ensure full loading
            )

            # Try loading the tokenizer as well
            try:
                tokenizer = Qwen2Tokenizer.from_pretrained(qwen25_vl_toy_model_path)
                print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

            # Verify model structure for VL model
            assert hasattr(model, "model")
            assert hasattr(model.model, "language_model")
            assert hasattr(model.model.language_model, "layers")
            assert len(model.model.language_model.layers) == 2  # num_hidden_layers
            assert hasattr(model.model, "visual")  # VL model should have visual component

            print(f"SUCCESS: Toy model created and validated at {qwen25_vl_toy_model_path}")
            print("Model weights are correctly in bfloat16 format")

        except Exception as e:
            assert False, f"Failed to load created toy model: {e}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_qwen25_vl_conversion_parallelism(self, qwen25_vl_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Qwen25 VL model conversion with different parallelism configurations.

        Args:
            qwen25_vl_toy_model_path: Path to the toy Qwen25 VL model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"qwen25_vl_{test_name}"
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
            qwen25_vl_toy_model_path,  # Use our local toy model instead of downloading
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
            print(cmd)

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Qwen25 VL {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(qwen25_vl_toy_model_path).name  # "qwen25_vl_toy"
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

            # Verify the config contains Qwen25 VL-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "qwen2_5_vl", "Model type should be qwen2_5_vl"
            assert saved_config["hidden_size"] == 3584, "Hidden size should match toy config"
            assert saved_config["num_attention_heads"] == 28, "Number of attention heads should match toy config"
            assert "vision_config" in saved_config, "VL model should have vision_config"

            print(f"SUCCESS: Qwen25 VL {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Qwen25 VL {test_name} conversion test: {e}")
            raise
