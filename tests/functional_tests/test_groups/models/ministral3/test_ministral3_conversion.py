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


# Ministral 3 toy model configuration based on typical Ministral 3 structure
# This is a minimized version for testing purposes
HF_MINISTRAL3_TOY_MODEL_CONFIG = {
    "architectures": ["Mistral3ForConditionalGeneration"],
    "model_type": "mistral3",
    "torch_dtype": "bfloat16",
    "transformers_version": "5.0.0",
    "image_token_index": 10,
    "text_config": {
        "model_type": "ministral3",
        "hidden_size": 512,
        "head_dim": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 32768,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "rope_theta": 1000000,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "llama_4_scaling_beta": 0.0,
        },
    },
    "vision_config": {
        "model_type": "pixtral",
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "image_size": 448,
        "patch_size": 14,
        "num_channels": 3,
    },
    "spatial_merge_size": 2,
    "vision_feature_layer": -1,
}


class TestMinistral3Conversion:
    """
    Test Ministral 3 model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def ministral3_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Ministral 3 toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        try:
            from transformers import Ministral3ForCausalLM, Mistral3ForConditionalGeneration  # noqa: F401
            from transformers.models.mistral3.configuration_mistral3 import Mistral3Config  # noqa: F401
        except ImportError:
            pytest.skip("Ministral 3 not available in transformers")

        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("ministral3_toy_model")
        model_dir = temp_dir / "ministral3_toy"

        # Create config from the toy model config
        config_dict = HF_MINISTRAL3_TOY_MODEL_CONFIG.copy()

        # Create config object
        config = Mistral3Config(**config_dict)
        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        model = Mistral3ForConditionalGeneration(config)
        model = model.bfloat16()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Create minimal tokenizer files
        tokenizer_config = {
            "tokenizer_class": "LlamaTokenizer",
            "vocab_size": 32768,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Save config.json explicitly to ensure compatibility
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, ministral3_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            ministral3_toy_model_path: Path to the toy Ministral 3 model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(ministral3_toy_model_path)
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

        assert config_data["model_type"] == "mistral3"
        assert "text_config" in config_data
        assert "vision_config" in config_data
        assert config_data["text_config"]["hidden_size"] == 512
        assert config_data["text_config"]["num_hidden_layers"] == 4
        assert config_data["text_config"]["num_attention_heads"] == 8
        assert config_data["vision_config"]["hidden_size"] == 256

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_ministral3_conversion_parallelism(self, ministral3_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Ministral 3 model conversion with different parallelism configurations.

        Args:
            ministral3_toy_model_path: Path to the toy Ministral 3 model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"ministral3_{test_name}"
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
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            ministral3_toy_model_path,  # Use our local toy model instead of downloading
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
        print(cmd)

        # Check that the conversion completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Ministral 3 {test_name} conversion failed with return code {result.returncode}"

        # Verify that the converted model was saved
        # The output directory should be named after the last part of the model path
        model_name = Path(ministral3_toy_model_path).name  # "ministral3_toy"
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

        # Verify the config contains Ministral 3-specific parameters
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "mistral3", "Model type should be mistral3"
        assert "text_config" in saved_config, "VL model should have text_config"
        assert "vision_config" in saved_config, "VL model should have vision_config"
        assert saved_config["text_config"]["hidden_size"] == 512, "Hidden size should match toy config"
        assert saved_config["text_config"]["num_attention_heads"] == 8, (
            "Number of attention heads should match toy config"
        )
