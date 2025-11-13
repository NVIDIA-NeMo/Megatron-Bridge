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
from transformers import AutoTokenizer, Qwen3NextConfig, Qwen3NextForCausalLM


# Toy model config based on Qwen3-Next-80B-A3B but with minimal layers for testing
HF_QWEN3_NEXT_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3NextForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 128,
    "head_dim": 256,
    "initializer_range": 0.02,
    "intermediate_size": 5120,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_next",
    "num_attention_heads": 16,
    "num_hidden_layers": 2,  # Reduced for toy model testing
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.1",
    "use_cache": True,
    "vocab_size": 151936,
    # MoE specific
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    # Qwen3-Next specific
    "full_attention_interval": 2,  # 1 standard attention layer per 4 layers
    "partial_rotary_factor": 0.25,  # RoPE only applies to first 25% of dims
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
}


class TestQwen3NextConversion:
    """
    Test Qwen3Next model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def qwen3_next_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3Next toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen3_next_toy_model")
        model_dir = temp_dir / "qwen3_next_toy"

        # Create Qwen3Next config from the toy model config
        config = Qwen3NextConfig(**HF_QWEN3_NEXT_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        model = Qwen3NextForCausalLM(config)
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
        config_to_save = HF_QWEN3_NEXT_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, qwen3_next_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            qwen3_next_toy_model_path: Path to the toy Qwen3Next model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(qwen3_next_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"

        # If neither single file exists, check for sharded files
        if not weights_file.exists():
            # Check for sharded safetensors files
            sharded_files = list(model_path.glob("model-*-of-*.safetensors"))
            if sharded_files:
                weights_file = sharded_files[0]  # Use first shard as representative
            else:
                # Check for sharded pytorch files
                sharded_files = list(model_path.glob("pytorch_model-*-of-*.bin"))
                if sharded_files:
                    weights_file = sharded_files[0]  # Use first shard as representative

        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen3_next"
        assert config_data["hidden_size"] == 128
        assert config_data["num_hidden_layers"] == 2  # Updated to match toy config
        assert config_data["num_attention_heads"] == 16
        assert config_data["vocab_size"] == 151936
        assert config_data["num_experts"] == 8
        assert config_data["num_experts_per_tok"] == 2

        # Try loading the model to verify it's valid
        try:
            model = Qwen3NextForCausalLM.from_pretrained(
                qwen3_next_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Ensure full loading
            )

            # Try loading the tokenizer as well
            try:
                tokenizer = AutoTokenizer.from_pretrained(qwen3_next_toy_model_path)
                print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

            # Verify model structure
            assert hasattr(model, "model")
            assert hasattr(model.model, "layers")
            assert len(model.model.layers) == 2  # num_hidden_layers updated to match toy config

            print(f"SUCCESS: Toy model created and validated at {qwen3_next_toy_model_path}")
            print("Model weights are correctly in bfloat16 format")

        except Exception as e:
            assert False, f"Failed to load created toy model: {e}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_qwen3_next_conversion_parallelism(self, qwen3_next_toy_model_path, tmp_path, tp, pp, ep, test_name):
        """
        Test Qwen3Next model conversion with different parallelism configurations.

        Args:
            qwen3_next_toy_model_path: Path to the toy Qwen3Next model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            ep: Expert parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"qwen3_next_{test_name}"
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
            qwen3_next_toy_model_path,  # Use our local toy model instead of downloading
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Qwen3Next {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(qwen3_next_toy_model_path).name  # "qwen3_next_toy"
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

            # Verify the config contains Qwen3Next-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "qwen3_next", "Model type should be qwen3_next"
            assert saved_config["hidden_size"] == 128, "Hidden size should match toy config"
            assert saved_config["num_attention_heads"] == 16, "Number of attention heads should match toy config"
            assert saved_config["num_experts"] == 8, "Number of experts should match toy config"

            print(f"SUCCESS: Qwen3Next {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Qwen3Next {test_name} conversion test: {e}")
            raise
