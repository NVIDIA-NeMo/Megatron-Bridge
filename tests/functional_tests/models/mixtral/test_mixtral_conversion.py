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


HF_MIXTRAL_TOY_MODEL_CONFIG = {
    "architectures": ["MixtralForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 1024,  # Smaller than real model for faster testing
    "initializer_range": 0.02,
    "intermediate_size": 2048,  # Reduced for testing
    "max_position_embeddings": 4096,
    "model_type": "mixtral",
    "num_attention_heads": 16,
    "num_hidden_layers": 2,  # Much smaller for testing
    "num_key_value_heads": 8,
    "num_local_experts": 8,  # Mixtral uses 8 experts
    "num_experts_per_tok": 2,  # Mixtral uses top-2 routing
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.01,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.0",
    "use_cache": True,
    "vocab_size": 32000,
}


@pytest.mark.integration
class TestMixtralConversion:
    """
    Test Mixtral model conversion from local HuggingFace model with different parallelism configurations.

    Requires: 2 GPUs for the conversion test (nproc_per_node=2).
    """

    @pytest.fixture(scope="class")
    def mixtral_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Mixtral toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("mixtral_toy_model")
        model_dir = temp_dir / "mixtral_toy"

        # Create Mixtral config from the toy model config
        from transformers import AutoConfig, MixtralForCausalLM

        config = AutoConfig.for_model(**HF_MIXTRAL_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        model = MixtralForCausalLM(config)
        model = model.bfloat16()  # Use .bfloat16() method instead of .to()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Download and save tokenizer from a reference model
        # Try to use Mixtral tokenizer if available, otherwise fall back to GPT2
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        except (OSError, ValueError):
            # Fallback to a generic tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.save_pretrained(model_dir)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_MIXTRAL_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, mixtral_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            mixtral_toy_model_path: Path to the toy Mixtral model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(mixtral_toy_model_path)
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

        assert config_data["model_type"] == "mixtral"
        assert config_data["hidden_size"] == 1024
        assert config_data["intermediate_size"] == 2048
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 16
        assert config_data["num_key_value_heads"] == 8
        assert config_data["vocab_size"] == 32000
        # Check Mixtral-specific MoE parameters
        assert config_data["num_local_experts"] == 8
        assert config_data["num_experts_per_tok"] == 2
        assert config_data["router_aux_loss_coef"] == 0.01

        # Try loading the model to verify it's valid
        from transformers import MixtralForCausalLM

        model = MixtralForCausalLM.from_pretrained(
            mixtral_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,  # Ensure full loading
        )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(mixtral_toy_model_path)
        print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")

        # Verify model structure
        assert hasattr(model, "model")
        assert hasattr(model.model, "layers")
        assert len(model.model.layers) == 2  # num_hidden_layers

        print(f"SUCCESS: Toy model created and validated at {mixtral_toy_model_path}")
        print("Model weights are correctly in bfloat16 format")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_mixtral_conversion_parallelism(self, mixtral_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Mixtral model conversion with different parallelism configurations.

        Args:
            mixtral_toy_model_path: Path to the toy Mixtral model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """
        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"mixtral_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        repo_root = Path(__file__).resolve().parents[4]
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file",
            str(repo_root / ".coverage"),
            "--source",
            str(repo_root),
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            mixtral_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        try:
            subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_root,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"STDOUT: {exc.stdout}")
            print(f"STDERR: {exc.stderr}")
            pytest.fail(f"Mixtral {test_name} conversion failed with return code {exc.returncode}")

        # Verify that the converted model was saved
        # The output directory should be named after the last part of the model path
        model_name = Path(mixtral_toy_model_path).name  # "mixtral_toy"
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

        # Verify the config contains Mixtral-specific parameters
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "mixtral", "Model type should be mixtral"
        assert saved_config["hidden_size"] == 1024, "Hidden size should match toy config"
        assert saved_config["intermediate_size"] == 2048, "Intermediate size should match toy config"
        assert saved_config["num_attention_heads"] == 16, "Number of attention heads should match toy config"
        assert saved_config["num_key_value_heads"] == 8, "Number of key-value heads should match toy config"
        # Verify Mixtral-specific MoE parameters
        assert saved_config["num_local_experts"] == 8, "Number of experts should match"
        assert saved_config["num_experts_per_tok"] == 2, "Number of experts per token should match"
        assert saved_config["router_aux_loss_coef"] == 0.01, "Router aux loss coefficient should match"

        print(f"SUCCESS: Mixtral {test_name} conversion test completed successfully")
        print(f"Converted model saved at: {converted_model_dir}")
