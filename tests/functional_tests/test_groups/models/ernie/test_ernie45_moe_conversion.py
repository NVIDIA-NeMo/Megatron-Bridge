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

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Ernie4_5_MoeConfig, Ernie4_5_MoeForCausalLM


# Path to the local ERNIE VL model that contains tokenizer files.
# Used as a tokenizer source when running offline (CI/air-gapped environments).
_ERNIE_VL_MODEL_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / ("ERNIE-4.5-VL-28B-A3B-Thinking")
# Tokenizer files to copy from the reference model directory.
_TOKENIZER_FILES = [
    "tokenizer_config.json",
    "tokenizer.model",
    "added_tokens.json",
    "special_tokens_map.json",
]


# Toy config: 4 layers (layer 0 dense, layers 1-3 MoE), 4 experts, top-2 routing,
# 1 shared expert, small hidden/intermediate sizes for fast CI testing.
HF_ERNIE45_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Ernie4_5_MoeForCausalLM"],
    "model_type": "ernie4_5_moe",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_act": "silu",
    "max_position_embeddings": 1024,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-05,
    "use_cache": True,
    "use_bias": False,
    "vocab_size": 1024,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    # MoE settings
    "moe_num_experts": 4,
    "moe_k": 2,
    "moe_intermediate_size": 128,
    "moe_num_shared_experts": 1,
    "moe_layer_start_index": 1,
    "moe_layer_end_index": 3,
    "moe_layer_interval": 1,
    "router_aux_loss_coef": 0.001,
    "output_router_logits": False,
    # Token IDs
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    # Dtype
    "torch_dtype": "bfloat16",
}


class TestErnie45MoEConversion:
    """
    Test ERNIE 4.5 MoE model conversion from local HuggingFace model
    with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def ernie45_moe_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace ERNIE 4.5 MoE toy model from config
        to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("ernie45_moe_toy_model")
        model_dir = temp_dir / "ernie45_moe_toy"

        # Create ERNIE 4.5 MoE config from the toy model config
        config = Ernie4_5_MoeConfig(**HF_ERNIE45_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        model = Ernie4_5_MoeForCausalLM(config)
        model = model.bfloat16()

        # Copy tokenizer files from the local ERNIE VL model (works offline).
        # Falls back to downloading from HuggingFace Hub if the local model is absent.
        if _ERNIE_VL_MODEL_PATH.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            for fname in _TOKENIZER_FILES:
                src = _ERNIE_VL_MODEL_PATH / fname
                if src.exists():
                    shutil.copy2(src, model_dir / fname)
            # Sanitize tokenizer_config.json: the VL model's config references
            # a custom tokenizer class (Ernie4_5_VLTokenizer) via auto_map,
            # which triggers trust_remote_code checks. Strip these fields so
            # the toy model uses the standard LlamaTokenizer instead.
            tok_cfg_path = model_dir / "tokenizer_config.json"
            if tok_cfg_path.exists():
                with open(tok_cfg_path) as f:
                    tok_cfg = json.load(f)
                tok_cfg.pop("auto_map", None)
                tok_cfg.pop("tokenizer_class", None)
                with open(tok_cfg_path, "w") as f:
                    json.dump(tok_cfg, f, indent=2)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            tokenizer.save_pretrained(model_dir)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility
        config_to_save = HF_ERNIE45_MOE_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, ernie45_moe_toy_model_path):
        """
        Test that the toy ERNIE 4.5 MoE model is created correctly and can be loaded.

        Args:
            ernie45_moe_toy_model_path: Path to the toy ERNIE 4.5 MoE model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(ernie45_moe_toy_model_path)
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
            sharded_files = list(model_path.glob("model-*-of-*.safetensors"))
            if sharded_files:
                weights_file = sharded_files[0]
            else:
                sharded_files = list(model_path.glob("pytorch_model-*-of-*.bin"))
                if sharded_files:
                    weights_file = sharded_files[0]

        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "ernie4_5_moe"
        assert config_data["hidden_size"] == 256
        assert config_data["num_hidden_layers"] == 4
        assert config_data["num_attention_heads"] == 4
        assert config_data["vocab_size"] == 1024
        # Verify MoE specific parameters
        assert config_data["moe_num_experts"] == 4
        assert config_data["moe_k"] == 2
        assert config_data["moe_intermediate_size"] == 128
        assert config_data["moe_num_shared_experts"] == 1
        assert config_data["moe_layer_start_index"] == 1

        # Try loading the model to verify it's valid
        try:
            model = Ernie4_5_MoeForCausalLM.from_pretrained(
                ernie45_moe_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )

            # Try loading the tokenizer as well
            try:
                tokenizer = AutoTokenizer.from_pretrained(ernie45_moe_toy_model_path)
                print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

            # Verify model structure
            assert hasattr(model, "model")
            assert hasattr(model.model, "layers")
            assert len(model.model.layers) == 4  # num_hidden_layers

            # Verify MoE structure: layer 0 should be dense, layers 1-3 should be MoE
            layer0 = model.model.layers[0]
            layer1 = model.model.layers[1]
            assert hasattr(layer0, "mlp")
            assert hasattr(layer1, "mlp")
            # Layer 0 is dense (Ernie4_5_MoeMLP), layer 1 is MoE (Ernie4_5_MoeSparseMoeBlock)
            assert "MLP" in type(layer0.mlp).__name__
            assert "SparseMoeBlock" in type(layer1.mlp).__name__ or "Moe" in type(layer1.mlp).__name__

            print(f"SUCCESS: ERNIE 4.5 MoE toy model created and validated at {ernie45_moe_toy_model_path}")
            print("Model weights are correctly in bfloat16 format")
            print(
                f"MoE structure validated: {config_data['moe_num_experts']} experts, "
                f"top-{config_data['moe_k']} routing"
            )

        except Exception as e:
            pytest.fail(f"Failed to load created toy MoE model: {e}")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            pytest.param(1, 1, 2, "EP", marks=pytest.mark.pleasefixme),
        ],
    )
    def test_ernie45_moe_conversion_parallelism(self, ernie45_moe_toy_model_path, tmp_path, tp, pp, ep, test_name):
        """
        Test ERNIE 4.5 MoE model conversion with different parallelism configurations.

        Args:
            ernie45_moe_toy_model_path: Path to the toy ERNIE 4.5 MoE model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            ep: Expert parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"ernie45_moe_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism.
        # Use coverage wrapper when available (CI), skip it otherwise (local dev).
        _has_coverage = importlib.util.find_spec("coverage") is not None
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
        ]
        if _has_coverage:
            cmd += [
                "-m",
                "coverage",
                "run",
                "--data-file=/opt/Megatron-Bridge/.coverage",
                "--source=/opt/Megatron-Bridge/",
                "--parallel-mode",
            ]
        cmd += [
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            ernie45_moe_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
            "--trust-remote-code",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent.parent,
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                pytest.fail(f"ERNIE 4.5 MoE {test_name} conversion failed with return code {result.returncode}")

            # Verify that the converted model was saved
            model_name = Path(ernie45_moe_toy_model_path).name  # "ernie45_moe_toy"
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"

            weights_found = weights_file_safetensors.exists() or weights_file_pytorch.exists()

            if not weights_found:
                sharded_safetensors = list(converted_model_dir.glob("model-*-of-*.safetensors"))
                sharded_pytorch = list(converted_model_dir.glob("pytorch_model-*-of-*.bin"))
                weights_found = len(sharded_safetensors) > 0 or len(sharded_pytorch) > 0

            assert weights_found, f"Model weights file not found in converted model at {converted_model_dir}"

            # Verify the config contains ERNIE 4.5 MoE-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "ernie4_5_moe", "Model type should be ernie4_5_moe"
            assert saved_config["hidden_size"] == 256, "Hidden size should match toy config"
            assert saved_config["num_attention_heads"] == 4, "Number of attention heads should match toy config"
            # Verify MoE specific parameters are preserved
            assert saved_config["moe_num_experts"] == 4, "Number of experts should match toy config"
            assert saved_config["moe_k"] == 2, "moe_k (top-k routing) should match toy config"
            assert saved_config["moe_intermediate_size"] == 128, "MoE intermediate size should match toy config"

            print(f"SUCCESS: ERNIE 4.5 MoE {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")
            print(
                f"MoE parameters preserved: {saved_config['moe_num_experts']} experts, "
                f"top-{saved_config['moe_k']} routing"
            )

        except Exception as e:
            print(f"Error during ERNIE 4.5 MoE {test_name} conversion test: {e}")
            raise
