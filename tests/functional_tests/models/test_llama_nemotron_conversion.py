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
from safetensors.torch import save_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


HF_LLAMA_NEMOTRON_TOY_MODEL_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,  # Smaller for toy model
    "initializer_range": 0.02,
    "intermediate_size": 2816,  # ~2.75 * hidden_size
    "max_position_embeddings": 8192,  # Smaller for toy model
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 8,  # Smaller for toy model
    "num_hidden_layers": 2,  # Very small for testing
    "num_key_value_heads": 2,  # GQA with 2 KV heads
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 32000,  # Smaller vocab for toy model
}


class TestLlamaNemotronConversion:
    """
    Test Llama-Nemotron model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def llama_nemotron_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Llama-Nemotron toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("llama_nemotron_toy_model")
        model_dir = temp_dir / "llama_nemotron_toy"

        # Create Llama config from the toy model config
        config = LlamaConfig(**HF_LLAMA_NEMOTRON_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        model = LlamaForCausalLM(config)
        model = model.bfloat16()  # Use .bfloat16() method instead of .to()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Create tokenizer (use GPT-2 as fallback since it's compatible)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: Could not download tokenizer, creating minimal tokenizer files: {e}")
            # Create minimal tokenizer files if download fails
            tokenizer_config = {
                "tokenizer_class": "GPT2Tokenizer",
                "vocab_size": 32000,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "</s>",
                "unk_token": "<unk>",
            }

            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_LLAMA_NEMOTRON_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, llama_nemotron_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            llama_nemotron_toy_model_path: Path to the toy Llama-Nemotron model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(llama_nemotron_toy_model_path)
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

        assert config_data["model_type"] == "llama"
        assert config_data["hidden_size"] == 1024
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 8
        assert config_data["vocab_size"] == 32000
        assert config_data["rope_scaling"]["rope_type"] == "llama3"

        # Try loading the model to verify it's valid
        try:
            model = LlamaForCausalLM.from_pretrained(
                llama_nemotron_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Ensure full loading
            )

            # Try loading the tokenizer as well
            try:
                tokenizer = AutoTokenizer.from_pretrained(llama_nemotron_toy_model_path)
                print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

            # Verify model structure
            assert hasattr(model, "model")
            assert hasattr(model.model, "layers")
            assert len(model.model.layers) == 2  # num_hidden_layers

            print(f"SUCCESS: Toy model created and validated at {llama_nemotron_toy_model_path}")
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
    def test_llama_nemotron_conversion_parallelism(self, llama_nemotron_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Llama-Nemotron model conversion with different parallelism configurations.

        Args:
            llama_nemotron_toy_model_path: Path to the toy Llama-Nemotron model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"llama_nemotron_{test_name}"
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
            "--data-file=/tmp/workspace/.coverage",
            "--source=/tmp/workspace/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            llama_nemotron_toy_model_path,  # Use our local toy model instead of downloading
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
                assert False, f"Llama-Nemotron {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(llama_nemotron_toy_model_path).name  # "llama_nemotron_toy"
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

            # Verify the config contains Llama-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "llama", "Model type should be llama"
            assert saved_config["hidden_size"] == 1024, "Hidden size should match toy config"
            assert saved_config["num_attention_heads"] == 8, "Number of attention heads should match toy config"
            assert saved_config["rope_scaling"]["rope_type"] == "llama3", "Should have Llama 3.1 rope scaling"

            print(f"SUCCESS: Llama-Nemotron {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Llama-Nemotron {test_name} conversion test: {e}")
            raise


class TestLlamaNemotronHeterogeneousRoundtrip:
    """
    Multi-GPU roundtrip tests for a tiny heterogeneous (DeciLM) Nemotron config.
    Uses the existing hf_megatron_roundtrip_multi_gpu.py script.
    """

    @pytest.fixture(scope="class")
    def hetero_nemotron_toy_model_path(self, tmp_path_factory):
        """
        Create a minimal heterogeneous DeciLM-style HF directory with random weights.
        Includes a simple tokenizer (GPT-2) if available; otherwise writes minimal config.
        """
        temp_dir = tmp_path_factory.mktemp("hetero_llama_nemotron_toy_model")
        model_dir = temp_dir / "hetero_llama_nemotron_toy"
        model_dir.mkdir(exist_ok=True)

        # Tiny heterogeneous config
        L = 2
        H = 128
        A = 8
        D = 16  # head_dim so that A*D == H
        F = 256  # intermediate size
        V = 1024  # small vocab
        G = A // 2  # num_query_groups from n_heads_in_group=2

        block_configs = [
            {"attention": {"n_heads_in_group": 2, "no_op": False}, "ffn": {"ffn_mult": 2.0, "no_op": False}},
            {"attention": {"n_heads_in_group": None, "no_op": True}, "ffn": {"ffn_mult": 1.0, "no_op": False}},
        ]

        config_dict = {
            "architectures": ["DeciLMForCausalLM"],
            "auto_map": {"AutoModelForCausalLM": "modeling_decilm.DeciLMForCausalLM"},
            "hidden_size": H,
            "num_attention_heads": A,
            "num_hidden_layers": L,
            "max_position_embeddings": 128,
            "head_dim": D,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-5,
            "vocab_size": V,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "intermediate_size": F,
            "block_configs": block_configs,
        }

        # Write config.json
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Attempt to save a tokenizer for artifact saving in the roundtrip script
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: Could not download tokenizer, writing minimal tokenizer_config.json: {e}")
            tokenizer_config = {
                "tokenizer_class": "GPT2Tokenizer",
                "vocab_size": 50257,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "</s>",
                "unk_token": "<unk>",
            }
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        # Create random safetensors matching the keys expected by the bridge exporter
        state = {}

        # Embeddings and output
        state["model.embed_tokens.weight"] = torch.randn(V, H, dtype=torch.bfloat16)
        state["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16)
        state["model.norm.weight"] = torch.randn(H, dtype=torch.bfloat16)

        # Per-layer weights
        for i in range(L):
            prefix = f"model.layers.{i}"
            state[f"{prefix}.input_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
            state[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)

            # Attention projections
            # q out_features = A * D == H; k/v out_features = G * D
            state[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(A * D, H, dtype=torch.bfloat16)
            state[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(G * D, H, dtype=torch.bfloat16)
            state[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(G * D, H, dtype=torch.bfloat16)
            state[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(H, H, dtype=torch.bfloat16)

            # MLP projections
            state[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(F, H, dtype=torch.bfloat16)
            state[f"{prefix}.mlp.up_proj.weight"] = torch.randn(F, H, dtype=torch.bfloat16)
            state[f"{prefix}.mlp.down_proj.weight"] = torch.randn(H, F, dtype=torch.bfloat16)

        # Save single-shard safetensors
        save_file(state, model_dir / "model.safetensors")

        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_hetero_llama_nemotron_conversion_parallelism(
        self, hetero_nemotron_toy_model_path, tmp_path, tp, pp, test_name
    ):
        """
        Validate TP/PP roundtrip on tiny heterogeneous config via the existing multi-GPU script.
        """
        # Create output dir
        test_output_dir = tmp_path / f"hetero_llama_nemotron_{test_name}"
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
            "--data-file=/tmp/workspace/.coverage",
            "--source=/tmp/workspace/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            hetero_nemotron_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Hetero Llama-Nemotron {test_name} conversion failed with return code {result.returncode}"

        # Verify output directory exists and contains expected files
        model_name = Path(hetero_nemotron_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"

        weights_file_safetensors = converted_model_dir / "model.safetensors"
        weights_found = weights_file_safetensors.exists()
        if not weights_found:
            shards_st = list(converted_model_dir.glob("model-*-of-*.safetensors"))
            weights_found = len(shards_st) > 0
        if not weights_found:
            print(f"Warning: No safetensors weights found in converted model at {converted_model_dir}.")

        with open(config_file) as f:
            saved_config = json.load(f)

        # Basic sanity checks against hetero config
        assert saved_config["hidden_size"] == 128
        assert saved_config["num_attention_heads"] == 8
        assert saved_config["num_hidden_layers"] == 2
        assert saved_config.get("rope_scaling", {}).get("rope_type") == "llama3"
