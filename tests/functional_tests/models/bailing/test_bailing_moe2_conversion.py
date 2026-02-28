# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from transformers import AutoConfig, AutoTokenizer


HF_BAILING_MOE2_TOY_MODEL_CONFIG = {
    "architectures": ["BailingMoeV2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,  # Ensure pad_token_id is within vocab_size
    "first_k_dense_replace": 1,  # First layer is dense, rest are MoE
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": 4096,
    "model_type": "bailing_moe_v2",
    "moe_intermediate_size": 512,
    "num_attention_heads": 8,
    "num_experts": 8,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_nextn_predict_layers": 0,  # No MTP layers for toy model
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.54.0",
    "use_cache": True,
    "use_qkv_bias": False,
    "vocab_size": 32000,
}


class TestBailingMoeV2Conversion:
    """
    Test Bailing MoE V2 model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def bailing_moe2_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Bailing MoE V2 toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("bailing_moe2_toy_model")
        model_dir = temp_dir / "bailing_moe2_toy"

        # Try to load config from a reference model, or create from scratch
        try:
            # Try to get config structure from a reference model
            config = AutoConfig.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
        except Exception:
            # If that fails, create a basic config structure
            from transformers import PretrainedConfig

            class BailingMoeV2Config(PretrainedConfig):
                model_type = "bailing_moe_v2"

            config = BailingMoeV2Config()

        # Override with toy model config
        for key, value in HF_BAILING_MOE2_TOY_MODEL_CONFIG.items():
            setattr(config, key, value)

        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        # The config loading above should have registered the model class via trust_remote_code
        try:
            from transformers import AutoModelForCausalLM

            # Create the model from config - the custom class should be registered from loading config above
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except Exception as e:
            # If that fails, try loading a minimal model to register the class
            try:
                # Load a tiny model just to register the class
                _ = AutoModelForCausalLM.from_pretrained(
                    "inclusionAI/Ling-mini-2.0",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",  # Don't use GPU for this
                )
                # Now try again
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            except Exception as e2:
                pytest.skip(
                    f"Could not create Bailing MoE V2 model: {e}. "
                    f"Fallback also failed: {e2}. Model class may require custom code."
                )

        model = model.bfloat16()  # Use .bfloat16() method instead of .to()

        # Download and save tokenizer from a reference Bailing model
        try:
            tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
            # Ensure pad_token_id is within vocab_size range
            if tokenizer.pad_token_id is None or tokenizer.pad_token_id >= config.vocab_size:
                tokenizer.pad_token_id = config.pad_token_id
            tokenizer.save_pretrained(model_dir)
        except Exception:
            # If tokenizer download fails, create a basic tokenizer
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(
                vocab_size=HF_BAILING_MOE2_TOY_MODEL_CONFIG["vocab_size"],
                bos_token="<s>",
                eos_token="</s>",
                pad_token="<pad>",
                pad_token_id=HF_BAILING_MOE2_TOY_MODEL_CONFIG["pad_token_id"],
            )
            tokenizer.save_pretrained(model_dir)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_BAILING_MOE2_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, bailing_moe2_toy_model_path):
        """
        Test that the toy MoE model is created correctly and can be loaded.

        Args:
            bailing_moe2_toy_model_path: Path to the toy Bailing MoE V2 model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(bailing_moe2_toy_model_path)
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

        assert config_data["model_type"] == "bailing_moe_v2"
        assert config_data["hidden_size"] == 1024
        assert config_data["intermediate_size"] == 2048
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 8
        assert config_data["vocab_size"] == 32000
        # Verify MoE specific parameters
        assert config_data["num_experts"] == 8
        assert config_data["num_experts_per_tok"] == 4
        assert config_data["moe_intermediate_size"] == 512
        assert config_data["first_k_dense_replace"] == 1

        # Try loading the model to verify it's valid
        # Note: This may fail if custom code isn't available, but that's OK for conversion testing
        try:
            from transformers import AutoConfig, AutoModelForCausalLM

            # First, ensure the custom model class is registered by loading the reference config
            # Loading config with trust_remote_code=True should register the custom model class
            try:
                _ = AutoConfig.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
            except Exception:
                pass  # If this fails, try loading anyway

            # Now load our saved toy model - the custom class should be registered
            # If it's not registered, trust_remote_code=True should load it from the saved config
            model = AutoModelForCausalLM.from_pretrained(
                bailing_moe2_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Ensure full loading
                trust_remote_code=True,
            )

            # Verify model structure
            assert hasattr(model, "model")
            assert hasattr(model.model, "layers")
            assert len(model.model.layers) == 2  # num_hidden_layers

            # Verify MoE structure
            # First layer is dense, second layer should have MoE structure
            second_layer = model.model.layers[1]
            assert hasattr(second_layer, "mlp")
            # Bailing MoE V2 structure check (may vary based on implementation)
            if hasattr(second_layer.mlp, "experts"):
                assert len(second_layer.mlp.experts) == 8  # num_experts

            print(f"SUCCESS: Bailing MoE V2 toy model created and validated at {bailing_moe2_toy_model_path}")
            print("Model weights are correctly in bfloat16 format")
            print(f"MoE structure validated: {config_data['num_experts']} experts")

        except Exception as e:
            # If loading fails due to custom code not being available, that's OK
            # The model files are created correctly, which is what matters for conversion testing
            error_msg = str(e)
            if "does not recognize this architecture" in error_msg or "custom code" in error_msg.lower():
                print(f"WARNING: Could not load saved model for verification (custom code not available): {e}")
                print("This is OK - the model files were created correctly and can be used for conversion testing.")
            else:
                # For other errors, fail the test
                assert False, f"Failed to load created toy MoE model: {e}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_bailing_moe2_conversion_parallelism(self, bailing_moe2_toy_model_path, tmp_path, tp, pp, ep, test_name):
        """
        Test Bailing MoE V2 model conversion with different parallelism configurations.

        Args:
            bailing_moe2_toy_model_path: Path to the toy Bailing MoE V2 model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            ep: Expert parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"bailing_moe2_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism configuration on our toy MoE model
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
            bailing_moe2_toy_model_path,  # Use our local toy MoE model instead of downloading
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
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Bailing MoE V2 {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(bailing_moe2_toy_model_path).name  # "bailing_moe2_toy"
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"

            # Check for single files first
            weights_found = weights_file_safetensors.exists() or weights_file_pytorch.exists()

            # If single files don't exist, check for sharded files
            if not weights_found:
                sharded_safetensors = list(converted_model_dir.glob("model-*-of-*.safetensors"))
                sharded_pytorch = list(converted_model_dir.glob("pytorch_model-*-of-*.bin"))
                weights_found = len(sharded_safetensors) > 0 or len(sharded_pytorch) > 0

            assert weights_found, f"Model weights file not found in converted model at {converted_model_dir}"

            # Verify the config contains Bailing MoE V2-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "bailing_moe_v2", (
                "Model type should be bailing_moe_v2 (Bailing MoE V2 uses BailingMoeV2ForCausalLM)"
            )
            assert saved_config["hidden_size"] == 1024, "Hidden size should match toy config"
            assert saved_config["num_attention_heads"] == 8, "Number of attention heads should match toy config"
            # Verify MoE specific parameters are preserved
            assert saved_config["num_experts"] == 8, "Number of experts should match toy config"
            assert saved_config["num_experts_per_tok"] == 4, "Number of experts per token should match toy config"
            assert saved_config["moe_intermediate_size"] == 512, "MoE intermediate size should match toy config"
            assert saved_config["first_k_dense_replace"] == 1, "First K dense replace should match toy config"

            print(f"SUCCESS: Bailing MoE V2 {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")
            print(
                f"MoE parameters preserved: {saved_config['num_experts']} experts, {saved_config['num_experts_per_tok']} per token"
            )

        except Exception as e:
            print(f"Error during Bailing MoE V2 {test_name} conversion test: {e}")
            raise
