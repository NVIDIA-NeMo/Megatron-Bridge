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


# GLM-4.5V toy model configuration based on GLM-4.5 Air structure
# This is a minimized version for testing purposes
HF_GLM_45V_TOY_MODEL_CONFIG = {
    "architectures": ["Glm4vMoeForConditionalGeneration"],
    "model_type": "glm4v_moe",
    "image_start_token_id": 151339,
    "image_end_token_id": 151340,
    "video_start_token_id": 151341,
    "video_end_token_id": 151342,
    "image_token_id": 151363,
    "video_token_id": 151364,
    "tie_word_embeddings": False,
    "transformers_version": "4.57.1",
    "text_config": {
        "model_type": "glm4v_moe_text",
        "pad_token_id": 151329,
        "vocab_size": 151552,
        "eos_token_id": [151329, 151336, 151338],
        "head_dim": 128,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 65536,
        "moe_intermediate_size": 128,
        "n_group": 1,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 16,
        "num_experts_per_tok": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 8,
        "partial_rotary_factor": 0.5,
        "rms_norm_eps": 1e-05,
        "dtype": "bfloat16",
        "rope_scaling": {"rope_type": "default", "mrope_section": [8, 12, 12]},
        "rope_theta": 10000.0,
        "routed_scaling_factor": 1.0,
        "topk_group": 1,
        "use_cache": True,
        "use_qk_norm": False,
    },
    "vision_config": {
        "model_type": "glm4v_moe",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "depth": 2,
        "hidden_act": "silu",
        "hidden_size": 256,
        "image_size": 336,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 10944,
        "num_heads": 2,
        "out_hidden_size": 128,
        "patch_size": 14,
        "rms_norm_eps": 1e-05,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}


class TestGLM45VConversion:
    """
    Test GLM-4.5V model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def glm_45v_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace GLM-4.5V toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        try:
            from transformers import Glm4vMoeForConditionalGeneration
            from transformers.models.glm4v.configuration_glm4v import Glm4vConfig
        except ImportError:
            pytest.skip("GLM-4.5V not available in transformers")

        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("glm_45v_toy_model")
        model_dir = temp_dir / "glm_45v_toy"

        # Create config from the toy model config
        config_dict = HF_GLM_45V_TOY_MODEL_CONFIG.copy()

        # Create config object
        config = Glm4vConfig(**config_dict)
        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        model = Glm4vMoeForConditionalGeneration(config)
        model = model.bfloat16()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Download and save tokenizer from a reference Qwen25 VL model
        # We use the smallest available Qwen25 VL model for tokenizer artifacts
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5V")
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

        # Save config.json explicitly to ensure compatibility
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, glm_45v_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            glm_45v_toy_model_path: Path to the toy GLM-4.5V model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(glm_45v_toy_model_path)
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

        assert config_data["model_type"] == "glm4v_moe"
        assert "text_config" in config_data
        assert "vision_config" in config_data
        assert config_data["text_config"]["hidden_size"] == 512
        assert config_data["text_config"]["num_hidden_layers"] == 4
        assert config_data["text_config"]["num_attention_heads"] == 16
        assert config_data["vision_config"]["hidden_size"] == 256

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_glm_45v_conversion_parallelism(self, glm_45v_toy_model_path, tmp_path, tp, pp, ep, test_name):
        """
        Test GLM-4.5V model conversion with different parallelism configurations.

        Args:
            glm_45v_toy_model_path: Path to the toy GLM-4.5V model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            ep: Expert parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"glm_45v_{test_name}"
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
            glm_45v_toy_model_path,  # Use our local toy model instead of downloading
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )
        print(cmd)

        # Check that the conversion completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"GLM-4.5V {test_name} conversion failed with return code {result.returncode}"

        # Verify that the converted model was saved
        # The output directory should be named after the last part of the model path
        model_name = Path(glm_45v_toy_model_path).name  # "glm_45v_toy"
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

        # Verify the config contains GLM-4.5V-specific parameters
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "glm4v_moe", "Model type should be glm4v_moe"
        assert "text_config" in saved_config, "VL model should have text_config"
        assert "vision_config" in saved_config, "VL model should have vision_config"
        assert saved_config["text_config"]["hidden_size"] == 512, "Hidden size should match toy config"
        assert saved_config["text_config"]["num_attention_heads"] == 16, (
            "Number of attention heads should match toy config"
        )

    @pytest.mark.run_only_on("GPU")
    def test_glm_45v_conversion_tp_ep_combined(self, glm_45v_toy_model_path, tmp_path):
        """
        Test GLM-4.5V model conversion with combined TP and EP parallelism.

        Args:
            glm_45v_toy_model_path: Path to the toy GLM-4.5V model (from fixture)
            tmp_path: Pytest temporary path fixture
        """
        # Skip if not enough GPUs for combined parallelism
        # This test requires 4 GPUs (TP=2 * EP=2)

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / "glm_45v_TP_EP"
        test_output_dir.mkdir(exist_ok=True)

        # Run with TP=2, EP=2 (requires 4 GPUs)
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=4",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            glm_45v_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            "2",
            "--pp",
            "1",
            "--ep",
            "2",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )
        print(cmd)

        # Check that the conversion completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            # This test may be skipped if not enough GPUs are available
            if "CUDA" in result.stderr or "GPU" in result.stderr:
                pytest.skip("Not enough GPUs for TP=2, EP=2 test")
            assert False, f"GLM-4.5V TP+EP conversion failed with return code {result.returncode}"

        # Verify that the converted model was saved
        model_name = Path(glm_45v_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"
