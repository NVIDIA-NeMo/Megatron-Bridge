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
Functional tests for Qwen3-ASR model conversion.

Run toy model creation test:
    python -m pytest tests/functional_tests/models/qwen3_asr/test_qwen3_asr_conversion.py::TestQwen3ASRConversion::test_toy_model_creation

Run full conversion roundtrip (requires 2 GPUs):
    bash tests/functional_tests/L0_Launch_models_qwen3_asr.sh
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

# Import custom HF classes to trigger auto-registration with transformers
from megatron.bridge.models.qwen3_asr.hf_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
)


# Tiny model config optimized for fast testing
# Key constraints:
# - audio_config.output_dim must match text_config.hidden_size (projection dimension)
# - num_kv_heads=2 for TP=2 compatibility
# - head_dim = hidden_size / num_attention_heads (256/4 = 64)
HF_QWEN3_ASR_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3ASRForConditionalGeneration"],
    "model_type": "qwen3_asr",
    "thinker_config": {
        "audio_config": {
            "num_mel_bins": 128,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 128,
            "d_model": 64,
            "output_dim": 256,
            "max_source_positions": 1500,
            "n_window": 100,
            "n_window_infer": 400,
            "conv_chunksize": 500,
            "downsample_hidden_size": 480,
        },
        "text_config": {
            "vocab_size": 2048,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000.0,
            "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20]},
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        },
        "audio_token_id": 151646,
        "audio_start_token_id": 151647,
    },
}


class TestQwen3ASRConversion:
    """
    Test Qwen3-ASR model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def qwen3_asr_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3-ASR toy model from config to a temporary directory.

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        temp_dir = tmp_path_factory.mktemp("qwen3_asr_toy_model")
        model_dir = temp_dir / "qwen3_asr_toy"

        # Create Qwen3ASR config from the toy model config
        config = Qwen3ASRConfig(**HF_QWEN3_ASR_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        model = Qwen3ASRForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Save config.json explicitly to ensure compatibility
        config_to_save = HF_QWEN3_ASR_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        # Download and save a real tokenizer (needed for roundtrip save)
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            # Create minimal tokenizer files if download fails
            tokenizer_config = {
                "tokenizer_class": "Qwen2Tokenizer",
                "vocab_size": 2048,
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, qwen3_asr_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.
        """
        model_path = Path(qwen3_asr_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "model.safetensors.index.json"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer config
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen3_asr"
        assert "thinker_config" in config_data
        thinker = config_data["thinker_config"]
        assert thinker["text_config"]["hidden_size"] == 256
        assert thinker["text_config"]["num_hidden_layers"] == 2
        assert thinker["text_config"]["num_attention_heads"] == 4
        assert thinker["text_config"]["vocab_size"] == 2048
        assert thinker["audio_config"]["output_dim"] == 256
        assert thinker["audio_config"]["d_model"] == 64

        # Verify model can be loaded
        _ = Qwen3ASRForConditionalGeneration.from_pretrained(
            qwen3_asr_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

        print(f"SUCCESS: Toy model created and validated at {qwen3_asr_toy_model_path}")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
        ],
    )
    def test_qwen3_asr_conversion_parallelism(self, qwen3_asr_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Qwen3-ASR model conversion with different parallelism configurations.
        """
        test_output_dir = tmp_path / f"qwen3_asr_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen3_asr_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--trust-remote-code",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Qwen3-ASR {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            model_name = Path(qwen3_asr_toy_model_path).name
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Verify the config contains Qwen3-ASR-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "qwen3_asr", "Model type should be qwen3_asr"
            assert "thinker_config" in saved_config, "ASR model should have thinker_config"

            print(f"SUCCESS: Qwen3-ASR {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Qwen3-ASR {test_name} conversion test: {e}")
            raise
