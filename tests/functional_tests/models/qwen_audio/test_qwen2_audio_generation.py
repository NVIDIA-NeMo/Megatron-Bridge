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
Functional tests for Qwen2 Audio HF to Megatron generation.

Example run commands:
    # Run all generation tests
    pytest tests/functional_tests/models/qwen_audio/test_qwen2_audio_generation.py

    # Run specific test
    pytest tests/functional_tests/models/qwen_audio/test_qwen2_audio_generation.py::TestQwen2AudioGeneration::test_qwen2_audio_generation

Note: These tests use small proxy/toy models for fast generation testing.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen2AudioConfig, Qwen2AudioForConditionalGeneration


HF_QWEN2_AUDIO_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen2AudioForConditionalGeneration"],
    "audio_token_index": 151646,
    "model_type": "qwen2_audio",
    "audio_config": {
        "model_type": "qwen2_audio_encoder",
        "num_mel_bins": 128,
        "d_model": 256,
        "encoder_layers": 4,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 512,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_function": "gelu",
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "num_hidden_layers": 4,
        "initializer_range": 0.02,
        "scale_embedding": False,
        "max_source_positions": 1500,
    },
    "text_config": {
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-06,
        "use_cache": True,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    },
}


class TestQwen2AudioGeneration:
    """
    Test Qwen2 Audio model generation using HF to Megatron conversion with audio inputs.
    Uses small proxy/toy models for fast generation testing.
    """

    @pytest.fixture(scope="class")
    def qwen2_audio_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen2 Audio toy model to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen2_audio_generation_toy_model")
        model_dir = temp_dir / "qwen2_audio_toy"

        # Create Qwen2 Audio config from the toy model config
        config = Qwen2AudioConfig(**HF_QWEN2_AUDIO_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        model = Qwen2AudioForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        # Download and save tokenizer and processor from a reference Qwen2 Audio model
        try:
            from transformers import AutoProcessor

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            tokenizer.save_pretrained(model_dir)

            # Also save the processor
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            processor.save_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: Could not download tokenizer/processor, creating minimal files: {e}")
            # Create minimal tokenizer files if download fails
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

        # Also save config.json explicitly
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN2_AUDIO_TOY_MODEL_CONFIG, f, indent=2)

        print(f"Created toy model at: {model_dir}")
        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    def test_qwen2_audio_generation(self, qwen2_audio_toy_model_path):
        """
        Test Qwen2 Audio toy model with audio generation.
        Uses a small proxy model instead of the full 7B model for fast testing.
        Uses real audio to test audio-language pipeline.

        Args:
            qwen2_audio_toy_model_path: Path to the toy Qwen2 Audio model (from fixture)
        """
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/conversion/hf_to_megatron_generate_audio_lm.py",
            f"--hf_model_path={qwen2_audio_toy_model_path}",
            "--audio_url=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
            "--prompt=What's that sound?",
            "--tp=2",
            "--max_new_tokens=50",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent,
            )

            # Print output for debugging
            print("\n" + "=" * 80)
            print("STDOUT:")
            print(result.stdout)
            print("\n" + "=" * 80)
            print("STDERR:")
            print(result.stderr)
            print("=" * 80 + "\n")

            if result.returncode != 0:
                assert False, f"Qwen2-Audio toy model generation failed with return code {result.returncode}"

            print("SUCCESS: Qwen2-Audio toy model generation test completed successfully")

        except subprocess.TimeoutExpired:
            assert False, "Qwen2-Audio toy model generation test timed out after 5 minutes"
        except Exception as e:
            print(f"Error during Qwen2-Audio toy model generation test: {e}")
            raise
