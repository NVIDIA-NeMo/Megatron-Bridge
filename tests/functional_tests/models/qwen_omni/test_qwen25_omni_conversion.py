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

"""Functional tests for Qwen2.5-Omni HF <-> Megatron roundtrip conversion."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch


try:
    from transformers import Qwen2_5OmniForConditionalGeneration
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
        Qwen2_5OmniConfig,
        Qwen2_5OmniTextConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniVisionEncoderConfig,
    )

    _HAS_QWEN25_OMNI = True
except ImportError:
    _HAS_QWEN25_OMNI = False


def _tiny_qwen25_omni_config() -> "Qwen2_5OmniConfig":
    """Build a small in-memory Qwen2.5-Omni config suitable for conversion smoke tests."""
    text_config = Qwen2_5OmniTextConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "mrope_section": [8, 8, 8]},
        tie_word_embeddings=False,
    )
    vision_config = Qwen2_5OmniVisionEncoderConfig(
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=256,
        fullatt_block_indexes=[0, 1],
    )
    audio_config = Qwen2_5OmniAudioEncoderConfig(
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=512,
        d_model=256,
        output_dim=256,
        n_window=8,
    )
    thinker_config = Qwen2_5OmniThinkerConfig(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=audio_config,
    )
    return Qwen2_5OmniConfig(thinker_config=thinker_config, enable_audio_output=False)


def _coverage_args(repo_root: Path) -> list[str]:
    """Return coverage CLI args that work both in CI containers and local checkouts."""
    coverage_root = Path(os.environ.get("MEGATRON_BRIDGE_COVERAGE_ROOT", str(repo_root)))
    return [
        "--data-file",
        str(coverage_root / ".coverage"),
        "--source",
        f"{coverage_root}/",
    ]


def _distributed_launch_args() -> tuple[list[str], str]:
    """Return torchrun launch args and tensor parallel size for local/CI GPU availability."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nproc = 2 if gpu_count >= 2 else 1
    return [
        "--nproc_per_node",
        str(nproc),
        "--nnodes=1",
    ], str(nproc)


@pytest.mark.skipif(not _HAS_QWEN25_OMNI, reason="transformers does not have Qwen2.5-Omni support")
class TestQwen25OmniConversion:
    """Test Qwen2.5-Omni conversion with a tiny self-contained toy checkpoint."""

    @pytest.fixture(scope="class")
    def qwen25_omni_toy_model_path(self, tmp_path_factory):
        """Create and save a toy Qwen2.5-Omni checkpoint for CI-safe conversion testing."""
        temp_dir = tmp_path_factory.mktemp("qwen25_omni_toy_model")
        model_dir = temp_dir / "qwen25_omni_toy"
        model_dir.mkdir(parents=True, exist_ok=True)

        config = _tiny_qwen25_omni_config()
        config.torch_dtype = torch.bfloat16

        model = Qwen2_5OmniForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)
        model.save_pretrained(model_dir, safe_serialization=True)

        with open(model_dir / "vocab.json", "w") as f:
            json.dump({"<|endoftext|>": 0, "a": 1, "b": 2, "c": 3}, f, indent=2)

        with open(model_dir / "merges.txt", "w") as f:
            f.write("#version: 0.2\n")

        with open(model_dir / "tokenizer_config.json", "w") as f:
            json.dump(
                {
                    "tokenizer_class": "Qwen2Tokenizer",
                    "vocab_size": 2048,
                    "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
                    "unk_token": "<|endoftext|>",
                },
                f,
                indent=2,
            )

        with open(model_dir / "special_tokens_map.json", "w") as f:
            json.dump(
                {
                    "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
                    "unk_token": "<|endoftext|>",
                },
                f,
                indent=2,
            )

        return str(model_dir)

    def test_toy_model_creation(self, qwen25_omni_toy_model_path):
        """Verify the toy Qwen2.5-Omni checkpoint can be created and reloaded."""
        model_path = Path(qwen25_omni_toy_model_path)
        assert model_path.exists()
        assert (model_path / "config.json").exists()
        assert (model_path / "model.safetensors").exists()
        assert (model_path / "tokenizer_config.json").exists()

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen2_5_omni"
        assert config_data["enable_audio_output"] is False
        assert config_data["thinker_config"]["text_config"]["hidden_size"] == 256
        assert config_data["thinker_config"]["text_config"]["num_hidden_layers"] == 4
        assert config_data["thinker_config"]["vision_config"]["depth"] == 2
        assert config_data["thinker_config"]["audio_config"]["encoder_layers"] == 2

        _ = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            qwen25_omni_toy_model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

    @pytest.mark.run_only_on("GPU")
    def test_qwen25_omni_conversion(self, qwen25_omni_toy_model_path, tmp_path):
        """Run the HF -> Megatron -> HF roundtrip conversion on the toy checkpoint."""
        output_dir = tmp_path / "qwen25_omni_test"
        output_dir.mkdir(exist_ok=True)
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        launch_args, tp_size = _distributed_launch_args()

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            *launch_args,
            "-m",
            "coverage",
            "run",
            *_coverage_args(repo_root),
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen25_omni_toy_model_path,
            "--output-dir",
            str(output_dir),
            "--tp",
            tp_size,
            "--pp",
            "1",
            "--ep",
            "1",
            "--etp",
            "1",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Qwen2.5-Omni conversion failed with return code {result.returncode}"

        model_name = Path(qwen25_omni_toy_model_path).name
        converted_model_dir = output_dir / model_name
        assert converted_model_dir.exists()
        assert (converted_model_dir / "config.json").exists()
        assert (converted_model_dir / "model.safetensors").exists() or any(
            converted_model_dir.glob("model-*-of-*.safetensors")
        )

        with open(converted_model_dir / "config.json") as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "qwen2_5_omni"
        assert saved_config["enable_audio_output"] is False
        assert saved_config["thinker_config"]["text_config"]["num_hidden_layers"] == 4
