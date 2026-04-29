# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Functional tests for Qwen3-Omni HF <-> Megatron roundtrip conversion."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch


try:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

    _HAS_QWEN3_OMNI = True
except ImportError:
    _HAS_QWEN3_OMNI = False


HF_QWEN3_OMNI_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
    "model_type": "qwen3_omni_moe",
    "enable_audio_output": False,
    "torch_dtype": "bfloat16",
    "thinker_config": {
        "audio_token_id": 151646,
        "audio_start_token_id": 151647,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "position_id_per_seconds": 25,
        "text_config": {
            "model_type": "qwen3_moe",
            "vocab_size": 2048,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 8, 8]},
            "rms_norm_eps": 1e-06,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "decoder_sparse_step": 1,
            "moe_intermediate_size": 128,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 256,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 256,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [1],
        },
        "audio_config": {
            "num_mel_bins": 128,
            "encoder_layers": 2,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 512,
            "d_model": 256,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_function": "gelu",
            "activation_dropout": 0.0,
            "scale_embedding": False,
            "max_source_positions": 128,
            "n_window": 8,
            "output_dim": 256,
            "n_window_infer": 8,
            "conv_chunksize": 8,
            "downsample_hidden_size": 64,
        },
    },
}


@pytest.mark.skipif(not _HAS_QWEN3_OMNI, reason="transformers does not have Qwen3-Omni support")
class TestQwen3OmniConversion:
    """Test Qwen3-Omni conversion with a tiny self-contained toy checkpoint."""

    @pytest.fixture(scope="class")
    def qwen3_omni_toy_model_path(self, tmp_path_factory):
        """Create and save a toy Qwen3-Omni checkpoint for CI-safe conversion testing."""
        temp_dir = tmp_path_factory.mktemp("qwen3_omni_toy_model")
        model_dir = temp_dir / "qwen3_omni_toy"
        model_dir.mkdir(parents=True, exist_ok=True)

        config = Qwen3OmniMoeConfig(**HF_QWEN3_OMNI_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        model = Qwen3OmniMoeForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)
        model.save_pretrained(model_dir, safe_serialization=True)

        with open(model_dir / "tokenizer_config.json", "w") as f:
            json.dump({"tokenizer_class": "Qwen2Tokenizer", "vocab_size": 2048}, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, qwen3_omni_toy_model_path):
        """Verify the toy Qwen3-Omni checkpoint can be created and reloaded."""
        model_path = Path(qwen3_omni_toy_model_path)
        assert model_path.exists()
        assert (model_path / "config.json").exists()
        assert (model_path / "model.safetensors").exists()
        assert (model_path / "tokenizer_config.json").exists()

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen3_omni_moe"
        assert config_data["enable_audio_output"] is False
        assert config_data["thinker_config"]["text_config"]["hidden_size"] == 256
        assert config_data["thinker_config"]["text_config"]["num_hidden_layers"] == 4
        assert config_data["thinker_config"]["vision_config"]["depth"] == 2
        assert config_data["thinker_config"]["audio_config"]["encoder_layers"] == 2

        _ = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            qwen3_omni_toy_model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_omni_conversion(self, qwen3_omni_toy_model_path, tmp_path):
        """Run the HF -> Megatron -> HF roundtrip conversion on the toy checkpoint."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Qwen3-Omni conversion test requires at least 2 GPUs.")
        output_dir = tmp_path / "qwen3_omni_test"
        output_dir.mkdir(exist_ok=True)
        repo_root = Path(__file__).resolve().parents[4]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root / '3rdparty' / 'Megatron-LM'}"

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            f"--data-file={repo_root / '.coverage'}",
            f"--source={repo_root}",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen3_omni_toy_model_path,
            "--output-dir",
            str(output_dir),
            "--tp",
            "2",
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
            env=env,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Qwen3-Omni conversion failed with return code {result.returncode}"

        model_name = Path(qwen3_omni_toy_model_path).name
        converted_model_dir = output_dir / model_name
        assert converted_model_dir.exists()
        assert (converted_model_dir / "config.json").exists()
        assert (converted_model_dir / "model.safetensors").exists() or any(
            converted_model_dir.glob("model-*-of-*.safetensors")
        )

        with open(converted_model_dir / "config.json") as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "qwen3_omni_moe"
        assert saved_config["enable_audio_output"] is False
        assert saved_config["thinker_config"]["text_config"]["num_hidden_layers"] == 4
