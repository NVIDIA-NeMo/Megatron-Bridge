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

"""
Functional tests for Qwen3.5 VL HF ↔ Megatron roundtrip conversion.

Qwen3.5 uses a hybrid Gated DeltaNet (GDN) + Gated Attention architecture.
The full_attention_interval=4 means every 4th layer is standard attention,
so num_hidden_layers must be a multiple of 4.

Run dense test:
  uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/functional_tests/models/qwen_vl/test_qwen35_vl_conversion.py::TestQwen35VLConversion -v -s

Run MoE test:
  uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/functional_tests/models/qwen_vl/test_qwen35_vl_conversion.py::TestQwen35VLMoEConversion -v -s
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch


try:
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    _HAS_QWEN3_5 = True
except ImportError:
    _HAS_QWEN3_5 = False

try:
    from transformers import Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    _HAS_QWEN3_5_MOE = True
except ImportError:
    _HAS_QWEN3_5_MOE = False


# ---------------------------------------------------------------------------
# Tiny dense config (Qwen3.5 dense style, ~small param count for fast tests)
# num_hidden_layers must be a multiple of full_attention_interval (4)
# ---------------------------------------------------------------------------
HF_QWEN35_VL_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "bos_token_id": 248045,
    "eos_token_id": 248044,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_5",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "attention_bias": False,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,
    "vision_config": {
        "depth": 1,
        "embed_dim": 256,
        "hidden_size": 256,
        "hidden_act": "silu",
        "in_channels": 3,
        "mlp_ratio": 2.0,
        "num_heads": 4,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "rope_scaling": {"mrope_section": [8, 8, 8]},
    "text_config": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 2048,
        "max_position_embeddings": 32768,
        "rope_theta": 10000000.0,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "rope_scaling": {"rope_type": "default", "mrope_section": [8, 8, 8]},
        "rope_parameters": {
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
            "mrope_section": [8, 8, 8],
        },
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 2048,
}


@pytest.mark.skipif(not _HAS_QWEN3_5, reason="transformers does not have Qwen3.5 (dense) support")
class TestQwen35VLConversion:
    """Test Qwen3.5 VL dense model conversion from HuggingFace to Megatron."""

    @pytest.fixture(scope="class")
    def qwen35_vl_toy_model_path(self, tmp_path_factory):
        """Create and save a dense Qwen3.5 VL toy model."""
        temp_dir = tmp_path_factory.mktemp("qwen35_vl_toy_model")
        model_dir = temp_dir / "qwen35_vl_toy"

        config = Qwen3_5Config(**HF_QWEN35_VL_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_parameters = {
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
                "mrope_section": [8, 8, 8],
                "rope_theta": 10000000.0,
            }

        model = Qwen3_5ForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            tokenizer_config = {
                "tokenizer_class": "Qwen2Tokenizer",
                "vocab_size": 248320,
            }
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        model.save_pretrained(model_dir, safe_serialization=True)
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN35_VL_TOY_MODEL_CONFIG, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, qwen35_vl_toy_model_path):
        """Verify the toy model was created correctly."""
        model_path = Path(qwen35_vl_toy_model_path)
        assert model_path.exists()

        config_file = model_path / "config.json"
        assert config_file.exists()

        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "model.safetensors.index.json"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists()

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen3_5"
        assert config_data["hidden_size"] == 256
        assert config_data["num_hidden_layers"] == 4
        assert config_data["full_attention_interval"] == 4
        assert "vision_config" in config_data

        _ = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen35_vl_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("tp,pp", [(2, 1)])
    def test_qwen35_vl_conversion(self, qwen35_vl_toy_model_path, tmp_path, tp, pp):
        """Test dense Qwen3.5 VL conversion with TP parallelism."""
        test_output_dir = tmp_path / "qwen35_vl_test"
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
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen35_vl_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Qwen3.5 VL dense conversion failed with return code {result.returncode}"

        model_name = Path(qwen35_vl_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists()

        config_file = converted_model_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "qwen3_5"
        assert saved_config["hidden_size"] == 256
        assert "vision_config" in saved_config


# ---------------------------------------------------------------------------
# Tiny MoE config (Qwen3.5 MoE style)
# ---------------------------------------------------------------------------
HF_QWEN35_VL_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "bos_token_id": 248045,
    "eos_token_id": 248046,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_5_moe",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "attention_bias": False,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "shared_expert_intermediate_size": 512,
    "decoder_sparse_step": 1,
    "vision_config": {
        "depth": 1,
        "embed_dim": 256,
        "hidden_size": 256,
        "hidden_act": "silu",
        "in_channels": 3,
        "mlp_ratio": 2.0,
        "num_heads": 4,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "rope_scaling": {"mrope_section": [8, 8, 8]},
    "text_config": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 2048,
        "max_position_embeddings": 32768,
        "rope_theta": 10000000.0,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 512,
        "decoder_sparse_step": 1,
        "rope_scaling": {"rope_type": "default", "mrope_section": [8, 8, 8]},
        "rope_parameters": {
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
            "mrope_section": [8, 8, 8],
        },
        "torch_dtype": "bfloat16",
    },
    "vocab_size": 2048,
}


@pytest.mark.skipif(not _HAS_QWEN3_5_MOE, reason="transformers does not have Qwen3.5 MoE support")
class TestQwen35VLMoEConversion:
    """Test Qwen3.5 VL MoE model conversion."""

    @pytest.fixture(scope="class")
    def qwen35_vl_moe_toy_model_path(self, tmp_path_factory):
        """Create and save a MoE Qwen3.5 VL toy model."""
        temp_dir = tmp_path_factory.mktemp("qwen35_vl_moe_toy_model")
        model_dir = temp_dir / "qwen35_vl_moe_toy"

        config = Qwen3_5MoeConfig(**HF_QWEN35_VL_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_parameters = {
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
                "mrope_section": [8, 8, 8],
                "rope_theta": 10000000.0,
            }

        model = Qwen3_5MoeForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            tokenizer_config = {
                "tokenizer_class": "Qwen2Tokenizer",
                "vocab_size": 248320,
            }
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        model.save_pretrained(model_dir, safe_serialization=True)
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN35_VL_MOE_TOY_MODEL_CONFIG, f, indent=2)

        return str(model_dir)

    def test_moe_toy_model_creation(self, qwen35_vl_moe_toy_model_path):
        """Verify the MoE toy model was created correctly."""
        model_path = Path(qwen35_vl_moe_toy_model_path)
        assert model_path.exists()

        config_file = model_path / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "qwen3_5_moe"
        assert config_data["num_experts"] == 4
        assert config_data["full_attention_interval"] == 4

        _ = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            qwen35_vl_moe_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("tp,pp", [(2, 1)])
    def test_moe_conversion(self, qwen35_vl_moe_toy_model_path, tmp_path, tp, pp):
        """Test MoE Qwen3.5 VL conversion with TP parallelism."""
        test_output_dir = tmp_path / "qwen35_vl_moe_test"
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
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen35_vl_moe_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Qwen3.5 VL MoE conversion failed with return code {result.returncode}"

        model_name = Path(qwen35_vl_moe_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists()

        config_file = converted_model_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "qwen3_5_moe"
        assert saved_config["num_experts"] == 4
