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
Functional test for Gemma 4 VL HF ↔ Megatron round-trip conversion.

Builds a tiny MoE Gemma 4 VLM (2 layers, 4 experts, mix of sliding +
full attention to exercise the K=V global-attention path) and runs
``examples/conversion/hf_megatron_roundtrip_multi_gpu.py`` over it.

The Gemma 4 VL bridge requires ``enable_moe_block=True`` — dense Gemma 4
is not yet supported — so the toy must be a real MoE config.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch


# Skip the entire module if the transformers Gemma 4 implementation is not
# available in the installed transformers version.
pytest.importorskip("transformers.models.gemma4")

from transformers import AutoTokenizer  # noqa: E402
from transformers.models.gemma4 import (  # noqa: E402
    Gemma4Config,
    Gemma4ForConditionalGeneration,
)


# Tiny MoE VLM config — drastically reduced from 26B-A4B for fast CI runs.
# - num_hidden_layers=2:        layer 0 sliding, layer 1 full (covers K=V global)
# - num_experts=4, top_k=2:     minimum to exercise MoE routing
# - vocab_size=2048:            small embedding to keep checkpoint tiny
# - image_token_id < vocab:     stay within the toy tokenizer range
HF_GEMMA4_VL_TOY_MODEL_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "torch_dtype": "bfloat16",
    "bos_token_id": 2,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "image_token_id": 200,
    "video_token_id": 201,
    "vision_soft_tokens_per_image": 16,
    "text_config": {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "global_head_dim": 128,
        "num_global_key_value_heads": 2,
        "attention_k_eq_v": True,
        "vocab_size": 2048,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "sliding_window": 1024,
        "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0,
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000.0},
        },
        "enable_moe_block": True,
        "num_experts": 4,
        "top_k_experts": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "final_logit_softcapping": 30.0,
        "hidden_act": "gelu_pytorch_tanh",
        "torch_dtype": "bfloat16",
    },
    "vision_config": {
        "model_type": "siglip_vision_model",
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "patch_size": 14,
        "image_size": 224,
        "rms_norm_eps": 1e-6,
        "vision_use_head": False,
    },
}


class TestGemma4VLConversion:
    """Round-trip conversion test for the Gemma 4 VL MoE bridge."""

    @pytest.fixture(scope="class")
    def gemma4_vl_toy_model_path(self, tmp_path_factory):
        """Build and save a tiny Gemma 4 VL MoE checkpoint."""
        temp_dir = tmp_path_factory.mktemp("gemma4_vl_toy_model")
        model_dir = temp_dir / "gemma4_vl_toy"

        config_dict = json.loads(json.dumps(HF_GEMMA4_VL_TOY_MODEL_CONFIG))
        config = Gemma4Config(**config_dict)
        config.torch_dtype = torch.bfloat16

        # Force eager attention on every sub-config: HF defaults to
        # flash_attention_2 in transformers >=5.5 when flash-attn is
        # installed, but the toy vision tower may not declare support.
        for sub in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
            if sub is not None:
                sub._attn_implementation = "eager"

        model = Gemma4ForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        model.save_pretrained(model_dir, safe_serialization=True)

        # Persist the dict form of config.json so the round-trip script
        # sees the toy values exactly as written (HF may re-emit defaults).
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Best-effort tokenizer download; fall back to a minimal stub.
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            tokenizer_config = {
                "tokenizer_class": "GemmaTokenizer",
                "vocab_size": 2048,
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, gemma4_vl_toy_model_path):
        """Sanity-check the toy checkpoint shape and config."""
        model_path = Path(gemma4_vl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "model.safetensors.index.json"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "gemma4"
        text_cfg = config_data["text_config"]
        assert text_cfg["num_hidden_layers"] == 2
        assert text_cfg["enable_moe_block"] is True
        assert text_cfg["num_experts"] == 4
        assert text_cfg["top_k_experts"] == 2
        assert text_cfg["layer_types"] == ["sliding_attention", "full_attention"]
        assert "vision_config" in config_data

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,etp,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 1, 2, "ETP"),
        ],
    )
    def test_gemma4_vl_conversion_parallelism(self, gemma4_vl_toy_model_path, tmp_path, tp, pp, etp, test_name):
        """Run HF → Megatron → HF round-trip across TP / ETP configs."""
        test_output_dir = tmp_path / f"gemma4_vl_{test_name}"
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
            gemma4_vl_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--etp",
            str(etp),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent.parent.parent,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Gemma4 VL {test_name} conversion failed with return code {result.returncode}"

        model_name = Path(gemma4_vl_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "gemma4", "Model type should be gemma4"
        assert "text_config" in saved_config, "VL model should have text_config"
        assert "vision_config" in saved_config, "VL model should have vision_config"
        assert saved_config["text_config"]["num_experts"] == 4, "Number of experts should match toy config"
        assert saved_config["text_config"]["enable_moe_block"] is True, "MoE block should be enabled"
