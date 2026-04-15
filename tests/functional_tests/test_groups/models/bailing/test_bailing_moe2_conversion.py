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
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from megatron.bridge.models.bailing.configuration_bailing_moe_v2 import BailingMoeV2Config
from megatron.bridge.models.bailing.modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM


# Register local config and model classes so that AutoConfig / AutoModelForCausalLM can resolve
# by model_type without network access (works in offline CI environments).
AutoConfig.register("bailing_moe_v2", BailingMoeV2Config, exist_ok=True)
AutoModelForCausalLM.register(BailingMoeV2Config, BailingMoeV2ForCausalLM, exist_ok=True)


# Toy config: reduced dims for fast testing.
# Keeps architectural properties: MoE with first_k_dense_replace, shared experts, QKV bias option.
HF_BAILING_MOE2_TOY_MODEL_CONFIG = {
    "architectures": ["BailingMoeV2ForCausalLM"],
    "model_type": "bailing_moe_v2",
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "moe_intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "vocab_size": 32000,
    "num_experts": 8,
    "num_experts_per_tok": 4,
    "first_k_dense_replace": 1,
    "num_nextn_predict_layers": 0,
    "use_qkv_bias": False,
    "tie_word_embeddings": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "initializer_range": 0.02,
    "torch_dtype": "bfloat16",
}


class TestBailingMoeV2Conversion:
    """
    Test Bailing MoE V2 model conversion with different parallelism configurations.
    Uses a toy model (2 layers, 8 experts) with random weights.
    """

    @pytest.fixture(scope="class")
    def toy_model_path(self, tmp_path_factory):
        """Create and save a toy Bailing MoE V2 model to a temporary directory."""
        temp_dir = tmp_path_factory.mktemp("bailing_moe2_toy_model")
        model_dir = temp_dir / "bailing_moe2_toy"

        config = BailingMoeV2Config(**HF_BAILING_MOE2_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        model = BailingMoeV2ForCausalLM(config).bfloat16()

        model.save_pretrained(model_dir, safe_serialization=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            pass

        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_BAILING_MOE2_TOY_MODEL_CONFIG, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, toy_model_path):
        """Verify the toy model was created correctly."""
        model_path = Path(toy_model_path)
        assert model_path.exists()

        config_file = model_path / "config.json"
        assert config_file.exists()

        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            sharded_files = list(model_path.glob("model-*-of-*.safetensors"))
            assert len(sharded_files) > 0, "No model weight files found"

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "bailing_moe_v2"
        assert config_data["hidden_size"] == 1024
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_experts"] == 8
        assert config_data["num_experts_per_tok"] == 4
        assert config_data["moe_intermediate_size"] == 512
        assert config_data["first_k_dense_replace"] == 1

        model = AutoModelForCausalLM.from_pretrained(
            toy_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        )

        assert len(model.model.layers) == 2
        # First layer is dense (first_k_dense_replace=1), second layer is MoE
        second_layer = model.model.layers[1]
        assert hasattr(second_layer, "mlp"), f"Expected 'mlp' attribute, got: {list(second_layer._modules.keys())}"
        moe_block = second_layer.mlp
        assert hasattr(moe_block, "experts"), f"MoE block missing 'experts', got: {list(moe_block._modules.keys())}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_bailing_moe2_conversion_parallelism(self, toy_model_path, tmp_path, tp, pp, ep, test_name):
        """Test Bailing MoE V2 model conversion with different parallelism configurations."""
        test_output_dir = tmp_path / f"bailing_moe2_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        repo_root = "/opt/Megatron-Bridge"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            f"--data-file={repo_root}/.coverage",
            f"--source={repo_root}/",
            "--parallel-mode",
            f"{repo_root}/examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            toy_model_path,
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

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Bailing MoE V2 {test_name} conversion failed with return code {result.returncode}")

        model_name = Path(toy_model_path).name
        converted_dir = test_output_dir / model_name
        assert converted_dir.exists(), f"Converted model directory not found at {converted_dir}"

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "bailing_moe_v2"
        assert saved_config["hidden_size"] == 1024
        assert saved_config["num_experts"] == 8
        assert saved_config["num_experts_per_tok"] == 4
        assert saved_config["moe_intermediate_size"] == 512
        assert saved_config["first_k_dense_replace"] == 1
