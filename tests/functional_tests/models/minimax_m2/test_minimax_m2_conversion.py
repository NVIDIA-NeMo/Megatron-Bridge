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


try:
    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM
except ImportError:
    pytest.skip(
        "MiniMaxM2 not available in current transformers version (requires >= 4.57)",
        allow_module_level=True,
    )

# Toy config: reduced dims for fast testing.
# Keeps architectural properties: MoE, partial RoPE, QK norm, sigmoid routing.
HF_MINIMAX_M2_TOY_MODEL_CONFIG = {
    "architectures": ["MiniMaxM2ForCausalLM"],
    "model_type": "minimax_m2",
    "hidden_size": 512,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "rotary_dim": 32,
    "vocab_size": 1024,
    "tie_word_embeddings": False,
    "attention_dropout": 0.0,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "scoring_func": "sigmoid",
    "use_routing_bias": True,
    "use_qk_norm": True,
    "qk_norm_type": "per_layer",
    "router_aux_loss_coef": 0.001,
    "router_jitter_noise": 0.0,
    "output_router_logits": False,
    "torch_dtype": "bfloat16",
}


class TestMiniMaxM2Conversion:
    """
    Test MiniMax-M2 MoE model conversion with different parallelism configurations.
    Uses a toy model (2 layers, 4 experts) with random weights.
    """

    @pytest.fixture(scope="class")
    def toy_model_path(self, tmp_path_factory):
        """Create and save a toy MiniMax-M2 model to a temporary directory."""
        temp_dir = tmp_path_factory.mktemp("minimax_m2_toy_model")
        model_dir = temp_dir / "minimax_m2_toy"

        config = MiniMaxM2Config(**HF_MINIMAX_M2_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        model = MiniMaxM2ForCausalLM(config).bfloat16()

        model.save_pretrained(model_dir, safe_serialization=True)

        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_MINIMAX_M2_TOY_MODEL_CONFIG, f, indent=2)

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

        assert config_data["model_type"] == "minimax_m2"
        assert config_data["hidden_size"] == 512
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_local_experts"] == 4
        assert config_data["num_experts_per_tok"] == 2

        model = MiniMaxM2ForCausalLM.from_pretrained(
            toy_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        )

        assert len(model.model.layers) == 2
        first_layer = model.model.layers[0]
        # Native transformers uses "mlp" for the MoE block
        assert hasattr(first_layer, "mlp"), f"Expected 'mlp' attribute, got: {list(first_layer._modules.keys())}"
        moe_block = first_layer.mlp
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
    def test_minimax_m2_conversion_parallelism(self, toy_model_path, tmp_path, tp, pp, ep, test_name):
        """
        Test MiniMax-M2 model conversion with different parallelism configurations.
        """
        test_output_dir = tmp_path / f"minimax_m2_{test_name}"
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
            toy_model_path,
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
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"MiniMax-M2 {test_name} conversion failed with return code {result.returncode}"

        model_name = Path(toy_model_path).name
        converted_dir = test_output_dir / model_name
        assert converted_dir.exists(), f"Converted model directory not found at {converted_dir}"

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "minimax_m2"
        assert saved_config["hidden_size"] == 512
        assert saved_config["num_local_experts"] == 4
