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
from packaging.version import Version
from transformers import AutoConfig, AutoTokenizer
from transformers import __version__ as TRANSFORMERS_VERSION


HF_GLM5_TOY_MODEL_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    # ---- Smaller toy dims ----
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "moe_intermediate_size": 256,
    "num_hidden_layers": 2,
    # ---- Attention ----
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "qk_head_dim": 128,
    "qk_nope_head_dim": 96,
    "qk_rope_head_dim": 32,
    "v_head_dim": 128,
    # ---- DSA indexer ----
    "index_head_dim": 128,
    "index_n_heads": 8,
    "index_topk": 256,
    "indexer_rope_interleave": True,
    # ---- LoRA ranks ----
    "q_lora_rank": 256,
    "kv_lora_rank": 128,
    # ---- MoE ----
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "mlp_layer_types": ["dense", "sparse"],
    # ---- Position encoding ----
    "max_position_embeddings": 8192,
    "rope_interleave": True,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    # ---- Norm / activation ----
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    # ---- Attention behavior ----
    "attention_bias": False,
    "attention_dropout": 0.0,
    # ---- Tokens ----
    "vocab_size": 154880,
    "bos_token_id": 0,
    "eos_token_id": [154820, 154827, 154829],
    "pad_token_id": 154820,
    # ---- Misc ----
    "ep_size": 1,
    "num_nextn_predict_layers": 1,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,
    "use_cache": True,
    "dtype": "bfloat16",
    "pretraining_tp": 1,
    "transformers_version": "5.2.0.dev0",
}

pytestmark = pytest.mark.skipif(
    Version(TRANSFORMERS_VERSION) < Version("5.2.0"),
    reason=f"GLM5 conversion tests require transformers>=5.2.0, found {TRANSFORMERS_VERSION}",
)


def _create_glm5_toy_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained("zai-org/GLM-5")

    for key, value in HF_GLM5_TOY_MODEL_CONFIG.items():
        setattr(config, key, value)

    config.torch_dtype = torch.bfloat16

    from transformers import GlmMoeDsaForCausalLM

    model = GlmMoeDsaForCausalLM(config)

    model = model.bfloat16()
    for k, v in model.named_buffers():
        if "e_score_correction_bias" in k:
            v.data = v.data.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5")
    tokenizer.save_pretrained(model_dir)

    model.save_pretrained(model_dir, safe_serialization=True)

    config_to_save = HF_GLM5_TOY_MODEL_CONFIG.copy()
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)


@pytest.mark.pleasefixme
class TestGLM5Conversion:
    """
    Test GLM-5 (MoE + MLA + DSA) model conversion with different parallelism configurations.
    Uses a toy model with random weights.
    """

    @pytest.fixture(scope="class")
    def glm5_toy_model_path(self, tmp_path_factory):
        """Create and save a HuggingFace GLM-5 MoE toy model to a temporary directory."""
        temp_dir = tmp_path_factory.mktemp("glm5_toy_model")
        model_dir = temp_dir / "glm5_toy"

        _create_glm5_toy_model(model_dir)

        return str(model_dir)

    def test_toy_model_creation(self, glm5_toy_model_path):
        """Test that the toy MoE model is created correctly and can be loaded."""
        model_path = Path(glm5_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"

        if not weights_file.exists():
            sharded_files = list(model_path.glob("model-*-of-*.safetensors"))
            if sharded_files:
                weights_file = sharded_files[0]
            else:
                sharded_files = list(model_path.glob("pytorch_model-*-of-*.bin"))
                if sharded_files:
                    weights_file = sharded_files[0]

        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == HF_GLM5_TOY_MODEL_CONFIG["model_type"]
        assert config_data["hidden_size"] == HF_GLM5_TOY_MODEL_CONFIG["hidden_size"]
        assert config_data["intermediate_size"] == HF_GLM5_TOY_MODEL_CONFIG["intermediate_size"]
        assert config_data["num_hidden_layers"] == HF_GLM5_TOY_MODEL_CONFIG["num_hidden_layers"]
        assert config_data["num_attention_heads"] == HF_GLM5_TOY_MODEL_CONFIG["num_attention_heads"]
        assert config_data["vocab_size"] == HF_GLM5_TOY_MODEL_CONFIG["vocab_size"]
        assert config_data["n_routed_experts"] == HF_GLM5_TOY_MODEL_CONFIG["n_routed_experts"]
        assert config_data["num_experts_per_tok"] == HF_GLM5_TOY_MODEL_CONFIG["num_experts_per_tok"]
        assert config_data["moe_intermediate_size"] == HF_GLM5_TOY_MODEL_CONFIG["moe_intermediate_size"]

        from transformers import GlmMoeDsaForCausalLM

        model = GlmMoeDsaForCausalLM.from_pretrained(
            glm5_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )

        assert hasattr(model, "model")
        assert hasattr(model.model, "layers")
        assert len(model.model.layers) == 2

        second_layer = model.model.layers[1]
        assert hasattr(second_layer, "mlp")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_glm5_conversion_parallelism(self, glm5_toy_model_path, tmp_path, tp, pp, ep, test_name):
        """Test GLM-5 MoE model conversion with different parallelism configurations."""
        test_output_dir = tmp_path / f"glm5_moe_{test_name}"
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
            glm5_toy_model_path,
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
            cwd=repo_root,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"GLM-5 MoE {test_name} conversion failed with return code {result.returncode}")

        model_name = Path(glm5_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"

        weights_file_safetensors = converted_model_dir / "model.safetensors"
        weights_file_pytorch = converted_model_dir / "pytorch_model.bin"

        weights_found = weights_file_safetensors.exists() or weights_file_pytorch.exists()

        if not weights_found:
            sharded_safetensors = list(converted_model_dir.glob("model-*-of-*.safetensors"))
            sharded_pytorch = list(converted_model_dir.glob("pytorch_model-*-of-*.bin"))
            weights_found = len(sharded_safetensors) > 0 or len(sharded_pytorch) > 0

        assert weights_found, f"Model weights file not found in converted model at {converted_model_dir}"

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "glm_moe_dsa"
        assert saved_config["hidden_size"] == HF_GLM5_TOY_MODEL_CONFIG["hidden_size"]
        assert saved_config["num_attention_heads"] == HF_GLM5_TOY_MODEL_CONFIG["num_attention_heads"]
        assert saved_config["n_routed_experts"] == HF_GLM5_TOY_MODEL_CONFIG["n_routed_experts"]
        assert saved_config["num_experts_per_tok"] == HF_GLM5_TOY_MODEL_CONFIG["num_experts_per_tok"]
        assert saved_config["moe_intermediate_size"] == HF_GLM5_TOY_MODEL_CONFIG["moe_intermediate_size"]

    @pytest.mark.run_only_on("GPU")
    def test_glm5_autoconfig_roundtrip(self, glm5_toy_model_path, tmp_path):
        from tests.functional_tests.utils import autoconfig_roundtrip

        autoconfig_roundtrip(glm5_toy_model_path, tmp_path)
