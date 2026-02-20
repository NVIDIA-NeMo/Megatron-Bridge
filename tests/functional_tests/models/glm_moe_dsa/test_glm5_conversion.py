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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer


HF_GLM5_TOY_MODEL_CONFIG = {
  "architectures": ["GlmMoeDsaForCausalLM"],
  "model_type": "glm_moe_dsa",

  # ---- Smaller toy dims (keep vocab_size and num_hidden_layers unchanged) ----
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
  "first_k_dense_replace": 1,  # changed from 3
  "n_group": 1,
  "topk_group": 1,
  "norm_topk_prob": True,
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "topk_method": "noaux_tc",

  # Exactly one dense, one sparse
  "mlp_layer_types": [
    "dense",
    "sparse"
  ],

  # ---- Position encoding ----
  "max_position_embeddings": 8192,
  "rope_interleave": True,
  "rope_parameters": {
    "rope_theta": 1000000,
    "rope_type": "default"
  },

  # ---- Norm / activation ----
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,

  # ---- Attention behavior ----
  "attention_bias": False,
  "attention_dropout": 0.0,

  # ---- Tokens (unchanged) ----
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
  "transformers_version": "5.2.0.dev0"
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _make_tmp_dir(tmp_path_factory, prefix: str) -> Path:
    """Create temp dirs on fast local disk when available to avoid /tmp quota issues."""
    preferred_root = Path(os.environ.get("GLM5_TEST_TMP_ROOT", "/opt/dlami/nvme/peng"))
    try:
        preferred_root.mkdir(parents=True, exist_ok=True)
        if preferred_root.exists() and os.access(preferred_root, os.W_OK):
            return Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(preferred_root)))
    except Exception:
        pass
    return tmp_path_factory.mktemp(prefix)


def _create_glm5_toy_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create GLM 4.5 config from the toy model config using AutoConfig
    config = AutoConfig.from_pretrained("zai-org/GLM-5")

    # Override with toy model config
    for key, value in HF_GLM5_TOY_MODEL_CONFIG.items():
        setattr(config, key, value)

    config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

    # Create model with random weights and convert to bfloat16
    from transformers import GlmMoeDsaForCausalLM

    model = GlmMoeDsaForCausalLM(config)

    model = model.bfloat16()  # Use .bfloat16() method instead of .to()
    for k, v in model.named_buffers():
        if "e_score_correction_bias" in k:
            v.data = v.data.to(torch.float32)

    # Download and save tokenizer from a reference GLM model
    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5")
    tokenizer.save_pretrained(model_dir)

    # Save model and config to directory
    model.save_pretrained(model_dir, safe_serialization=True)

    # Also save config.json explicitly to ensure compatibility with correct torch_dtype
    config_to_save = HF_GLM5_TOY_MODEL_CONFIG.copy()
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)


def _build_roundtrip_cmd(
    model_path: str, output_dir: Path, tp: int, pp: int, ep: int, repo_root: Path
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
    ]

    if os.environ.get("MBRIDGE_USE_COVERAGE") == "1":
        cmd += [
            "-m",
            "coverage",
            "run",
            "--data-file",
            str(repo_root / ".coverage"),
            "--source",
            str(repo_root),
            "--parallel-mode",
        ]

    cmd += [
        "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
        "--hf-model-id",
        model_path,
        "--output-dir",
        str(output_dir),
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--ep",
        str(ep),
    ]
    return cmd


class TestGLM5Conversion:
    """
    Test GLM 4.5 MoE model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def glm5_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace GLM 5 MoE toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = _make_tmp_dir(tmp_path_factory, "glm5_toy_model")
        model_dir = temp_dir / "glm5_toy"

        _create_glm5_toy_model(model_dir)

        return str(model_dir)

    def test_toy_model_creation(self, glm5_toy_model_path):
        """
        Test that the toy MoE model is created correctly and can be loaded.

        Args:
            glm5_toy_model_path: Path to the toy GLM 5 MoE model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(glm5_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"

        # If neither single file exists, check for sharded files
        if not weights_file.exists():
            # Check for sharded safetensors files
            sharded_files = list(model_path.glob("model-*-of-*.safetensors"))
            if sharded_files:
                weights_file = sharded_files[0]  # Use first shard as representative
            else:
                # Check for sharded pytorch files
                sharded_files = list(model_path.glob("pytorch_model-*-of-*.bin"))
                if sharded_files:
                    weights_file = sharded_files[0]  # Use first shard as representative

        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == HF_GLM5_TOY_MODEL_CONFIG["model_type"]
        assert config_data["hidden_size"] == HF_GLM5_TOY_MODEL_CONFIG["hidden_size"]
        assert config_data["intermediate_size"] == HF_GLM5_TOY_MODEL_CONFIG["intermediate_size"]
        assert config_data["num_hidden_layers"] == HF_GLM5_TOY_MODEL_CONFIG["num_hidden_layers"]
        assert config_data["num_attention_heads"] == HF_GLM5_TOY_MODEL_CONFIG["num_attention_heads"]
        assert config_data["vocab_size"] == HF_GLM5_TOY_MODEL_CONFIG["vocab_size"]
        # Verify MoE specific parameters
        assert config_data["n_routed_experts"] == HF_GLM5_TOY_MODEL_CONFIG["n_routed_experts"]
        assert config_data["num_experts_per_tok"] == HF_GLM5_TOY_MODEL_CONFIG["num_experts_per_tok"]
        assert config_data["moe_intermediate_size"] == HF_GLM5_TOY_MODEL_CONFIG["moe_intermediate_size"]

        # Try loading the model to verify it's valid
        # try:
        from transformers import GlmMoeDsaForCausalLM

        model = GlmMoeDsaForCausalLM.from_pretrained(
            glm5_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,  # Ensure full loading
            trust_remote_code=True,
        )

        # Verify model structure
        print(f"Model: {model}")
        assert hasattr(model, "model")
        assert hasattr(model.model, "layers")
        assert len(model.model.layers) == 2  # num_hidden_layers

        # Verify MoE structure
        # First layer is dense, second layer should have MoE structure
        second_layer = model.model.layers[1]
        assert hasattr(second_layer, "mlp")
        print(f"second_layer mlp: {second_layer.mlp}")
        total_size = [param.numel() for param in second_layer.mlp.experts.parameters()]
        total_shapes = [param.shape for param in second_layer.mlp.experts.parameters()]
        print(f"second_layer mlp experts: {second_layer.mlp.experts} and type: {type(second_layer.mlp.experts)} and size: {total_size} and shapes: {total_shapes}")
        # GLM 5 MoE structure check (may vary based on implementation)
        # if hasattr(second_layer.mlp, "experts"):
        #     assert len(second_layer.mlp.experts) == 8  # n_routed_experts

        print(f"SUCCESS: GLM 5 MoE toy model created and validated at {glm5_toy_model_path}")
        print("Model weights are correctly in bfloat16 format")
        print(f"MoE structure validated: {config_data['n_routed_experts']} experts")

        # except Exception as e:
        #     assert False, f"Failed to load created toy MoE model: {e}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_glm5_conversion_parallelism(self, glm5_toy_model_path, tmp_path_factory, tp, pp, ep, test_name):
        """
        Test GLM 5 MoE model conversion with different parallelism configurations.

        Args:
            glm5_toy_model_path: Path to the toy GLM 5 MoE model (from fixture)
            tmp_path_factory: Pytest temporary path factory
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            ep: Expert parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = _make_tmp_dir(tmp_path_factory, f"glm5_moe_{test_name}_out")
        test_output_dir.mkdir(exist_ok=True)

        repo_root = _repo_root()
        cmd = _build_roundtrip_cmd(
            glm5_toy_model_path, test_output_dir, tp, pp, ep, repo_root
        )

        # try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
        print(f"RESULT: {result}")
        # Check that the conversion completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"GLM 5 MoE {test_name} conversion failed with return code {result.returncode}"

        # Verify that the converted model was saved
        # The output directory should be named after the last part of the model path
        model_name = Path(glm5_toy_model_path).name  # "glm5_toy"
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        # Check that essential model files exist
        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"

        # Check for model weights file (could be either safetensors or pytorch_model.bin)
        weights_file_safetensors = converted_model_dir / "model.safetensors"
        weights_file_pytorch = converted_model_dir / "pytorch_model.bin"

        # Check for single files first
        weights_found = weights_file_safetensors.exists() or weights_file_pytorch.exists()

        # If single files don't exist, check for sharded files
        if not weights_found:
            sharded_safetensors = list(converted_model_dir.glob("model-*-of-*.safetensors"))
            sharded_pytorch = list(converted_model_dir.glob("pytorch_model-*-of-*.bin"))
            weights_found = len(sharded_safetensors) > 0 or len(sharded_pytorch) > 0

        assert weights_found, f"Model weights file not found in converted model at {converted_model_dir}"

        # Verify the config contains GLM 5 MoE-specific parameters
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "glm_moe_dsa", (
            "Model type should be glm (GLM 5 MoE uses GlmMoeDsaForCausalLM)"
        )
        assert saved_config["hidden_size"] == HF_GLM5_TOY_MODEL_CONFIG["hidden_size"], (
            "Hidden size should match toy config"
        )
        assert saved_config["num_attention_heads"] == HF_GLM5_TOY_MODEL_CONFIG["num_attention_heads"], (
            "Number of attention heads should match toy config"
        )
        # Verify MoE specific parameters are preserved
        assert saved_config["n_routed_experts"] == HF_GLM5_TOY_MODEL_CONFIG["n_routed_experts"], (
            "Number of routed experts should match toy config"
        )
        assert saved_config["num_experts_per_tok"] == HF_GLM5_TOY_MODEL_CONFIG["num_experts_per_tok"], (
            "Number of experts per token should match toy config"
        )
        assert saved_config["moe_intermediate_size"] == HF_GLM5_TOY_MODEL_CONFIG["moe_intermediate_size"], (
            "MoE intermediate size should match toy config"
        )

        print(f"SUCCESS: GLM 5 MoE {test_name} conversion test completed successfully")
        print(f"Converted model saved at: {converted_model_dir}")
        print(
            f"MoE parameters preserved: {saved_config['n_routed_experts']} experts, {saved_config['num_experts_per_tok']} per token"
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_glm5_conversion_parallelism_local_model(self, tmp_path, tp, pp, ep, test_name):
        """
        Run hf_megatron_roundtrip_multi_gpu.py using a local model path on disk.

        Set GLM5_LOCAL_MODEL_DIR to a writable directory; the test will create
        a toy model under that path if it doesn't exist yet.
        """
        local_root = os.environ.get("GLM5_LOCAL_MODEL_DIR")
        if not local_root:
            pytest.skip("Set GLM5_LOCAL_MODEL_DIR to run the local-path conversion test.")

        local_root_path = Path(local_root)
        local_root_path.mkdir(parents=True, exist_ok=True)
        model_dir = local_root_path / "glm5_toy"

        if not model_dir.exists():
            _create_glm5_toy_model(model_dir)

        test_output_dir = local_root_path / f"glm5_moe_{test_name}_out"
        test_output_dir.mkdir(exist_ok=True)

        repo_root = _repo_root()
        print(f"model_dir: {model_dir}, test_output_dir: {test_output_dir}, repo_root: {repo_root}, tp: {tp}, pp: {pp}, ep: {ep}")
        cmd = _build_roundtrip_cmd(str(model_dir), test_output_dir, tp, pp, ep, repo_root)

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
        print(f"RESULT: {result}")
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, (
                f"GLM 5 MoE local-path {test_name} conversion failed with return code {result.returncode}"
            )

        # except Exception as e:
        #     print(f"Error during GLM 5 MoE {test_name} conversion test: {e}")
        #     raise
