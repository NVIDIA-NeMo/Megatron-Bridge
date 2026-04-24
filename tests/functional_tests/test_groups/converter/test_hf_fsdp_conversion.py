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
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data(tmp_path_factory):
    """Override conftest fixture: FSDP roundtrip tests don't require pre-downloaded test data."""
    yield tmp_path_factory.mktemp("test_data")


HF_QWEN3_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 262144,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_experts": 4,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "output_router_logits": False,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "router_aux_loss_coef": 0.001,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


class TestHFFSDPConversion:
    """
    Test round-trip conversion between HuggingFace and Megatron-FSDP.
    """

    @pytest.fixture(scope="class")
    def qwen3_moe_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3 MoE toy model to a temporary directory.

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        temp_dir = tmp_path_factory.mktemp("qwen3_moe_fsdp_toy")
        model_dir = temp_dir / "qwen3_moe_fsdp_toy"

        config = Qwen3MoeConfig(**HF_QWEN3_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        model = Qwen3MoeForCausalLM(config).bfloat16()

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        tokenizer.save_pretrained(model_dir)

        model.save_pretrained(model_dir, safe_serialization=True)

        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN3_MOE_TOY_MODEL_CONFIG.copy(), f, indent=2)

        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,ep,nproc,test_name",
        [
            # CI only covers TP=1 cases. TP>1 cases depend on Megatron-LM commit
            # 8cbc45b6e (PR#3191, merged 2026-04-07).
            (1, 1, 1, "FSDP_base"),
        ],
    )
    def test_hf_fsdp_roundtrip(self, qwen3_moe_toy_model_path, tmp_path, tp, ep, nproc, test_name):
        """
        Test HF-to-Megatron-FSDP round-trip conversion with different parallelism configurations.

        Args:
            qwen3_moe_toy_model_path: Path to the toy Qwen3 MoE model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            ep: Expert parallelism size
            nproc: Number of processes for torchrun
            test_name: Name of the test for identification
        """
        test_output_dir = tmp_path / test_name
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--nnodes=1",
            "examples/conversion/fsdp/hf_fsdp_roundtrip.py",
            "--hf-model-id",
            qwen3_moe_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--ep",
            str(ep),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"{test_name} FSDP roundtrip failed with return code {result.returncode}"

            # Verify the converted model directory exists
            model_name = Path(qwen3_moe_toy_model_path).name
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check config.json
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights (single file or sharded)
            weights_found = (converted_model_dir / "model.safetensors").exists() or (
                converted_model_dir / "pytorch_model.bin"
            ).exists()
            if not weights_found:
                sharded = list(converted_model_dir.glob("model-*-of-*.safetensors")) + list(
                    converted_model_dir.glob("pytorch_model-*-of-*.bin")
                )
                weights_found = len(sharded) > 0
            assert weights_found, f"Model weights not found in converted model at {converted_model_dir}"

            # Verify MoE config parameters are preserved
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "qwen3_moe"
            assert saved_config["hidden_size"] == 2048
            assert saved_config["num_attention_heads"] == 32
            num_experts_key = "num_local_experts" if "num_local_experts" in saved_config else "num_experts"
            assert saved_config[num_experts_key] == 4
            assert saved_config["num_experts_per_tok"] == 4
            assert saved_config["moe_intermediate_size"] == 768

            print(f"SUCCESS: {test_name} Megatron-FSDP roundtrip completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during {test_name} Megatron-FSDP roundtrip test: {e}")
            raise
