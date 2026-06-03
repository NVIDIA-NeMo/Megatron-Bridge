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
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM


HF_QWEN3_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 262144,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "num_attention_heads": 16,
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


def _on_disk_tensor_names(converted_model_dir: Path) -> set[str]:
    """Collect the set of tensor names saved on disk (single-file or sharded)."""
    index_file = converted_model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        return set(index["weight_map"].keys())

    names: set[str] = set()
    single_file = converted_model_dir / "model.safetensors"
    shards = sorted(converted_model_dir.glob("model-*-of-*.safetensors"))
    files = [single_file] if single_file.exists() else shards
    for shard in files:
        with safe_open(str(shard), framework="pt") as handle:
            names.update(handle.keys())
    return names


class TestMultiGPUConversion:
    """
    Test multi-GPU conversion from HuggingFace models with different parallelism configurations.
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_conversion_parallelism(self, tmp_path, tp, pp, test_name):
        """
        Test model conversion with different parallelism configurations.

        Args:
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory
        test_output_dir = tmp_path / test_name
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism configuration
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
            "meta-llama/Llama-3.2-1B",
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            converted_model_dir = test_output_dir / "Llama-3.2-1B"
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in converted model at {converted_model_dir}"
            )

            print(f"SUCCESS: {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during {test_name} conversion test: {e}")
            raise

    @pytest.fixture(scope="class")
    def qwen3_moe_toy_model_path(self, tmp_path_factory):
        """Create and save a HuggingFace Qwen3 MoE toy model to a temporary directory."""
        temp_dir = tmp_path_factory.mktemp("qwen3_moe_multi_gpu_toy")
        model_dir = temp_dir / "qwen3_moe_multi_gpu_toy"

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
        "tp,pp,ep,etp,test_name",
        [
            (1, 1, 2, 1, "EP"),
            (1, 2, 2, 1, "PP_EP"),
            (1, 1, 2, 2, "EP_ETP"),
        ],
    )
    def test_moe_conversion_expert_parallelism(self, qwen3_moe_toy_model_path, tmp_path, tp, pp, ep, etp, test_name):
        """
        Test MoE model conversion across expert/expert-tensor parallel configurations.

        Exercises the WORLD-spanning parameter-name gather: expert names live on
        different EP/ETP ranks, so the saved on-disk tensor set must be a superset
        of the expected HuggingFace expert parameter names.

        Args:
            qwen3_moe_toy_model_path: Path to the toy Qwen3 MoE model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            ep: Expert parallelism size
            etp: Expert tensor parallelism size
            test_name: Name of the test for identification
        """
        test_output_dir = tmp_path / test_name
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
            qwen3_moe_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
            "--etp",
            str(etp),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"{test_name} MoE conversion failed with return code {result.returncode}"

            model_name = Path(qwen3_moe_toy_model_path).name
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            num_experts = HF_QWEN3_MOE_TOY_MODEL_CONFIG["num_experts"]
            num_layers = HF_QWEN3_MOE_TOY_MODEL_CONFIG["num_hidden_layers"]
            expected_expert_names = {
                f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                for layer in range(num_layers)
                for expert in range(num_experts)
                for proj in ("gate_proj", "up_proj", "down_proj")
            }

            on_disk_names = _on_disk_tensor_names(converted_model_dir)
            missing = expected_expert_names - on_disk_names
            assert not missing, (
                f"{test_name}: saved tensor set is missing expert weights from the gather path: {sorted(missing)}"
            )

            print(f"SUCCESS: {test_name} MoE conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during {test_name} MoE conversion test: {e}")
            raise
