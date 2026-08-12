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
from transformers import (
    AutoTokenizer,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
)


HF_DEEPSEEK_V3_TOY_MODEL_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "kv_lora_rank": 512,
    "max_position_embeddings": 163840,
    "moe_intermediate_size": 768,
    "n_group": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_attention_heads": 32,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 0,
    "q_lora_rank": 512,
    "topk_group": 2,
    "vocab_size": 129280,
    "torch_dtype": "bfloat16",
}


# The fixture above is the control. Without a query LoRA the HF architecture builds a bare
# `q_proj` and defines no query-side norm, which is the branch this PR corrects.
HF_DEEPSEEK_V3_NO_Q_LORA_CONFIG = {
    **HF_DEEPSEEK_V3_TOY_MODEL_CONFIG,
    "q_lora_rank": None,
}


class TestDeepSeekConversion:
    """Functional tests for DeepSeek toy conversion paths."""

    @pytest.fixture(scope="class")
    def deepseek_toy_model_path(self, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("deepseek_toy_model")
        model_dir = temp_dir / "deepseek_toy"

        # Create DeepSeek V3 config from the toy model config
        config = DeepseekV3Config(**HF_DEEPSEEK_V3_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        model = DeepseekV3ForCausalLM(config)
        model = model.bfloat16()

        # Save a tokenizer (use a lightweight compatible tokenizer)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            pass

        # Save model and config
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_deepseek_conversion_parallelism(self, deepseek_toy_model_path, tmp_path, tp, pp, ep, test_name):
        test_output_dir = tmp_path / f"deepseek_{test_name}"
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
            f"--data-file={Path(__file__).resolve().parents[5] / '.coverage'}",
            f"--source={Path(__file__).resolve().parents[5]}",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            deepseek_toy_model_path,
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
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        assert result.returncode == 0, f"DeepSeek {test_name} conversion failed with {result.returncode}"

        # Verify outputs
        model_name = Path(deepseek_toy_model_path).name
        converted_dir = test_output_dir / model_name
        assert converted_dir.exists()

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        weights_file_safetensors = converted_dir / "model.safetensors"
        weights_file_pytorch = converted_dir / "pytorch_model.bin"
        weights_found = weights_file_safetensors.exists() or weights_file_pytorch.exists()
        if not weights_found:
            shards_st = list(converted_dir.glob("model-*-of-*.safetensors"))
            shards_pt = list(converted_dir.glob("pytorch_model-*-of-*.bin"))
            weights_found = len(shards_st) > 0 or len(shards_pt) > 0
        assert weights_found

        with open(config_file) as f:
            saved = json.load(f)

        assert saved["model_type"] == "deepseek_v3", "Model type should be deepseek_v3"
        assert saved["vocab_size"] == 129280
        assert saved["hidden_size"] == 2048
        assert saved["n_routed_experts"] == 8
        assert saved["num_experts_per_tok"] == 4
        assert saved["num_hidden_layers"] == 2
        assert saved["moe_intermediate_size"] == 768

        print(f"SUCCESS: DeepSeek {test_name} conversion test completed successfully")
        print(f"Converted model saved at: {converted_dir}")
        print(
            f"MoE parameters preserved: {saved['n_routed_experts']} experts, {saved['num_experts_per_tok']} per token"
        )

    @pytest.mark.run_only_on("GPU")
    def test_deepseek_v3_autoconfig_roundtrip(self, deepseek_toy_model_path, tmp_path):
        from tests.functional_tests.utils import autoconfig_roundtrip

        autoconfig_roundtrip(deepseek_toy_model_path, tmp_path)


class TestDeepSeekWithoutQueryLoRA:
    """Cover the `q_lora_rank=None` branch end to end, not only at the spec level."""

    @pytest.fixture(scope="class")
    def deepseek_no_q_lora_model_path(self, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("deepseek_no_q_lora_model")
        model_dir = temp_dir / "deepseek_no_q_lora"

        config = DeepseekV3Config(**HF_DEEPSEEK_V3_NO_Q_LORA_CONFIG)
        config.torch_dtype = torch.bfloat16

        torch.manual_seed(1234)
        model = DeepseekV3ForCausalLM(config).bfloat16()
        model.save_pretrained(model_dir, safe_serialization=True)

        with open(model_dir / "config.json", "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    def test_state_dict_round_trip_without_a_query_norm(self, deepseek_no_q_lora_model_path):
        """HF to Megatron to HF must preserve every weight and invent none.

        The structural assertions come from the HF architecture rather than from the
        conversion registry, so the registry cannot act as its own oracle. HF has
        `q_proj.weight` and `kv_a_layernorm.weight` and no query LoRA; Megatron must have
        `linear_q_proj.weight` and `linear_kv_up_proj.layer_norm_weight` and no
        `linear_q_proj.layer_norm_weight`, which is the phantom parameter this PR removes.
        """
        from safetensors.torch import load_file

        from megatron.bridge import AutoBridge

        hf_state = load_file(Path(deepseek_no_q_lora_model_path) / "model.safetensors")
        assert "model.layers.1.self_attn.q_proj.weight" in hf_state
        assert "model.layers.1.self_attn.kv_a_layernorm.weight" in hf_state
        assert not [key for key in hf_state if "q_a_proj" in key or "q_a_layernorm" in key]

        # `get_model` is the builder-backed path and DeepSeek has not migrated to it; its
        # config conversion rejects the MLA fields before any weight handling happens.
        bridge = AutoBridge.from_hf_pretrained(deepseek_no_q_lora_model_path, torch_dtype=torch.bfloat16)
        model = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)
        try:
            megatron_names = {name for stage in model for name, _ in stage.named_parameters()}
            assert any(name.endswith("self_attention.linear_q_proj.weight") for name in megatron_names)
            assert any(name.endswith("self_attention.linear_kv_up_proj.layer_norm_weight") for name in megatron_names)
            assert not [name for name in megatron_names if "linear_q_proj.layer_norm_weight" in name]

            exported = {name: tensor.cpu() for name, tensor in bridge.export_hf_weights(model, cpu=True)}
        finally:
            _teardown_distributed()

        assert not [key for key in exported if "q_a_proj" in key or "q_a_layernorm" in key]
        assert not sorted(set(hf_state) - set(exported)), "export dropped weights"
        mismatched = sorted(key for key in hf_state if not torch.equal(exported[key], hf_state[key]))
        assert not mismatched, f"round trip changed {len(mismatched)} weights, e.g. {mismatched[:3]}"


def _teardown_distributed():
    """Release the standalone process groups `get_model` sets up for a single process."""
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
