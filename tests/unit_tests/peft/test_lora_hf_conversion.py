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

"""Tests for HF PEFT LoRA to Megatron LoRA weight conversion.

The convert_hf_lora_to_megatron utility converts HuggingFace PEFT-format LoRA
adapter weights to Megatron Bridge's internal LoRA format, handling QKV fusion,
gate/up fusion, per-expert grouping, and router LoRA mapping.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion.peft_bridge import convert_hf_lora_to_megatron


@pytest.fixture
def model_config():
    return SimpleNamespace(
        hidden_size=128,
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=16,
        attention_output_gate=False,
        num_moe_experts=None,
    )


class TestFuseQKVLoRA:
    def test_qkv_fusion_basic(self, model_config):
        rank = 8
        shared_lora_a = torch.randn(rank, model_config.hidden_size)
        q_lora_b = torch.randn(model_config.hidden_size, rank)
        k_lora_b = torch.randn(model_config.num_query_groups * model_config.kv_channels, rank)
        v_lora_b = torch.randn(model_config.num_query_groups * model_config.kv_channels, rank)

        hf_weights = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": shared_lora_a,
            "model.layers.0.self_attn.k_proj.lora_A.weight": shared_lora_a.clone(),
            "model.layers.0.self_attn.v_proj.lora_A.weight": shared_lora_a.clone(),
            "model.layers.0.self_attn.q_proj.lora_B.weight": q_lora_b,
            "model.layers.0.self_attn.k_proj.lora_B.weight": k_lora_b,
            "model.layers.0.self_attn.v_proj.lora_B.weight": v_lora_b,
        }

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        assert "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight" in result
        assert "decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight" in result

        # linear_in should be the shared lora_A
        assert torch.equal(
            result["decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"],
            shared_lora_a,
        )

        # linear_out should have the fused Q+K+V interleaved per query group
        fused_b = result["decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight"]
        expected_out_dim = q_lora_b.shape[0] + k_lora_b.shape[0] + v_lora_b.shape[0]
        assert fused_b.shape == (expected_out_dim, rank)

    def test_qkv_fusion_rejects_different_lora_a(self, model_config):
        rank = 8
        hf_weights = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, model_config.hidden_size),
            "model.layers.0.self_attn.k_proj.lora_A.weight": torch.randn(rank, model_config.hidden_size),
            "model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(rank, model_config.hidden_size),
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(model_config.hidden_size, rank),
            "model.layers.0.self_attn.k_proj.lora_B.weight": torch.randn(32, rank),
            "model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(32, rank),
        }
        with pytest.raises(ValueError, match="identical lora_A"):
            convert_hf_lora_to_megatron(hf_weights, model_config)


class TestFuseGateUpLoRA:
    def test_gate_up_fusion(self, model_config):
        rank = 8
        ffn_dim = 256
        shared_lora_a = torch.randn(rank, model_config.hidden_size)
        gate_lora_b = torch.randn(ffn_dim, rank)
        up_lora_b = torch.randn(ffn_dim, rank)

        hf_weights = {
            "model.layers.0.mlp.gate_proj.lora_A.weight": shared_lora_a,
            "model.layers.0.mlp.up_proj.lora_A.weight": shared_lora_a.clone(),
            "model.layers.0.mlp.gate_proj.lora_B.weight": gate_lora_b,
            "model.layers.0.mlp.up_proj.lora_B.weight": up_lora_b,
        }

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        assert "decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight" in result
        assert "decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight" in result

        fused_b = result["decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight"]
        assert fused_b.shape == (2 * ffn_dim, rank)
        assert torch.equal(fused_b[:ffn_dim], gate_lora_b)
        assert torch.equal(fused_b[ffn_dim:], up_lora_b)


class TestDirectMappings:
    def test_o_proj_mapping(self, model_config):
        rank = 8
        lora_a = torch.randn(rank, model_config.hidden_size)
        lora_b = torch.randn(model_config.hidden_size, rank)

        hf_weights = {
            "model.layers.0.self_attn.o_proj.lora_A.weight": lora_a,
            "model.layers.0.self_attn.o_proj.lora_B.weight": lora_b,
        }

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        assert "decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight" in result
        assert "decoder.layers.0.self_attention.linear_proj.adapter.linear_out.weight" in result
        assert torch.equal(result["decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight"], lora_a)

    def test_down_proj_mapping(self, model_config):
        rank = 8
        ffn_dim = 256
        lora_a = torch.randn(rank, ffn_dim)
        lora_b = torch.randn(model_config.hidden_size, rank)

        hf_weights = {
            "model.layers.0.mlp.down_proj.lora_A.weight": lora_a,
            "model.layers.0.mlp.down_proj.lora_B.weight": lora_b,
        }

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        assert "decoder.layers.0.mlp.linear_fc2.adapter.linear_in.weight" in result
        assert "decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight" in result


class TestExpertLoRA:
    def test_expert_gate_up_fusion(self, model_config):
        rank = 4
        ffn_dim = 64
        num_experts = 4
        model_config.num_moe_experts = num_experts

        hf_weights = {}
        for i in range(num_experts):
            shared_a = torch.randn(rank, model_config.hidden_size)
            hf_weights[f"model.layers.0.mlp.experts.{i}.gate_proj.lora_A.weight"] = shared_a
            hf_weights[f"model.layers.0.mlp.experts.{i}.up_proj.lora_A.weight"] = shared_a.clone()
            hf_weights[f"model.layers.0.mlp.experts.{i}.gate_proj.lora_B.weight"] = torch.randn(ffn_dim, rank)
            hf_weights[f"model.layers.0.mlp.experts.{i}.up_proj.lora_B.weight"] = torch.randn(ffn_dim, rank)

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        fc1_in = result["decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight"]
        fc1_out = result["decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_out.weight"]

        assert fc1_in.shape == (num_experts, rank, model_config.hidden_size)
        assert fc1_out.shape == (num_experts, 2 * ffn_dim, rank)

    def test_expert_down_proj(self, model_config):
        rank = 4
        ffn_dim = 64
        num_experts = 4
        model_config.num_moe_experts = num_experts

        hf_weights = {}
        for i in range(num_experts):
            hf_weights[f"model.layers.0.mlp.experts.{i}.down_proj.lora_A.weight"] = torch.randn(rank, ffn_dim)
            hf_weights[f"model.layers.0.mlp.experts.{i}.down_proj.lora_B.weight"] = torch.randn(
                model_config.hidden_size, rank
            )

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        fc2_in = result["decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight"]
        fc2_out = result["decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight"]

        assert fc2_in.shape == (num_experts, rank, ffn_dim)
        assert fc2_out.shape == (num_experts, model_config.hidden_size, rank)


class TestRouterLoRA:
    def test_router_lora_mapping(self, model_config):
        rank = 4
        num_experts = 8
        lora_a = torch.randn(rank, model_config.hidden_size)
        lora_b = torch.randn(num_experts, rank)

        hf_weights = {
            "model.layers.0.mlp.gate.lora_A.weight": lora_a,
            "model.layers.0.mlp.gate.lora_B.weight": lora_b,
        }

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        assert "decoder.layers.0.mlp.router.adapter.linear_in.weight" in result
        assert "decoder.layers.0.mlp.router.adapter.linear_out.weight" in result
        assert torch.equal(result["decoder.layers.0.mlp.router.adapter.linear_in.weight"], lora_a)
        assert torch.equal(result["decoder.layers.0.mlp.router.adapter.linear_out.weight"], lora_b)


class TestMultipleLayersConversion:
    def test_two_layers(self, model_config):
        rank = 8
        hf_weights = {}
        for layer_idx in range(2):
            shared_a = torch.randn(rank, model_config.hidden_size)
            for proj in ("q_proj", "k_proj", "v_proj"):
                suffix = "self_attn"
                hf_weights[f"model.layers.{layer_idx}.{suffix}.{proj}.lora_A.weight"] = shared_a.clone()
                out_dim = (
                    model_config.hidden_size
                    if proj == "q_proj"
                    else model_config.num_query_groups * model_config.kv_channels
                )
                hf_weights[f"model.layers.{layer_idx}.{suffix}.{proj}.lora_B.weight"] = torch.randn(out_dim, rank)

        result = convert_hf_lora_to_megatron(hf_weights, model_config)

        for layer_idx in range(2):
            assert f"decoder.layers.{layer_idx}.self_attention.linear_qkv.adapter.linear_in.weight" in result
            assert f"decoder.layers.{layer_idx}.self_attention.linear_qkv.adapter.linear_out.weight" in result
