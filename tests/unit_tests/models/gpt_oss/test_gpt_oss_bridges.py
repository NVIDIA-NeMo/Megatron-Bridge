#!/usr/bin/env python3
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

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider


class TestGptOssBridge:
    """Unit tests for GPT-OSS bridge provider mapping."""

    @pytest.fixture
    def gpt_oss_cfg(self):
        return {
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_local_experts": 32,
            "torch_dtype": "bfloat16",
            "vocab_size": 201088,
            "hidden_act": "silu",
            "sliding_window": 4096,
        }

    @pytest.fixture
    def mock_pretrained(self, gpt_oss_cfg):
        # Use spec to prevent Mock from auto-creating undefined attributes
        cfg = Mock(spec=list(gpt_oss_cfg.keys()))
        for k, v in gpt_oss_cfg.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock()
        return m

    def test_registration(self):
        assert issubclass(GPTOSSBridge, MegatronModelBridge)

    def test_provider_bridge_maps_config(self, mock_pretrained):
        bridge = GPTOSSBridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, HybridModelProvider)
        # Key fields mapped from HF config
        assert provider.num_layers == 2 * mock_pretrained.config.num_hidden_layers
        assert provider.num_moe_experts == mock_pretrained.config.num_local_experts
        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16
        assert provider.hybrid_stack_spec is None
        assert provider.hybrid_layer_pattern == "*E" * mock_pretrained.config.num_hidden_layers
        assert provider.window_attn_skip_freq[:4] == [True, False, False, False]
        assert provider.position_embedding_type == "yarn"
        assert provider.yarn_rotary_scaling_factor == 32.0
        assert provider.yarn_original_max_position_embeddings == 4096
        assert provider.yarn_beta_fast == 32.0
        assert provider.yarn_beta_slow == 1.0
        assert provider.yarn_correction_range_round_to_int is False
        assert provider.yarn_mscale is None
        assert provider.yarn_mscale_all_dim is None

    def test_mapping_registry_splits_hf_layer_across_attention_and_moe_layers(self, mock_pretrained):
        bridge = GPTOSSBridge()
        bridge.hf_config = mock_pretrained.config
        registry = bridge.mapping_registry()

        attention_mapping = registry.hf_to_megatron_lookup("model.layers.3.self_attn.o_proj.weight")
        moe_mapping = registry.hf_to_megatron_lookup("model.layers.3.post_attention_layernorm.weight")
        expert_mapping = registry.hf_to_megatron_lookup("model.layers.3.mlp.experts.down_proj")

        assert attention_mapping.megatron_param == "decoder.layers.6.self_attention.linear_proj.weight"
        assert moe_mapping.megatron_param == "decoder.layers.7.pre_mlp_layernorm.weight"
        assert expert_mapping.megatron_param == "decoder.layers.7.mlp.experts.linear_fc2.weight*"

    def test_mapping_registry_resolves_hybrid_final_norm(self):
        registry = GPTOSSBridge().mapping_registry()

        final_norm_mapping = registry.megatron_to_hf_lookup("decoder.final_norm.weight")
        final_layernorm_mapping = registry.megatron_to_hf_lookup("decoder.final_layernorm.weight")

        assert final_norm_mapping.hf_param == "model.norm.weight"
        assert final_layernorm_mapping.hf_param == "model.norm.weight"
