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
from transformers import GptOssConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


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
        assert isinstance(provider, GPTModelProvider)
        # Key fields mapped from HF config
        assert provider.num_layers == mock_pretrained.config.num_hidden_layers
        assert provider.num_moe_experts == mock_pretrained.config.num_local_experts
        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_autobridge_model_config_preserves_yarn_fields(self):
        config = GptOssConfig(architectures=["GptOssForCausalLM"])
        bridge = AutoBridge.from_hf_config(config)

        with pytest.warns(FutureWarning, match="get_model_config"):
            provider = bridge.to_megatron_provider(load_weights=False)
        model_config = bridge.get_model_config()

        aligned_fields = (
            "normalization",
            "gated_linear_unit",
            "add_bias_linear",
            "add_qkv_bias",
            "share_embeddings_and_output_weights",
            "position_embedding_type",
            "moe_router_pre_softmax",
            "moe_grouped_gemm",
            "moe_token_dispatcher_type",
            "moe_permute_fusion",
            "moe_router_load_balancing_type",
            "bias_activation_fusion",
            "bias_dropout_fusion",
            "hidden_dropout",
            "fp16",
            "bf16",
            "params_dtype",
            "activation_func",
            "activation_func_clamp_value",
            "glu_linear_offset",
            "softmax_type",
            "window_size",
            "window_attn_skip_freq",
            "moe_ffn_hidden_size",
        )
        assert {name: getattr(model_config, name) for name in aligned_fields} == {
            name: getattr(provider, name) for name in aligned_fields
        }

        expected_yarn_fields = {
            "yarn_rotary_scaling_factor": 32.0,
            "yarn_original_max_position_embeddings": 4096,
            "yarn_beta_fast": 32.0,
            "yarn_beta_slow": 1.0,
            "yarn_mscale": None,
            "yarn_mscale_all_dim": None,
            "yarn_correction_range_round_to_int": False,
        }
        assert {name: getattr(provider, name) for name in expected_yarn_fields} == expected_yarn_fields
        assert {name: getattr(model_config.transformer, name) for name in expected_yarn_fields} == expected_yarn_fields
        assert all(name not in model_config.__dict__ for name in expected_yarn_fields)
