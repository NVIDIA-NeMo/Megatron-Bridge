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

"""
Unit tests for DeepSeek provider classes.
"""

import torch

from megatron.bridge.models.deepseek.deepseek_provider import (
    DeepSeekProvider,
    DeepSeekV2LiteProvider,
    DeepSeekV2Provider,
    DeepSeekV3Provider,
    MoonlightProvider,
)


class TestDeepSeekProviderDefaults:
    """Test default configuration values for DeepSeek providers."""

    def test_deepseek_provider_base_defaults(self):
        # Provide minimal valid values to satisfy Megatron-Core post-init checks
        provider = DeepSeekProvider(num_layers=1, hidden_size=1024, num_attention_heads=8)

        # Generic model defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is not None
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.share_embeddings_and_output_weights is False
        assert provider.qk_layernorm is True

        # DType defaults
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

        # MoE and MLA flags
        assert provider.moe_grouped_gemm is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.q_lora_rank is not None
        assert provider.kv_lora_rank is not None

    def test_deepseek_v2_defaults(self):
        provider = DeepSeekV2Provider()

        assert provider.num_layers == 60
        assert provider.hidden_size == 5120
        assert provider.num_moe_experts == 160
        assert provider.moe_router_topk == 6
        assert provider.qk_layernorm is True
        assert provider.mscale == 0.707
        assert provider.mscale_all_dim == 0.707

    def test_deepseek_v2_lite_defaults(self):
        provider = DeepSeekV2LiteProvider()

        assert provider.num_layers == 27
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 16
        assert provider.num_moe_experts == 64
        assert provider.q_lora_rank is None
        assert provider.mscale == 0.707
        assert provider.mscale_all_dim == 0.707

    def test_deepseek_v3_defaults(self):
        provider = DeepSeekV3Provider()

        assert provider.num_layers == 61
        assert provider.hidden_size == 7168
        assert provider.num_moe_experts == 256
        assert provider.moe_router_topk == 8
        assert provider.kv_channels == 128
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_router_bias_update_rate == 1e-3
        assert provider.mscale == 1.0
        assert provider.mscale_all_dim == 1.0

    def test_moonlight_defaults(self):
        provider = MoonlightProvider()

        assert provider.num_layers == 27
        assert provider.hidden_size == 2048
        assert provider.ffn_hidden_size == 11264
        assert provider.num_moe_experts == 64
        assert provider.moe_ffn_hidden_size == 1408
        assert provider.moe_router_topk == 6
        assert provider.moe_router_num_groups == 1
        assert provider.moe_router_group_topk == 1
        assert provider.rotary_base == 50000
        assert provider.layernorm_epsilon == 1e-5
        assert provider.q_lora_rank is None
        assert provider.mscale == 1.0
        assert provider.mscale_all_dim == 1.0
