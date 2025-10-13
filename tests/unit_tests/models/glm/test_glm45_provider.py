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
Unit tests for GLM 4.5 provider classes.
"""

import torch
import torch.nn.functional as F

from megatron.bridge.models.glm.glm45_provider import (
    GLM45AirModelProvider106B,
    GLM45ModelProvider355B,
    GLMMoEModelProvider,
)


class TestGLM45ProviderDefaults:
    """Test default configuration values for GLM 4.5 providers."""

    def test_glm_moe_provider_base_defaults(self):
        # Provide minimal valid values to satisfy Megatron-Core post-init checks
        provider = GLMMoEModelProvider(num_layers=1, hidden_size=1024)

        # Generic model defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is True
        assert provider.position_embedding_type == "rope"
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5

        # Sequence and vocab defaults
        assert provider.seq_length == 131072
        assert provider.vocab_size == 151552
        assert provider.init_method_std == 0.02
        assert provider.hidden_dropout == 0.0

        # DType defaults
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16
        assert provider.autocast_dtype == torch.bfloat16

        # Attention defaults
        assert provider.num_query_groups == 8
        assert provider.num_attention_heads == 96
        assert provider.attention_dropout == 0.0
        assert provider.kv_channels == 128

        # RoPE defaults
        assert provider.rotary_base == 1000000.0
        assert provider.rotary_percent == 0.5

        # MoE specific parameters
        assert provider.moe_router_topk == 8
        assert provider.moe_shared_expert_overlap is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_router_load_balancing_type == "seq_aux_loss"
        assert provider.moe_aux_loss_coeff == 1e-3
        assert provider.moe_router_pre_softmax is False
        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_permute_fusion is True
        assert provider.moe_router_dtype == "fp32"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_router_bias_update_rate == 0

        # Optimization defaults
        assert provider.persist_layer_norm is True
        assert provider.bias_activation_fusion is True
        assert provider.bias_dropout_fusion is True

    def test_glm45_355b_defaults(self):
        provider = GLM45ModelProvider355B()

        assert provider.num_layers == 92
        assert provider.num_moe_experts == 160
        assert provider.hidden_size == 5120
        assert provider.ffn_hidden_size == 12288
        assert provider.moe_ffn_hidden_size == 1536
        assert provider.moe_shared_expert_intermediate_size == 1536
        assert provider.qk_layernorm is True
        assert provider.moe_router_topk_scaling_factor == 2.5

        # Test moe_layer_freq (first 3 layers are dense, rest are MoE)
        expected_freq = [0] * 3 + [1] * 89
        assert provider.moe_layer_freq == expected_freq

    def test_glm45_air_106b_defaults(self):
        provider = GLM45AirModelProvider106B()

        assert provider.num_layers == 46
        assert provider.num_moe_experts == 128
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 10944
        assert provider.moe_ffn_hidden_size == 1408
        assert provider.moe_shared_expert_intermediate_size == 1408
        assert provider.qk_layernorm is False

        # Test moe_layer_freq (first 1 layer is dense, rest are MoE)
        expected_freq = [0] * 1 + [1] * 45
        assert provider.moe_layer_freq == expected_freq
