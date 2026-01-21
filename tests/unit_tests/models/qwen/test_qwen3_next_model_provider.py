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

import torch
import torch.nn.functional as F

from megatron.bridge.models.qwen import (
    Qwen3NextModelProvider,
    Qwen3NextModelProvider80B_A3B,
)


class TestQwen3NextModelProvider:
    """Test cases for base Qwen3NextModelProvider class."""

    def test_qwen3_next_model_provider_initialization(self):
        """Test Qwen3NextModelProvider can be initialized with default values."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

        # Check Qwen3 MoE-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True
        assert provider.kv_channels == 256
        assert provider.num_query_groups == 2
        assert provider.seq_length == 262144  # 256k tokens
        assert provider.init_method_std == 0.02
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.vocab_size == 151936
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == 10000000.0
        assert provider.rotary_percent == 0.25
        assert provider.attention_output_gate is True
        assert provider.position_embedding_type == "rope"
        assert provider.autocast_dtype == torch.bfloat16
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True

        # Check MoE-specific defaults
        assert provider.num_moe_experts == 512
        assert provider.moe_router_load_balancing_type == "global_aux_loss"
        assert provider.moe_aux_loss_coeff == 1e-3
        assert provider.moe_router_topk == 10
        assert provider.moe_router_pre_softmax is False
        assert provider.moe_grouped_gemm is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_permute_fusion is True
        assert provider.moe_shared_expert_gate is True
        assert provider.moe_router_dtype == "fp32"

        # Linear Attention specific defaults
        assert provider.experimental_attention_variant == "gated_delta_net"
        assert provider.linear_attention_freq == 4
        assert provider.linear_conv_kernel_dim == 4
        assert provider.linear_key_head_dim == 128
        assert provider.linear_value_head_dim == 128
        assert provider.linear_num_key_heads == 16
        assert provider.linear_num_value_heads == 32

        # Checkpointing specific defaults
        assert provider.hetereogenous_dist_checkpoint is True

    def test_qwen3_next_model_provider_with_custom_moe_config(self):
        """Test Qwen3NextModelProvider with custom MoE configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_moe_experts=64,
            moe_router_topk=4,
            moe_aux_loss_coeff=1e-2,
            moe_ffn_hidden_size=256,
            ffn_hidden_size=1024,
            moe_shared_expert_gate=False,
        )

        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 4
        assert provider.moe_aux_loss_coeff == 1e-2
        assert provider.moe_ffn_hidden_size == 256
        assert provider.ffn_hidden_size == 1024
        assert provider.moe_shared_expert_gate is False

    def test_qwen3_next_model_provider_with_custom_linear_attention_config(self):
        """Test Qwen3NextModelProvider with custom Linear Attention configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=2,
            linear_conv_kernel_dim=8,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=8,
            linear_num_value_heads=16,
        )

        assert provider.experimental_attention_variant == "gated_delta_net"
        assert provider.linear_attention_freq == 2
        assert provider.linear_conv_kernel_dim == 8
        assert provider.linear_key_head_dim == 64
        assert provider.linear_value_head_dim == 64
        assert provider.linear_num_key_heads == 8
        assert provider.linear_num_value_heads == 16

    def test_qwen3_next_model_provider_with_custom_rope(self):
        """Test Qwen3NextModelProvider with custom RoPE configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
            rotary_percent=0.5,
        )

        assert provider.rotary_base == 500000.0
        assert provider.rotary_percent == 0.5

    def test_qwen3_next_model_provider_ffn_hidden_size(self):
        """Test Qwen3NextModelProvider FFN hidden size configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
            moe_ffn_hidden_size=1536,
            moe_shared_expert_intermediate_size=1024,
        )

        assert provider.ffn_hidden_size == 11008
        assert provider.moe_ffn_hidden_size == 1536
        assert provider.moe_shared_expert_intermediate_size == 1024

    def test_qwen3_next_model_provider_group_query_attention(self):
        """Test Qwen3NextModelProvider with group query attention."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=4,
        )

        assert provider.num_query_groups == 4

    def test_qwen3_next_model_provider_custom_vocab_size(self):
        """Test Qwen3NextModelProvider with custom vocabulary size."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
        )

        assert provider.vocab_size == 32000

    def test_qwen3_next_model_provider_custom_sequence_length(self):
        """Test Qwen3NextModelProvider with custom sequence length."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
        )

        assert provider.seq_length == 8192

    def test_qwen3_next_model_provider_qk_layernorm(self):
        """Test Qwen3NextModelProvider has Zero-Centered QK layernorm enabled by default."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert provider.qk_layernorm is True
        assert provider.layernorm_zero_centered_gamma is True

    def test_qwen3_next_model_provider_dtype_configuration(self):
        """Test Qwen3NextModelProvider dtype configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            fp16=True,
            bf16=False,
            params_dtype=torch.float16,
        )

        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    def test_qwen3_next_model_provider_custom_mtp_configuration(self):
        """Test Qwen3NextModelProvider with custom MTP configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            mtp_num_layers=2,
            mtp_loss_scaling_factor=0.2,
        )
        assert provider.mtp_num_layers == 2
        assert provider.mtp_loss_scaling_factor == 0.2


class TestQwen3NextModelProvider80B_A3B:
    """Test cases for Qwen3NextModelProvider80B_A3B class."""

    def test_qwen3_next_80b_a3b_default_configuration(self):
        """Test Qwen3 Next 80B-A3B model has correct default configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        # Check Qwen3 Next 80B-A3B specific configuration
        assert provider.num_layers == 48
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 2
        assert provider.ffn_hidden_size == 5120
        assert provider.moe_ffn_hidden_size == 512
        assert provider.moe_shared_expert_intermediate_size == 512

        # Check inherited MoE defaults
        assert provider.num_moe_experts == 512
        assert provider.moe_router_topk == 10
        assert provider.moe_shared_expert_gate is True
        assert provider.moe_router_dtype == "fp32"
        assert provider.moe_router_pre_softmax is False

        # Check inherited Linear Attention defaults
        assert provider.experimental_attention_variant == "gated_delta_net"
        assert provider.linear_attention_freq == 4
        assert provider.linear_conv_kernel_dim == 4
        assert provider.linear_key_head_dim == 128
        assert provider.linear_value_head_dim == 128
        assert provider.linear_num_key_heads == 16
        assert provider.linear_num_value_heads == 32

        # Check inherited base defaults
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.qk_layernorm is True
        assert provider.kv_channels == 256
        assert provider.vocab_size == 151936
        assert provider.seq_length == 262144  # 256k tokens
        assert provider.rotary_base == 10000000.0
        assert provider.rotary_percent == 0.25
        assert provider.attention_output_gate is True
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.hetereogenous_dist_checkpoint is True

    def test_qwen3_next_80b_a3b_override_configuration(self):
        """Test Qwen3 Next 80B-A3B model with overridden configuration."""
        provider = Qwen3NextModelProvider80B_A3B(
            seq_length=8192,
            num_moe_experts=64,
            moe_router_topk=4,
        )

        # Check overridden values
        assert provider.seq_length == 8192
        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 4

        # Check defaults remain
        assert provider.num_layers == 48
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 2
        assert provider.ffn_hidden_size == 5120
        assert provider.moe_ffn_hidden_size == 512
        assert provider.moe_shared_expert_intermediate_size == 512
        assert provider.moe_router_pre_softmax is False


class TestQwen3NextProviderInheritance:
    """Test inheritance relationships between Qwen3 Next providers."""

    def test_qwen3_next_models_inherit_from_base(self):
        """Test Qwen3 Next providers inherit from Qwen3NextModelProvider."""
        assert issubclass(Qwen3NextModelProvider80B_A3B, Qwen3NextModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Qwen3 Next 80B-A3B
        provider = Qwen3NextModelProvider80B_A3B()

        # The provide method should be inherited from GPTModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)


class TestQwen3NextProviderEdgeCases:
    """Test edge cases and error conditions for Qwen3 Next providers."""

    def test_valid_num_query_groups(self):
        """Test that valid num_query_groups configuration works."""
        # num_attention_heads must be divisible by num_query_groups
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,  # 32 divisible by 8
        )
        assert provider.num_query_groups == 8

    def test_valid_linear_attention_num_key_heads(self):
        """Test that valid linear_num_key_heads configuration works."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            linear_num_value_heads=64,
            linear_num_key_heads=8,  # 64 divisible by 8
        )
        assert provider.linear_num_key_heads == 8

    def test_valid_linear_attention_frequency(self):
        """Test that valid linear_attention_frequency configuration works."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            linear_attention_freq=2,  # 32 divisible by 2
        )
        assert provider.linear_attention_freq == 2

    def test_valid_linear_attention_frequency_list(self):
        """Test that valid linear_attention_frequency configuration works."""
        provider = Qwen3NextModelProvider(
            num_layers=8,
            hidden_size=4096,
            linear_attention_freq=[1, 1, 1, 0, 1, 1, 1, 0],  # 8 layers, 6 linear attention layers
        )
        assert provider.linear_attention_freq == [1, 1, 1, 0, 1, 1, 1, 0]

    def test_moe_configuration_validity(self):
        """Test MoE configuration parameters."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_moe_experts=64,
            moe_router_topk=4,
        )

        # moe_router_topk should be <= num_moe_experts
        assert provider.moe_router_topk <= provider.num_moe_experts
        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 4

    def test_vocabulary_size_divisibility(self):
        """Test vocabulary size divisibility configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=152064,
            make_vocab_size_divisible_by=128,
        )

        # The actual vocab size should be adjusted if needed
        assert provider.make_vocab_size_divisible_by == 128

    def test_seq_length_override(self):
        """Test sequence length configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
        )

        assert provider.seq_length == 8192

    def test_rotary_base_configuration(self):
        """Test rotary base configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
        )

        assert provider.rotary_base == 500000.0

    def test_layernorm_epsilon_override(self):
        """Test layernorm epsilon configuration."""
        provider = Qwen3NextModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            layernorm_epsilon=1e-5,
        )

        assert provider.layernorm_epsilon == 1e-5


class TestQwen3NextProviderArchitecturalFeatures:
    """Test cases to verify Qwen3 MoE architectural features."""

    def test_qwen3_next_qk_layernorm_feature(self):
        """Test that Qwen3 Next models have Zero-Centered QK layernorm enabled."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.qk_layernorm is True
        assert provider.layernorm_zero_centered_gamma is True

    def test_qwen3_next_dtype_defaults(self):
        """Test that Qwen3 Next models have correct dtype defaults."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.autocast_dtype == torch.bfloat16
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True

    def test_qwen3_next_kv_channels(self):
        """Test that Qwen3 Next models have correct KV channels configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.kv_channels == 256

    def test_qwen3_next_rope_percent(self):
        """Test that Qwen3 Next models have correct RoPE percent configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.rotary_percent == 0.25

    def test_qwen3_next_bias_configuration(self):
        """Test that Qwen3 Next models have correct bias configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False

    def test_qwen3_next_attention_output_gate(self):
        """Test that Qwen3 Next models have correct attention output gate configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.attention_output_gate is True

    def test_qwen3_next_shared_expert(self):
        """Test that Qwen3 Next models have correct shared expert configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.moe_shared_expert_gate is True
        assert provider.moe_shared_expert_intermediate_size == 512

    def test_qwen3_next_linear_attention_type(self):
        """Test that Qwen3 Next models have correct linear attention type (Gated Delta Net) configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.experimental_attention_variant == "gated_delta_net"

    def test_qwen3_next_linear_attention_freq(self):
        """Test that Qwen3 Next models have correct linear attention frequency configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.linear_attention_freq == 4

    def test_qwen3_next_moe_router_load_balancing_type(self):
        """Test that Qwen3 Next models have correct Moe Router Load Balancing Type configuration."""
        provider = Qwen3NextModelProvider80B_A3B()

        assert provider.moe_router_load_balancing_type == "global_aux_loss"
