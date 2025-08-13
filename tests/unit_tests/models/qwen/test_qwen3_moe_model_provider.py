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
    Qwen3MoEModelProvider,
    Qwen3MoEModelProvider30B_A3B,
    Qwen3MoEModelProvider235B_A22B,
)


class TestQwen3MoEModelProvider:
    """Test cases for base Qwen3MoEModelProvider class."""

    def test_qwen3_moe_model_provider_initialization(self):
        """Test Qwen3MoEModelProvider can be initialized with default values."""
        provider = Qwen3MoEModelProvider(
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
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True  # Qwen3 uses QK layernorm
        assert provider.kv_channels == 128
        assert provider.num_query_groups == 8
        assert provider.seq_length == 40960  # Extended context for Qwen3
        assert provider.init_method_std == 0.02
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.vocab_size == 151936
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == 1000000.0
        assert provider.position_embedding_type == "rope"
        assert provider.autocast_dtype == torch.bfloat16
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True

        # Check MoE-specific defaults
        assert provider.num_moe_experts == 128
        assert provider.moe_router_load_balancing_type == "aux_loss"
        assert provider.moe_aux_loss_coeff == 1e-3
        assert provider.moe_router_topk == 8
        assert provider.moe_router_pre_softmax is False
        assert provider.moe_grouped_gemm is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_permute_fusion is True

    def test_qwen3_moe_model_provider_with_custom_moe_config(self):
        """Test Qwen3MoEModelProvider with custom MoE configuration."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_moe_experts=64,
            moe_router_topk=4,
            moe_aux_loss_coeff=1e-2,
        )

        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 4
        assert provider.moe_aux_loss_coeff == 1e-2

    def test_qwen3_moe_model_provider_with_custom_rope(self):
        """Test Qwen3MoEModelProvider with custom RoPE configuration."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
            rotary_percent=0.5,
        )

        assert provider.rotary_base == 500000.0
        assert provider.rotary_percent == 0.5

    def test_qwen3_moe_model_provider_ffn_hidden_size(self):
        """Test Qwen3MoEModelProvider FFN hidden size configuration."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
            moe_ffn_hidden_size=1536,
        )

        assert provider.ffn_hidden_size == 11008
        assert provider.moe_ffn_hidden_size == 1536

    def test_qwen3_moe_model_provider_group_query_attention(self):
        """Test Qwen3MoEModelProvider with group query attention."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=4,
        )

        assert provider.num_query_groups == 4

    def test_qwen3_moe_model_provider_custom_vocab_size(self):
        """Test Qwen3MoEModelProvider with custom vocabulary size."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
        )

        assert provider.vocab_size == 32000

    def test_qwen3_moe_model_provider_custom_sequence_length(self):
        """Test Qwen3MoEModelProvider with custom sequence length."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
        )

        assert provider.seq_length == 8192

    def test_qwen3_moe_model_provider_qk_layernorm(self):
        """Test Qwen3MoEModelProvider has QK layernorm enabled by default."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Qwen3 MoE uses QK layernorm unlike Qwen2
        assert provider.qk_layernorm is True

    def test_qwen3_moe_model_provider_dtype_configuration(self):
        """Test Qwen3MoEModelProvider dtype configuration."""
        provider = Qwen3MoEModelProvider(
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


class TestQwen3MoEModelProvider30B_A3B:
    """Test cases for Qwen3MoEModelProvider30B_A3B class."""

    def test_qwen3_moe_30b_a3b_default_configuration(self):
        """Test Qwen3 MoE 30B-A3B model has correct default configuration."""
        provider = Qwen3MoEModelProvider30B_A3B()

        # Check Qwen3 MoE 30B-A3B specific configuration
        assert provider.num_layers == 48
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 4
        assert provider.ffn_hidden_size == 6144
        assert provider.moe_ffn_hidden_size == 768

        # Check inherited MoE defaults
        assert provider.num_moe_experts == 128
        assert provider.moe_router_topk == 8
        assert provider.qk_layernorm is True
        assert provider.vocab_size == 151936
        assert provider.seq_length == 40960

        # Check inherited base defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True

    def test_qwen3_moe_30b_a3b_override_configuration(self):
        """Test Qwen3 MoE 30B-A3B model with overridden configuration."""
        provider = Qwen3MoEModelProvider30B_A3B(
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
        assert provider.moe_ffn_hidden_size == 768


class TestQwen3MoEModelProvider235B_A22B:
    """Test cases for Qwen3MoEModelProvider235B_A22B class."""

    def test_qwen3_moe_235b_a22b_default_configuration(self):
        """Test Qwen3 MoE 235B-A22B model has correct default configuration."""
        provider = Qwen3MoEModelProvider235B_A22B()

        # Check Qwen3 MoE 235B-A22B specific configuration
        assert provider.num_layers == 94
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 64
        assert provider.num_query_groups == 4
        assert provider.ffn_hidden_size == 12288
        assert provider.moe_ffn_hidden_size == 1536

        # Check inherited MoE defaults
        assert provider.num_moe_experts == 128
        assert provider.moe_router_topk == 8
        assert provider.qk_layernorm is True
        assert provider.vocab_size == 151936
        assert provider.seq_length == 40960

        # Check inherited base defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True

    def test_qwen3_moe_235b_a22b_override_configuration(self):
        """Test Qwen3 MoE 235B-A22B model with overridden configuration."""
        provider = Qwen3MoEModelProvider235B_A22B(
            seq_length=16384,
            moe_aux_loss_coeff=1e-2,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 16384
        assert provider.moe_aux_loss_coeff == 1e-2
        assert provider.hidden_dropout == 0.1

        # Check defaults remain
        assert provider.num_layers == 94
        assert provider.hidden_size == 4096
        assert provider.moe_ffn_hidden_size == 1536


class TestQwen3MoEProviderInheritance:
    """Test inheritance relationships between Qwen3 MoE providers."""

    def test_qwen3_moe_models_inherit_from_base(self):
        """Test Qwen3 MoE providers inherit from Qwen3MoEModelProvider."""
        assert issubclass(Qwen3MoEModelProvider30B_A3B, Qwen3MoEModelProvider)
        assert issubclass(Qwen3MoEModelProvider235B_A22B, Qwen3MoEModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Qwen3 MoE 30B-A3B
        provider = Qwen3MoEModelProvider30B_A3B()

        # The provide method should be inherited from GPTModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)


class TestQwen3MoEProviderEdgeCases:
    """Test edge cases and error conditions for Qwen3 MoE providers."""

    def test_valid_num_query_groups(self):
        """Test that valid num_query_groups configuration works."""
        # num_attention_heads must be divisible by num_query_groups
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,  # 32 divisible by 8
        )
        assert provider.num_query_groups == 8

    def test_moe_configuration_validity(self):
        """Test MoE configuration parameters."""
        provider = Qwen3MoEModelProvider(
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
        provider = Qwen3MoEModelProvider(
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
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
        )

        assert provider.seq_length == 8192

    def test_rotary_base_configuration(self):
        """Test rotary base configuration."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
        )

        assert provider.rotary_base == 500000.0

    def test_layernorm_epsilon_override(self):
        """Test layernorm epsilon configuration."""
        provider = Qwen3MoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            layernorm_epsilon=1e-5,
        )

        assert provider.layernorm_epsilon == 1e-5


class TestQwen3MoEProviderQueryGroupsConsistency:
    """Test cases to verify query groups consistency across Qwen3 MoE models."""

    def test_qwen3_moe_30b_a3b_num_query_groups(self):
        """Test that Qwen3 MoE 30B-A3B has correct num_query_groups."""
        provider = Qwen3MoEModelProvider30B_A3B()
        assert provider.num_query_groups == 4

    def test_qwen3_moe_235b_a22b_num_query_groups(self):
        """Test that Qwen3 MoE 235B-A22B has correct num_query_groups."""
        provider = Qwen3MoEModelProvider235B_A22B()
        assert provider.num_query_groups == 4


class TestQwen3MoEProviderVocabularyConsistency:
    """Test cases to verify vocabulary size consistency across Qwen3 MoE models."""

    def test_qwen3_moe_models_vocab_size(self):
        """Test that Qwen3 MoE models have consistent vocab size."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.vocab_size == 151936


class TestQwen3MoEProviderMoEConsistency:
    """Test cases to verify MoE configuration consistency across Qwen3 MoE models."""

    def test_qwen3_moe_models_expert_count(self):
        """Test that Qwen3 MoE models have consistent expert count."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.num_moe_experts == 128
            assert provider.moe_router_topk == 8

    def test_qwen3_moe_models_router_config(self):
        """Test that Qwen3 MoE models have consistent router configuration."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.moe_router_load_balancing_type == "aux_loss"
            assert provider.moe_aux_loss_coeff == 1e-3
            assert provider.moe_router_pre_softmax is False
            assert provider.moe_grouped_gemm is True

    def test_qwen3_moe_models_dispatcher_config(self):
        """Test that Qwen3 MoE models have consistent dispatcher configuration."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.moe_token_dispatcher_type == "alltoall"
            assert provider.moe_permute_fusion is True


class TestQwen3MoEProviderArchitecturalFeatures:
    """Test cases to verify Qwen3 MoE architectural features."""

    def test_qwen3_moe_qk_layernorm_feature(self):
        """Test that Qwen3 MoE models have QK layernorm enabled."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.qk_layernorm is True

    def test_qwen3_moe_extended_context(self):
        """Test that Qwen3 MoE models have extended context length."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.seq_length == 40960  # 40K context

    def test_qwen3_moe_dtype_defaults(self):
        """Test that Qwen3 MoE models have correct dtype defaults."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.autocast_dtype == torch.bfloat16
            assert provider.params_dtype == torch.bfloat16
            assert provider.bf16 is True

    def test_qwen3_moe_kv_channels(self):
        """Test that Qwen3 MoE models have correct KV channels configuration."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.kv_channels == 128

    def test_qwen3_moe_bias_configuration(self):
        """Test that Qwen3 MoE models have correct bias configuration."""
        providers = [
            Qwen3MoEModelProvider30B_A3B(),
            Qwen3MoEModelProvider235B_A22B(),
        ]

        for provider in providers:
            assert provider.add_bias_linear is False
            assert provider.add_qkv_bias is False
