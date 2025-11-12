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

from megatron.bridge.models.mixtral.mixtral_provider import MixtralModelProvider


class TestMixtralModelProvider:
    """Test cases for MixtralModelProvider class."""

    def test_mixtral_model_provider_initialization(self):
        """Test MixtralModelProvider can be initialized with default values."""
        provider = MixtralModelProvider()

        # Check base transformer config fields
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 14336

        # Check Mixtral-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is False
        assert provider.seq_length == 32768
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5
        assert provider.rotary_base == 1000000.0
        assert provider.rotary_percent == 1.0
        assert provider.position_embedding_type == "rope"
        assert provider.vocab_size == 32000
        assert provider.init_method_std == 0.02
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True

        # Check MoE-specific parameters
        assert provider.num_moe_experts == 8
        assert provider.moe_router_topk == 2
        assert provider.moe_ffn_hidden_size == 14336
        assert provider.moe_aux_loss_coeff == 0.01
        assert provider.moe_router_pre_softmax is True
        assert provider.moe_router_load_balancing_type == "none"
        assert provider.moe_router_score_function == "softmax"
        assert provider.moe_grouped_gemm is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_permute_fusion is True

    def test_mixtral_model_provider_custom_initialization(self):
        """Test MixtralModelProvider can be initialized with custom values."""
        provider = MixtralModelProvider(
            num_layers=64,
            hidden_size=8192,
            num_attention_heads=64,
            num_moe_experts=16,
            moe_router_topk=4,
        )

        assert provider.num_layers == 64
        assert provider.hidden_size == 8192
        assert provider.num_attention_heads == 64
        assert provider.num_moe_experts == 16
        assert provider.moe_router_topk == 4

    def test_mixtral_model_provider_inheritance(self):
        """Test that MixtralModelProvider properly inherits from GPTModelProvider."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        provider = MixtralModelProvider()
        assert isinstance(provider, GPTModelProvider)

    def test_mixtral_model_provider_has_provide_method(self):
        """Test that MixtralModelProvider has the provide method."""
        provider = MixtralModelProvider()
        assert hasattr(provider, "provide")
        assert callable(getattr(provider, "provide"))

    def test_mixtral_model_provider_moe_parameters(self):
        """Test that MixtralModelProvider MoE parameters are correctly set."""
        provider = MixtralModelProvider(
            num_moe_experts=16,
            moe_router_topk=4,
            moe_aux_loss_coeff=0.02,
        )

        assert provider.num_moe_experts == 16
        assert provider.moe_router_topk == 4
        assert provider.moe_aux_loss_coeff == 0.02
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_router_load_balancing_type == "none"

    def test_mixtral_model_provider_dtype_configuration(self):
        """Test that MixtralModelProvider dtype parameters are correctly configured."""
        provider = MixtralModelProvider()

        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True

        # Test custom dtype
        provider_fp16 = MixtralModelProvider(
            params_dtype=torch.float16,
            fp16=True,
            bf16=False,
        )

        assert provider_fp16.params_dtype == torch.float16
        assert provider_fp16.fp16 is True
        assert provider_fp16.bf16 is False

    def test_mixtral_model_provider_with_custom_rope(self):
        """Test MixtralModelProvider with custom RoPE configuration."""
        provider = MixtralModelProvider(
            rotary_base=10000000.0,
            rotary_percent=0.5,
        )

        assert provider.rotary_base == 10000000.0
        assert provider.rotary_percent == 0.5

    def test_mixtral_model_provider_custom_vocab_size(self):
        """Test MixtralModelProvider with custom vocabulary size."""
        provider = MixtralModelProvider(vocab_size=64000)

        assert provider.vocab_size == 64000

    def test_mixtral_model_provider_custom_sequence_length(self):
        """Test MixtralModelProvider with custom sequence length."""
        provider = MixtralModelProvider(seq_length=65536)

        assert provider.seq_length == 65536

    def test_mixtral_8x7b_configuration(self):
        """Test MixtralModelProvider with Mixtral 8x7B configuration."""
        provider = MixtralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
            ffn_hidden_size=14336,
            num_moe_experts=8,
            moe_router_topk=2,
            vocab_size=32000,
        )

        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_moe_experts == 8
        assert provider.moe_router_topk == 2
        assert provider.vocab_size == 32000

    def test_mixtral_8x22b_configuration(self):
        """Test MixtralModelProvider with Mixtral 8x22B configuration."""
        provider = MixtralModelProvider(
            num_layers=56,
            hidden_size=6144,
            num_attention_heads=48,
            num_query_groups=8,
            ffn_hidden_size=16384,
            num_moe_experts=8,
            moe_router_topk=2,
            vocab_size=32768,
        )

        assert provider.num_layers == 56
        assert provider.hidden_size == 6144
        assert provider.num_attention_heads == 48
        assert provider.num_moe_experts == 8
        assert provider.moe_router_topk == 2

    def test_mixtral_model_provider_moe_routing_configuration(self):
        """Test MixtralModelProvider MoE routing configuration."""
        provider = MixtralModelProvider(
            moe_router_load_balancing_type="aux_loss",
            moe_router_score_function="sigmoid",
            moe_router_pre_softmax=False,
        )

        assert provider.moe_router_load_balancing_type == "aux_loss"
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_pre_softmax is False

    def test_mixtral_model_provider_expert_parallelism(self):
        """Test MixtralModelProvider with expert parallelism configuration."""
        provider = MixtralModelProvider(
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=2,
        )

        assert provider.expert_model_parallel_size == 4
        assert provider.expert_tensor_parallel_size == 2


class TestMixtralProviderIntegration:
    """Integration tests for Mixtral model provider."""

    def test_mixtral_provider_creates_valid_config(self):
        """Test that MixtralModelProvider creates a valid configuration."""
        provider = MixtralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
            ffn_hidden_size=14336,
            moe_ffn_hidden_size=14336,
            num_moe_experts=8,
            moe_router_topk=2,
        )

        # Verify that all MoE parameters are set
        assert provider.num_moe_experts == 8
        assert provider.moe_router_topk == 2
        assert provider.moe_ffn_hidden_size == 14336

        # Verify that attention parameters are set
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8

    def test_mixtral_provider_with_different_expert_configurations(self):
        """Test MixtralModelProvider with different expert configurations."""
        configs = [
            {"num_moe_experts": 4, "moe_router_topk": 1},
            {"num_moe_experts": 8, "moe_router_topk": 2},
            {"num_moe_experts": 16, "moe_router_topk": 4},
        ]

        for config in configs:
            provider = MixtralModelProvider(**config)
            assert provider.num_moe_experts == config["num_moe_experts"]
            assert provider.moe_router_topk == config["moe_router_topk"]

    def test_mixtral_provider_uses_rmsnorm(self):
        """Test that MixtralModelProvider uses RMSNorm normalization."""
        provider = MixtralModelProvider()
        assert provider.normalization == "RMSNorm"

    def test_mixtral_provider_uses_silu_activation(self):
        """Test that MixtralModelProvider uses SiLU activation."""
        provider = MixtralModelProvider()
        assert provider.activation_func is F.silu

    def test_mixtral_provider_uses_rope_embeddings(self):
        """Test that MixtralModelProvider uses RoPE position embeddings."""
        provider = MixtralModelProvider()
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == 1000000.0

    def test_mixtral_provider_moe_optimizations(self):
        """Test that MixtralModelProvider has MoE optimizations enabled."""
        provider = MixtralModelProvider()
        assert provider.moe_grouped_gemm is True
        assert provider.moe_permute_fusion is True
        assert provider.moe_token_dispatcher_type == "alltoall"


class TestMixtralProviderEdgeCases:
    """Test edge cases and error conditions."""

    def test_valid_num_query_groups(self):
        """Test that valid num_query_groups configuration works."""
        # num_attention_heads must be divisible by num_query_groups
        provider = MixtralModelProvider(
            num_attention_heads=32,
            num_query_groups=8,  # 32 divisible by 8
        )
        assert provider.num_query_groups == 8

    def test_vocabulary_size_divisibility(self):
        """Test vocabulary size divisibility configuration."""
        provider = MixtralModelProvider(
            vocab_size=32768,
            make_vocab_size_divisible_by=128,
        )

        assert provider.make_vocab_size_divisible_by == 128

    def test_seq_length_override(self):
        """Test sequence length configuration."""
        provider = MixtralModelProvider(
            seq_length=131072,  # Very long context
        )

        assert provider.seq_length == 131072

    def test_rotary_base_configuration(self):
        """Test rotary base configuration for long context."""
        provider = MixtralModelProvider(
            rotary_base=1000000.0,  # Extended RoPE base
        )

        assert provider.rotary_base == 1000000.0

    def test_layernorm_epsilon_override(self):
        """Test layernorm epsilon configuration."""
        provider = MixtralModelProvider(
            layernorm_epsilon=1e-6,
        )

        assert provider.layernorm_epsilon == 1e-6
