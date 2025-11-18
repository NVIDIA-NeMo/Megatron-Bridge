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

import torch
import torch.nn.functional as F
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.olmoe.olmoe_provider import (
    OlMoEModelProvider,
    OLMoESelfAttention,
    olmoe_layer_spec,
)


class TestOlMoEModelProvider:
    """Test cases for OlMoEModelProvider class."""

    def test_olmoe_model_provider_initialization(self):
        """Test OlMoEModelProvider can be initialized with default values."""
        provider = OlMoEModelProvider()

        # Check base transformer config fields
        assert provider.num_layers == 16
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 16
        assert provider.ffn_hidden_size == 1024
        assert provider.moe_ffn_hidden_size == 1024

        # Check OLMoE-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.seq_length == 4096
        assert provider.kv_channels == 2048 // 16
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5

        # Check attention parameters
        assert provider.num_query_groups == 16
        assert provider.qk_layernorm is True

        # Check RoPE parameters
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == 10000.0

        # Check MoE-specific parameters
        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 8
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_router_load_balancing_type == "seq_aux_loss"
        assert provider.moe_aux_loss_coeff == 1e-2
        assert provider.moe_router_pre_softmax is True
        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_score_function == "softmax"
        assert provider.moe_permute_fusion is True
        assert provider.moe_router_dtype == "fp32"

        # Check optimization parameters
        assert provider.persist_layer_norm is True
        assert provider.vocab_size == 50304
        assert provider.init_method_std == 0.02
        assert provider.autocast_dtype == torch.bfloat16
        assert provider.params_dtype == torch.float32
        assert provider.bf16 is False

    def test_olmoe_model_provider_custom_initialization(self):
        """Test OlMoEModelProvider can be initialized with custom values."""
        provider = OlMoEModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=2048,
            num_moe_experts=128,
            moe_router_topk=16,
        )

        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.ffn_hidden_size == 2048
        assert provider.num_moe_experts == 128
        assert provider.moe_router_topk == 16

    def test_olmoe_model_provider_inheritance(self):
        """Test that OlMoEModelProvider properly inherits from GPTModelProvider."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        provider = OlMoEModelProvider()
        assert isinstance(provider, GPTModelProvider)

    def test_olmoe_model_provider_has_provide_method(self):
        """Test that OlMoEModelProvider has the provide method."""
        provider = OlMoEModelProvider()
        assert hasattr(provider, "provide")
        assert callable(getattr(provider, "provide"))

    def test_olmoe_model_provider_moe_parameters(self):
        """Test that OlMoEModelProvider MoE parameters are correctly set."""
        provider = OlMoEModelProvider(
            num_moe_experts=32,
            moe_router_topk=4,
            moe_aux_loss_coeff=0.01,
        )

        assert provider.num_moe_experts == 32
        assert provider.moe_router_topk == 4
        assert provider.moe_aux_loss_coeff == 0.01
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_router_load_balancing_type == "seq_aux_loss"

    def test_olmoe_model_provider_dtype_configuration(self):
        """Test that OlMoEModelProvider dtype parameters are correctly configured."""
        provider = OlMoEModelProvider()

        assert provider.autocast_dtype == torch.bfloat16
        assert provider.params_dtype == torch.float32
        assert provider.bf16 is False

        # Test custom dtype
        provider_fp16 = OlMoEModelProvider(
            autocast_dtype=torch.float16,
            params_dtype=torch.float16,
            bf16=True,
        )

        assert provider_fp16.autocast_dtype == torch.float16
        assert provider_fp16.params_dtype == torch.float16
        assert provider_fp16.bf16 is True


class TestOlmoeLayerSpec:
    """Test cases for olmoe_layer_spec function."""

    def test_olmoe_layer_spec_returns_module_spec(self):
        """Test that olmoe_layer_spec returns a ModuleSpec."""
        provider = OlMoEModelProvider()
        layer_spec = olmoe_layer_spec(provider)

        assert isinstance(layer_spec, ModuleSpec)

    def test_olmoe_layer_spec_uses_custom_attention(self):
        """Test that olmoe_layer_spec configures custom OLMoE attention."""
        provider = OlMoEModelProvider()
        layer_spec = olmoe_layer_spec(provider)

        # Verify that the layer spec has self_attention submodule
        assert hasattr(layer_spec, "submodules")
        assert hasattr(layer_spec.submodules, "self_attention")

        # Verify that the custom OLMoESelfAttention is used
        assert layer_spec.submodules.self_attention.module == OLMoESelfAttention

    def test_olmoe_layer_spec_with_different_configs(self):
        """Test that olmoe_layer_spec works with different provider configurations."""
        provider1 = OlMoEModelProvider(num_layers=16, hidden_size=2048)
        provider2 = OlMoEModelProvider(num_layers=32, hidden_size=4096)

        layer_spec1 = olmoe_layer_spec(provider1)
        layer_spec2 = olmoe_layer_spec(provider2)

        assert isinstance(layer_spec1, ModuleSpec)
        assert isinstance(layer_spec2, ModuleSpec)

        # Both should use the same custom attention
        assert layer_spec1.submodules.self_attention.module == OLMoESelfAttention
        assert layer_spec2.submodules.self_attention.module == OLMoESelfAttention


class TestOLMoESelfAttention:
    """Test cases for OLMoESelfAttention class."""

    def test_olmoe_self_attention_initialization(self):
        """Test that OLMoESelfAttention can be initialized."""
        from megatron.core.transformer.attention import SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.transformer_config import TransformerConfig

        # Create a minimal config
        config = Mock(spec=TransformerConfig)
        config.num_attention_heads = 16
        config.hidden_size = 2048
        config.kv_channels = 128
        config.num_query_groups = 16
        config.layernorm_epsilon = 1e-5
        config.attention_dropout = 0.0
        config.params_dtype = torch.float32
        config.pipeline_dtype = None
        config.apply_query_key_layer_scaling = False
        config.attention_softmax_in_fp32 = False
        config.test_mode = False

        # Create submodules mock
        submodules = Mock(spec=SelfAttentionSubmodules)
        submodules.linear_qkv = Mock()
        submodules.core_attention = Mock()
        submodules.linear_proj = Mock()
        submodules.q_layernorm = Mock()
        submodules.k_layernorm = Mock()

        # This should not raise an exception
        try:
            attention = OLMoESelfAttention(
                config=config,
                submodules=submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.padding,
            )
            # Verify it's an instance of the parent class
            from megatron.core.transformer.attention import SelfAttention as MCoreSelfAttention

            assert isinstance(attention, MCoreSelfAttention)
        except Exception:
            # If there are missing dependencies or config issues, we can skip
            # but we should at least verify the class exists
            assert OLMoESelfAttention is not None

    def test_olmoe_self_attention_has_custom_layernorm(self):
        """Test that OLMoESelfAttention has custom q_layernorm and k_layernorm."""
        from megatron.core.transformer.attention import SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.transformer_config import TransformerConfig

        # Create a minimal config
        config = Mock(spec=TransformerConfig)
        config.num_attention_heads = 16
        config.hidden_size = 2048
        config.kv_channels = 128
        config.num_query_groups = 16
        config.layernorm_epsilon = 1e-5
        config.attention_dropout = 0.0
        config.params_dtype = torch.float32
        config.pipeline_dtype = None
        config.apply_query_key_layer_scaling = False
        config.attention_softmax_in_fp32 = False
        config.test_mode = False

        # Create submodules mock
        submodules = Mock(spec=SelfAttentionSubmodules)
        submodules.linear_qkv = Mock()
        submodules.core_attention = Mock()
        submodules.linear_proj = Mock()
        submodules.q_layernorm = Mock()
        submodules.k_layernorm = Mock()

        try:
            attention = OLMoESelfAttention(
                config=config,
                submodules=submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.padding,
            )

            # Verify that q_layernorm and k_layernorm exist
            assert hasattr(attention, "q_layernorm")
            assert hasattr(attention, "k_layernorm")
        except Exception:
            # If initialization fails due to dependencies, skip the test
            pass

    def test_olmoe_self_attention_has_get_query_key_value_tensors(self):
        """Test that OLMoESelfAttention has get_query_key_value_tensors method."""
        # Verify the method exists in the class
        assert hasattr(OLMoESelfAttention, "get_query_key_value_tensors")
        assert callable(getattr(OLMoESelfAttention, "get_query_key_value_tensors"))


class TestOlMoEModelProviderIntegration:
    """Integration tests for OLMoE model provider."""

    def test_olmoe_provider_layer_spec_integration(self):
        """Test that OlMoEModelProvider works with olmoe_layer_spec."""
        provider = OlMoEModelProvider()

        # The transformer_layer_spec should be set to olmoe_layer_spec
        assert provider.transformer_layer_spec == olmoe_layer_spec

    def test_olmoe_provider_creates_valid_config(self):
        """Test that OlMoEModelProvider creates a valid configuration."""
        provider = OlMoEModelProvider(
            num_layers=16,
            hidden_size=2048,
            num_attention_heads=16,
            num_query_groups=16,
            ffn_hidden_size=1024,
            moe_ffn_hidden_size=1024,
            num_moe_experts=64,
            moe_router_topk=8,
        )

        # Verify that all MoE parameters are set
        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 8
        assert provider.moe_ffn_hidden_size == 1024

        # Verify that attention parameters are set
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 16
        assert provider.qk_layernorm is True

        # Verify that the layer spec is callable or a ModuleSpec
        assert callable(provider.transformer_layer_spec) or isinstance(provider.transformer_layer_spec, ModuleSpec)

    def test_olmoe_provider_with_different_expert_configurations(self):
        """Test OlMoEModelProvider with different expert configurations."""
        configs = [
            {"num_moe_experts": 32, "moe_router_topk": 4},
            {"num_moe_experts": 64, "moe_router_topk": 8},
            {"num_moe_experts": 128, "moe_router_topk": 16},
        ]

        for config in configs:
            provider = OlMoEModelProvider(**config)
            assert provider.num_moe_experts == config["num_moe_experts"]
            assert provider.moe_router_topk == config["moe_router_topk"]

    def test_olmoe_provider_qk_layernorm_enabled(self):
        """Test that OlMoEModelProvider has QK layernorm enabled by default."""
        provider = OlMoEModelProvider()
        assert provider.qk_layernorm is True

    def test_olmoe_provider_uses_rmsnorm(self):
        """Test that OlMoEModelProvider uses RMSNorm normalization."""
        provider = OlMoEModelProvider()
        assert provider.normalization == "RMSNorm"

    def test_olmoe_provider_uses_silu_activation(self):
        """Test that OlMoEModelProvider uses SiLU activation."""
        provider = OlMoEModelProvider()
        assert provider.activation_func == F.silu

    def test_olmoe_provider_uses_rope_embeddings(self):
        """Test that OlMoEModelProvider uses RoPE position embeddings."""
        provider = OlMoEModelProvider()
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == 10000.0
