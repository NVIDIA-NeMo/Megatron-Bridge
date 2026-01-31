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

from megatron.bridge.models.kimi.kimi_provider import KimiK2Provider


class TestKimiK2Provider:
    """Test cases for KimiK2Provider class."""

    def test_kimi_k2_provider_initialization(self):
        """Test KimiK2Provider can be initialized with default values."""
        provider = KimiK2Provider()

        # Check core model architecture
        assert provider.num_layers == 61
        assert provider.hidden_size == 7168
        assert provider.num_attention_heads == 64
        assert provider.vocab_size == 163840

        # Check key configuration
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_kimi_k2_moe_configuration(self):
        """Test KimiK2Provider MoE-specific configuration."""
        provider = KimiK2Provider()

        # Check key MoE settings
        assert provider.num_moe_experts == 384
        assert provider.moe_router_topk == 8
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_token_dispatcher_type == "alltoall"

        # Check moe_layer_freq format
        assert isinstance(provider.moe_layer_freq, list)
        assert len(provider.moe_layer_freq) == 61
        assert provider.moe_layer_freq[0] == 0  # first layer is dense
        assert all(freq == 1 for freq in provider.moe_layer_freq[1:])  # rest are MoE

    def test_kimi_k2_mla_configuration(self):
        """Test KimiK2Provider MLA (Multi-Latent Attention) configuration."""
        provider = KimiK2Provider()

        # Check key MLA settings
        assert provider.multi_latent_attention is True
        assert provider.q_lora_rank == 1536
        assert provider.kv_lora_rank == 512
        assert provider.qk_head_dim == 128
        assert provider.v_head_dim == 128

    def test_kimi_k2_custom_parameters(self):
        """Test that KimiK2Provider can be initialized with custom parameters."""
        custom_provider = KimiK2Provider(
            num_layers=30,
            hidden_size=4096,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=16,
            sequence_parallel=False,
        )

        # Check custom values
        assert custom_provider.num_layers == 30
        assert custom_provider.hidden_size == 4096
        assert custom_provider.tensor_model_parallel_size == 2
        assert custom_provider.pipeline_model_parallel_size == 4
        assert custom_provider.expert_model_parallel_size == 16
        assert custom_provider.sequence_parallel is False

        # Check defaults are still preserved
        assert custom_provider.num_moe_experts == 384
        assert custom_provider.multi_latent_attention is True

    def test_kimi_k2_inheritance(self):
        """Test that KimiK2Provider properly inherits from required base classes."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.bridge.models.transformer_config import MLATransformerConfig

        provider = KimiK2Provider()

        # Check it's a dataclass
        assert hasattr(provider, "__dataclass_fields__")

        # Check inheritance
        assert isinstance(provider, GPTModelProvider)
        assert isinstance(provider, MLATransformerConfig)

        # Check it has the provide method
        assert hasattr(provider, "provide")
        assert callable(getattr(provider, "provide"))
