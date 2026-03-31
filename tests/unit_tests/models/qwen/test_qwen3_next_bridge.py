# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
Unit tests for Qwen3 Next bridge functionality.
"""

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen3_next_bridge import Qwen3NextBridge


class TestQwen3NextBridge:
    """Test cases for Qwen3NextBridge class."""

    @pytest.fixture
    def qwen3_next_80b_a3b_config_dict(self):
        """Create a sample Qwen3 Next 80B configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3NextForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "decoder_sparse_step": 1,
            "eos_token_id": 151645,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 5120,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_next",
            "moe_intermediate_size": 512,
            "norm_topk_prob": True,
            "num_attention_heads": 16,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "output_router_logits": False,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000000,
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        }

    @pytest.fixture
    def mock_qwen3_next_config(self, qwen3_next_80b_a3b_config_dict):
        """Create a mock Qwen3 Next configuration."""
        config = Mock()
        for key, value in qwen3_next_80b_a3b_config_dict.items():
            setattr(config, key, value)
        # Explicitly set to None so hf_config_to_provider_kwargs skips them.
        # Qwen3-Next is not an MLA model and not a DeepSeek-style MoE / MTP model.
        for null_attr in (
            # MLA attrs
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            # Alternative MoE expert count attrs (would overwrite num_experts=512)
            "n_routed_experts",
            "num_local_experts",
            # MTP attrs (not used by Qwen3-Next, but truthy in Mock)
            "num_nextn_predict_layers",
            "mtp_num_hidden_layers",
        ):
            setattr(config, null_attr, None)
        return config

    @pytest.fixture
    def mock_pretrained_qwen3_next(self, mock_qwen3_next_config):
        """Create a mock PreTrainedCausalLM with Qwen3 Next model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Qwen3NextBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(Qwen3NextBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test basic provider_bridge functionality."""
        bridge = Qwen3NextBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check that it returns a GPTModelProvider instance (not a model-specific subclass)
        assert isinstance(result, GPTModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == mock_qwen3_next_config.num_hidden_layers
        assert result.hidden_size == mock_qwen3_next_config.hidden_size
        assert result.num_attention_heads == mock_qwen3_next_config.num_attention_heads
        assert result.seq_length == mock_qwen3_next_config.max_position_embeddings
        assert result.rotary_base == mock_qwen3_next_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test vocabulary size mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check vocabulary configuration
        assert result.vocab_size == mock_qwen3_next_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_qwen3_next_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test attention configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check attention configuration
        assert result.num_attention_heads == mock_qwen3_next_config.num_attention_heads
        assert result.num_query_groups == mock_qwen3_next_config.num_key_value_heads
        assert result.qk_layernorm is True  # Qwen3 Next uses QK layernorm
        assert result.layernorm_zero_centered_gamma is True
        assert result.attention_output_gate is True

    def test_provider_bridge_linear_attention_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test linear attention configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check linear attention configuration
        if isinstance(result.linear_attention_freq, int):
            assert result.linear_attention_freq == mock_qwen3_next_config.full_attention_interval
        else:
            assert isinstance(result.linear_attention_freq, list)
            for i in range(result.num_layers):
                if (i + 1) % mock_qwen3_next_config.full_attention_interval == 0:
                    assert result.linear_attention_freq[i] == 0
                else:
                    assert result.linear_attention_freq[i] == 1
        assert result.linear_conv_kernel_dim == mock_qwen3_next_config.linear_conv_kernel_dim
        assert result.linear_key_head_dim == mock_qwen3_next_config.linear_key_head_dim
        assert result.linear_value_head_dim == mock_qwen3_next_config.linear_value_head_dim
        assert result.linear_num_key_heads == mock_qwen3_next_config.linear_num_key_heads
        assert result.linear_num_value_heads == mock_qwen3_next_config.linear_num_value_heads
        assert result.experimental_attention_variant == "gated_delta_net"

    def test_provider_bridge_mlp_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test MLP configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check MLP configuration
        assert result.ffn_hidden_size == mock_qwen3_next_config.intermediate_size
        assert result.moe_ffn_hidden_size == mock_qwen3_next_config.moe_intermediate_size
        assert result.gated_linear_unit is True  # Qwen3 Next uses gated linear units

    def test_provider_bridge_moe_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test MoE-specific configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check MoE-specific configuration
        assert result.num_moe_experts == mock_qwen3_next_config.num_experts
        assert result.moe_router_topk == mock_qwen3_next_config.num_experts_per_tok
        assert result.moe_grouped_gemm is True
        assert result.moe_shared_expert_intermediate_size == mock_qwen3_next_config.shared_expert_intermediate_size
        assert result.moe_shared_expert_gate is True

    def test_provider_bridge_normalization(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test normalization configuration."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check normalization settings
        assert result.layernorm_epsilon == mock_qwen3_next_config.rms_norm_eps
        assert result.init_method_std == mock_qwen3_next_config.initializer_range

    def test_provider_bridge_position_embedding(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test position embedding configuration."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check position embedding
        assert result.rotary_base == mock_qwen3_next_config.rope_theta
        assert result.rotary_percent == mock_qwen3_next_config.partial_rotary_factor
        assert result.position_embedding_type == "rope"

    def test_provider_bridge_mtp_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test MTP configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check MTP configuration
        assert not result.mtp_num_layers

    def test_provider_bridge_dtype_handling(self, qwen3_next_80b_a3b_config_dict):
        """Test dtype handling in provider_bridge."""
        # Test with bfloat16
        config = Mock()
        for key, value in qwen3_next_80b_a3b_config_dict.items():
            setattr(config, key, value)
        for null_attr in (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "n_routed_experts",
            "num_local_experts",
            "num_nextn_predict_layers",
        ):
            setattr(config, null_attr, None)
        config.torch_dtype = "bfloat16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

        # Test with float16
        config.torch_dtype = "float16"
        result = bridge.provider_bridge(mock_pretrained)

        assert result.fp16 is True
        assert result.bf16 is False
        assert result.params_dtype == torch.float16

    def test_provider_bridge_tie_word_embeddings_true(self, mock_qwen3_next_config):
        """Test provider_bridge with tie_word_embeddings=True."""
        mock_qwen3_next_config.tie_word_embeddings = True

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is True

    def test_provider_bridge_tie_word_embeddings_false(self, mock_qwen3_next_config):
        """Test provider_bridge with tie_word_embeddings=False."""
        mock_qwen3_next_config.tie_word_embeddings = False

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_missing_tie_word_embeddings(self, mock_qwen3_next_config):
        """Test provider_bridge when tie_word_embeddings is missing."""
        # Remove tie_word_embeddings attribute
        if hasattr(mock_qwen3_next_config, "tie_word_embeddings"):
            delattr(mock_qwen3_next_config, "tie_word_embeddings")

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should default to False when missing
        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_80b_a3b_config(self, qwen3_next_80b_a3b_config_dict):
        """Test provider_bridge with Qwen3 Next 80B A3B configuration."""
        config = Mock()
        for key, value in qwen3_next_80b_a3b_config_dict.items():
            setattr(config, key, value)
        for null_attr in (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "n_routed_experts",
            "num_local_experts",
            "num_nextn_predict_layers",
        ):
            setattr(config, null_attr, None)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Check 80B-A3B-specific configuration
        assert result.num_layers == 48
        assert result.hidden_size == 2048
        assert result.num_attention_heads == 16
        assert result.ffn_hidden_size == 5120
        assert result.moe_ffn_hidden_size == 512

    def test_mapping_registry(self):
        """Test mapping_registry returns valid mappings."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Check that registry is not None and has mappings
        assert registry is not None
        assert len(registry.mappings) > 0

        # Check for expected mapping types
        mapping_types = [type(mapping).__name__ for mapping in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "QKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types
        assert "GDNLinearMapping" in mapping_types
        assert "RMSNorm2ZeroCenteredRMSNormMapping" in mapping_types

    def test_mapping_registry_parameter_mappings(self):
        """Test that mapping_registry contains expected parameter mappings."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract all AutoMapping instances
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        # Check for critical parameter mappings
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Should have embedding mappings
        assert "model.embed_tokens.weight" in hf_params
        assert "embedding.word_embeddings.weight" in megatron_params

        # Should have output layer mappings
        assert "lm_head.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        # Should have layer norm mappings
        assert "model.norm.weight" in hf_params
        assert "decoder.final_layernorm.weight" in megatron_params

    def test_mapping_registry_mtp_mapping(self):
        """Test that mapping_registry contains MTP mapping."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract MTP mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        # Check for critical parameter mappings
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Should have embedding and hidden projection
        assert "mtp.fc.weight" in hf_params
        assert "mtp.layers.0.eh_proj.weight" in megatron_params

        # Should have pre-fc norms for embedding and hidden
        assert "mtp.pre_fc_norm_embedding.weight" in hf_params
        assert "mtp.pre_fc_norm_hidden.weight" in hf_params
        assert "mtp.layers.0.enorm.weight" in megatron_params
        assert "mtp.layers.0.hnorm.weight" in megatron_params

        # Should have final layernorm
        assert "mtp.norm.weight" in hf_params
        assert "mtp.layers.0.final_layernorm.weight" in megatron_params

    def test_mapping_registry_qkv_mapping(self):
        """Test that mapping_registry contains QKV mapping."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract QKVMapping instances
        qkv_mappings = [m for m in registry.mappings if type(m).__name__ == "QKVMapping"]

        # Should have at least one QKV mapping
        assert len(qkv_mappings) > 0

        # Check the QKV mapping structure
        qkv_mapping = qkv_mappings[0]
        assert hasattr(qkv_mapping, "hf_param")
        assert isinstance(qkv_mapping.hf_param, dict)
        assert "q" in qkv_mapping.hf_param
        assert "k" in qkv_mapping.hf_param
        assert "v" in qkv_mapping.hf_param
        assert hasattr(qkv_mapping, "megatron_param")

    def test_mapping_registry_gdn_linear_mapping(self):
        """Test that mapping_registry contains GDN linear mapping."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract GDNLinearMapping instances
        gdn_linear_mappings = [m for m in registry.mappings if type(m).__name__ == "GDNLinearMapping"]
        assert len(gdn_linear_mappings) > 0

        # Check the GDN linear mapping structure
        gdn_linear_mapping = gdn_linear_mappings[0]
        assert hasattr(gdn_linear_mapping, "hf_param")
        assert isinstance(gdn_linear_mapping.hf_param, dict)
        assert "qkvz" in gdn_linear_mapping.hf_param
        assert "ba" in gdn_linear_mapping.hf_param
        assert hasattr(gdn_linear_mapping, "megatron_param")

    def test_mapping_registry_moe_mappings(self):
        """Test that mapping_registry contains MoE-specific mappings."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract all mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        replicated_mappings = [m for m in registry.mappings if type(m).__name__ == "ReplicatedMapping"]
        gated_mlp_mappings = [m for m in registry.mappings if type(m).__name__ == "GatedMLPMapping"]

        # Check for MoE router mapping
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        assert "model.layers.*.mlp.gate.weight" in hf_params
        # shared_expert_gate is represented via ReplicatedMapping in bridge
        replicated_hf_params = [mapping.hf_param for mapping in replicated_mappings]
        assert "model.layers.*.mlp.shared_expert_gate.weight" in replicated_hf_params

        # Check for expert mappings in GatedMLPMapping
        assert len(gated_mlp_mappings) > 0

        # Check expert down projection mapping
        expert_down_params = [
            mapping.hf_param
            for mapping in auto_mappings
            if "experts" in mapping.hf_param and "down_proj" in mapping.hf_param
        ]
        assert len(expert_down_params) > 0


class TestQwen3NextMambaLayerIndexing:
    """Test that the MambaModel bridge generates correct physical layer indices.

    In MambaModel, each HF logical layer N becomes two physical layers:
      - Physical layer 2*N:   attention (GDN or standard)
      - Physical layer 2*N+1: MoE FFN
    """

    @pytest.fixture
    def mock_pretrained_qwen3_next(self):
        """Mock PreTrainedCausalLM with Qwen3-Next 80B config (48 HF layers)."""
        config = Mock()
        config_dict = {
            "architectures": ["Qwen3NextForCausalLM"],
            "attention_dropout": 0.0,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 5120,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000000,
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "vocab_size": 151936,
            "hidden_act": "silu",
            "hidden_dropout": 0.0,
            "attention_bias": False,
            "mlp_bias": False,
            "use_qk_norm": True,
        }
        for key, value in config_dict.items():
            setattr(config, key, value)
        for null_attr in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                          "qk_rope_head_dim", "v_head_dim", "n_routed_experts",
                          "num_local_experts", "num_nextn_predict_layers",
                          "mtp_num_hidden_layers"):
            setattr(config, null_attr, None)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def bridge_with_config(self):
        """Create a bridge with hf_config set (for mapping_registry)."""
        bridge = Qwen3NextBridge()
        config = Mock()
        config.num_hidden_layers = 8  # Small for testing
        config.full_attention_interval = 4
        bridge.hf_config = config
        return bridge

    def _all_megatron_params(self, registry):
        """Extract all megatron param names from the registry."""
        return [m.megatron_param for m in registry.mappings if hasattr(m, "megatron_param")]

    def test_hybrid_override_pattern(self, mock_pretrained_qwen3_next):
        """Test that hybrid_override_pattern is correctly generated from HF config."""
        bridge = Qwen3NextBridge()
        # Mock _hf_model_has_mtp to avoid network calls
        bridge._hf_model_has_mtp = staticmethod(lambda _: False)

        provider = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # 48 HF layers, full_attention_interval=4: GEGEGE*E repeated 12 times
        assert provider.hybrid_override_pattern == "GEGEGE*E" * 12
        assert provider.num_layers == 96

    def test_hybrid_override_pattern_with_mtp(self, mock_pretrained_qwen3_next):
        """Test that MTP suffix is appended when MTP is detected."""
        bridge = Qwen3NextBridge()
        bridge._hf_model_has_mtp = staticmethod(lambda _: True)

        provider = bridge.provider_bridge(mock_pretrained_qwen3_next)

        assert provider.hybrid_override_pattern == "GEGEGE*E" * 12
        assert provider.mtp_num_layers == 1
        assert provider.mtp_hybrid_override_pattern == "*E"

    def test_hybrid_pattern_small_model(self):
        """Test pattern generation for a small 8-layer model."""
        bridge = Qwen3NextBridge()
        bridge._hf_model_has_mtp = staticmethod(lambda _: False)

        config = Mock()
        for null_attr in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                          "qk_rope_head_dim", "v_head_dim", "n_routed_experts",
                          "num_local_experts", "num_nextn_predict_layers",
                          "mtp_num_hidden_layers"):
            setattr(config, null_attr, None)
        config.num_hidden_layers = 8
        config.full_attention_interval = 4
        config.hidden_size = 256
        config.intermediate_size = 512
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 64
        config.vocab_size = 1000
        config.max_position_embeddings = 1024
        config.rms_norm_eps = 1e-6
        config.initializer_range = 0.02
        config.rope_theta = 10000
        config.partial_rotary_factor = 0.25
        config.rope_scaling = None
        config.torch_dtype = "bfloat16"
        config.hidden_act = "silu"
        config.attention_dropout = 0.0
        config.hidden_dropout = 0.0
        config.tie_word_embeddings = False
        config.attention_bias = False
        config.mlp_bias = False
        config.use_qk_norm = True
        config.num_experts = 8
        config.num_experts_per_tok = 2
        config.moe_intermediate_size = 64
        config.shared_expert_intermediate_size = 64
        config.linear_conv_kernel_dim = 4
        config.linear_key_head_dim = 32
        config.linear_value_head_dim = 32
        config.linear_num_key_heads = 4
        config.linear_num_value_heads = 8

        mock_pretrained = Mock()
        mock_pretrained.config = config

        provider = bridge.provider_bridge(mock_pretrained)

        # 8 layers, interval=4: layers 0-2 GDN, 3 standard, 4-6 GDN, 7 standard
        assert provider.hybrid_override_pattern == "GEGEGE*EGEGEGE*E"
        assert provider.num_layers == 16

    def test_gdn_layers_at_even_physical_indices(self, bridge_with_config):
        """Test that GDN attention is at physical layer 2*N for non-standard-attn layers."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        # HF layer 0 -> physical 0 (GDN), physical 1 (MoE)
        assert any("decoder.layers.0.self_attention.in_proj.weight" in p for p in params)
        assert any("decoder.layers.1.mlp.router.weight" in p for p in params)

        # HF layer 1 -> physical 2 (GDN), physical 3 (MoE)
        assert any("decoder.layers.2.self_attention.in_proj.weight" in p for p in params)
        assert any("decoder.layers.3.mlp.router.weight" in p for p in params)

    def test_standard_attention_at_interval(self, bridge_with_config):
        """Test that standard attention is at physical layer 2*N where (N+1) % interval == 0."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        # HF layer 3 (full_attention_interval=4) -> physical 6 (standard attn)
        # Standard attention has linear_qkv, not in_proj
        assert any("decoder.layers.6.self_attention.linear_qkv.layer_norm_weight" in p for p in params)
        # Should NOT have GDN in_proj at this position
        assert not any("decoder.layers.6.self_attention.in_proj.weight" in p for p in params)

        # HF layer 7 -> physical 14 (standard attn)
        assert any("decoder.layers.14.self_attention.linear_qkv.layer_norm_weight" in p for p in params)

    def test_mlp_at_odd_physical_indices(self, bridge_with_config):
        """Test that MoE MLP is always at physical layer 2*N+1."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        for n in range(8):
            mlp_idx = 2 * n + 1
            assert any(f"decoder.layers.{mlp_idx}.mlp.router.weight" in p for p in params), (
                f"Missing MoE router at physical layer {mlp_idx} (HF layer {n})"
            )
            assert any(f"decoder.layers.{mlp_idx}.pre_mlp_layernorm.weight" in p for p in params), (
                f"Missing pre_mlp_layernorm at physical layer {mlp_idx} (HF layer {n})"
            )

    def test_no_attention_at_odd_indices(self, bridge_with_config):
        """Test that odd physical indices never have attention parameters."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        for n in range(8):
            odd_idx = 2 * n + 1
            assert not any(f"decoder.layers.{odd_idx}.self_attention." in p for p in params), (
                f"Unexpected attention at physical layer {odd_idx} (should be MLP only)"
            )

    def test_no_mlp_at_even_indices(self, bridge_with_config):
        """Test that even physical indices never have MLP parameters."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        for n in range(8):
            even_idx = 2 * n
            assert not any(f"decoder.layers.{even_idx}.mlp." in p for p in params), (
                f"Unexpected MLP at physical layer {even_idx} (should be attention only)"
            )

    def test_hf_layer_index_mapping(self, bridge_with_config):
        """Test that HF layer indices are correctly mapped to physical indices."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        # Build a map of HF -> physical from the auto mappings
        for n in range(8):
            hf_prefix = f"model.layers.{n}."
            attn_prefix = f"decoder.layers.{2 * n}."
            mlp_prefix = f"decoder.layers.{2 * n + 1}."

            # Find mappings that reference this HF layer
            hf_params = [
                m.hf_param
                for m in registry.mappings
                if hasattr(m, "hf_param") and isinstance(m.hf_param, str) and hf_prefix in m.hf_param
            ]
            megatron_params_for_layer = [
                m.megatron_param
                for m in registry.mappings
                if hasattr(m, "megatron_param")
                and isinstance(m.megatron_param, str)
                and (attn_prefix in m.megatron_param or mlp_prefix in m.megatron_param)
            ]

            assert len(hf_params) > 0, f"No HF mappings for layer {n}"
            assert len(megatron_params_for_layer) > 0, f"No Megatron mappings for layer {n}"

    def test_final_norm_key(self, bridge_with_config):
        """Test that final layernorm uses MambaModel's 'final_norm' key, not 'final_layernorm'."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        assert "decoder.final_norm.weight" in params
        assert "decoder.final_layernorm.weight" not in params

    def test_mtp_uses_standard_attention(self, bridge_with_config):
        """Test that MTP inner layers use standard attention (not GDN)."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        mtp_inner = "mtp.layers.0.mtp_model_layer.layers"

        # MTP layer 0 should have standard attention (linear_qkv), not GDN (in_proj)
        assert any(f"{mtp_inner}.0.self_attention.linear_qkv" in p for p in params)
        assert not any(f"{mtp_inner}.0.self_attention.in_proj" in p for p in params)

        # MTP layer 1 should have MoE
        assert any(f"{mtp_inner}.1.mlp.router.weight" in p for p in params)

    def test_mtp_uses_mtp_model_layer_prefix(self, bridge_with_config):
        """Test that MTP uses 'mtp_model_layer' prefix (not 'decoder' or 'transformer_layer')."""
        registry = bridge_with_config.mapping_registry()
        params = self._all_megatron_params(registry)

        mtp_params = [p for p in params if p.startswith("mtp.layers.0.")]
        inner_layer_params = [p for p in mtp_params if ".layers." in p.replace("mtp.layers.0.", "", 1)]

        for p in inner_layer_params:
            assert "mtp_model_layer.layers." in p, (
                f"MTP inner param '{p}' should use 'mtp_model_layer.layers' prefix"
            )
