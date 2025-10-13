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
Unit tests for GLM 4.5 bridges.
"""

from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.glm.glm45_bridge import GLM45Bridge
from megatron.bridge.models.glm.glm45_provider import GLMMoEModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestGLM45Bridge:
    """Test cases for GLM45Bridge."""

    @pytest.fixture
    def glm45_355b_config(self):
        """Mock config for GLM 4.5 355B model."""
        return {
            "architectures": ["Glm4MoeForCausalLM"],
            "attention_bias": True,
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_glm.Glm4Config",
                "AutoModel": "modeling_glm.Glm4MoeModel",
                "AutoModelForCausalLM": "modeling_glm.Glm4MoeForCausalLM",
            },
            "bos_token_id": 151329,
            "eos_token_id": 151336,
            "first_k_dense_replace": 3,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "max_position_embeddings": 131072,
            "model_type": "glm",
            "moe_intermediate_size": 1536,
            "n_routed_experts": 160,
            "n_shared_experts": 1,
            "num_attention_heads": 96,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 92,
            "num_key_value_heads": 8,
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "routed_scaling_factor": 2.5,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_qk_norm": True,
            "vocab_size": 151552,
        }

    @pytest.fixture
    def glm45_air_106b_config(self):
        """Mock config for GLM 4.5 Air 106B model."""
        return {
            "architectures": ["Glm4MoeForCausalLM"],
            "attention_bias": True,
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_glm.Glm4Config",
                "AutoModel": "modeling_glm.Glm4MoeModel",
                "AutoModelForCausalLM": "modeling_glm.Glm4MoeForCausalLM",
            },
            "bos_token_id": 151329,
            "eos_token_id": 151336,
            "first_k_dense_replace": 1,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 10944,
            "max_position_embeddings": 131072,
            "model_type": "glm",
            "moe_intermediate_size": 1408,
            "n_routed_experts": 128,
            "n_shared_experts": 1,
            "num_attention_heads": 96,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 46,
            "num_key_value_heads": 8,
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "routed_scaling_factor": 2.5,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_qk_norm": False,
            "vocab_size": 151552,
        }

    @pytest.fixture
    def mock_pretrained_355b(self, glm45_355b_config):
        """Create mock pretrained model for GLM 4.5 355B."""
        cfg = Mock()
        for k, v in glm45_355b_config.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)
        return m

    @pytest.fixture
    def mock_pretrained_air_106b(self, glm45_air_106b_config):
        """Create mock pretrained model for GLM 4.5 Air 106B."""
        cfg = Mock()
        for k, v in glm45_air_106b_config.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)
        return m

    def test_registration(self):
        """Test that GLM45Bridge is properly registered as a MegatronModelBridge."""
        assert issubclass(GLM45Bridge, MegatronModelBridge)

    def test_provider_bridge_maps_config_355b(self, mock_pretrained_355b):
        """Test provider bridge correctly maps config for GLM 4.5 355B."""
        bridge = GLM45Bridge()
        provider = bridge.provider_bridge(mock_pretrained_355b)

        assert isinstance(provider, GLMMoEModelProvider)
        assert provider.hidden_size == mock_pretrained_355b.config.hidden_size
        assert provider.num_attention_heads == mock_pretrained_355b.config.num_attention_heads
        assert provider.ffn_hidden_size == mock_pretrained_355b.config.intermediate_size
        assert provider.vocab_size == mock_pretrained_355b.config.vocab_size
        assert provider.layernorm_epsilon == mock_pretrained_355b.config.rms_norm_eps
        assert provider.rotary_base == mock_pretrained_355b.config.rope_theta
        assert provider.rotary_percent == mock_pretrained_355b.config.partial_rotary_factor
        assert provider.kv_channels == mock_pretrained_355b.config.head_dim
        assert provider.seq_length == mock_pretrained_355b.config.max_position_embeddings
        assert provider.init_method_std == mock_pretrained_355b.config.initializer_range

        # MoE specific
        assert provider.num_moe_experts == mock_pretrained_355b.config.n_routed_experts
        assert provider.moe_ffn_hidden_size == mock_pretrained_355b.config.moe_intermediate_size
        assert provider.moe_shared_expert_intermediate_size == mock_pretrained_355b.config.moe_intermediate_size
        assert provider.moe_router_topk_scaling_factor == mock_pretrained_355b.config.routed_scaling_factor
        assert provider.moe_router_topk == mock_pretrained_355b.config.num_experts_per_tok
        assert provider.num_layers == mock_pretrained_355b.config.num_hidden_layers
        assert provider.num_query_groups == mock_pretrained_355b.config.num_key_value_heads
        assert provider.qk_layernorm == mock_pretrained_355b.config.use_qk_norm
        assert provider.add_qkv_bias == mock_pretrained_355b.config.attention_bias

        # Test moe_layer_freq (first 3 layers are dense, rest are MoE)
        expected_freq = [0] * 3 + [1] * (92 - 3)
        assert provider.moe_layer_freq == expected_freq

        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_provider_bridge_maps_config_air_106b(self, mock_pretrained_air_106b):
        """Test provider bridge correctly maps config for GLM 4.5 Air 106B."""
        bridge = GLM45Bridge()
        provider = bridge.provider_bridge(mock_pretrained_air_106b)

        assert isinstance(provider, GLMMoEModelProvider)
        assert provider.hidden_size == mock_pretrained_air_106b.config.hidden_size
        assert provider.num_attention_heads == mock_pretrained_air_106b.config.num_attention_heads
        assert provider.ffn_hidden_size == mock_pretrained_air_106b.config.intermediate_size
        assert provider.vocab_size == mock_pretrained_air_106b.config.vocab_size
        assert provider.num_layers == mock_pretrained_air_106b.config.num_hidden_layers
        assert provider.qk_layernorm == mock_pretrained_air_106b.config.use_qk_norm

        # Test moe_layer_freq (first 1 layer is dense, rest are MoE)
        expected_freq = [0] * 1 + [1] * (46 - 1)
        assert provider.moe_layer_freq == expected_freq

        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_mapping_registry_exists(self):
        """Test that mapping registry is properly defined."""
        bridge = GLM45Bridge()
        registry = bridge.mapping_registry()

        # Verify registry has mappings
        assert len(registry.mappings) > 0

        # Verify some key mappings exist
        mapping_dict = {m.megatron_param: m.hf_param for m in registry.mappings}

        # Check embedding mapping
        assert "embedding.word_embeddings.weight" in mapping_dict
        assert mapping_dict["embedding.word_embeddings.weight"] == "model.embed_tokens.weight"

        # Check final layernorm mapping
        assert "decoder.final_layernorm.weight" in mapping_dict
        assert mapping_dict["decoder.final_layernorm.weight"] == "model.norm.weight"

        # Check LM head mapping
        assert "output_layer.weight" in mapping_dict
        assert mapping_dict["output_layer.weight"] == "lm_head.weight"

    def test_dtype_mapping_fp16(self, glm45_355b_config):
        """Test dtype mapping for FP16 models."""
        glm45_355b_config["torch_dtype"] = "float16"

        cfg = Mock()
        for k, v in glm45_355b_config.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)

        bridge = GLM45Bridge()
        provider = bridge.provider_bridge(m)

        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    def test_dtype_mapping_default(self, glm45_355b_config):
        """Test dtype mapping defaults to float32 when not specified."""
        del glm45_355b_config["torch_dtype"]

        cfg = Mock()
        for k, v in glm45_355b_config.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)

        bridge = GLM45Bridge()
        provider = bridge.provider_bridge(m)

        assert provider.fp16 is False
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float32
