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

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider


@pytest.fixture
def mock_text_config():
    """Create a mock text config for Kimi K2.5."""

    _KIMI_TEXT_ATTRS = [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "n_routed_experts",
        "moe_intermediate_size",
        "n_shared_experts",
        "first_k_dense_replace",
        "num_experts_per_tok",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "vocab_size",
        "rope_theta",
        "rope_scaling",
        "initializer_range",
        "rms_norm_eps",
        "hidden_act",
        "max_position_embeddings",
        "torch_dtype",
        "bos_token_id",
        "eos_token_id",
        "aux_loss_alpha",
        "generation_config",
    ]
    config = Mock(spec=_KIMI_TEXT_ATTRS)
    config.num_hidden_layers = 61
    config.hidden_size = 7168
    config.intermediate_size = 18432
    config.num_attention_heads = 64
    config.num_key_value_heads = 64
    config.q_lora_rank = 1536
    config.kv_lora_rank = 512
    config.qk_nope_head_dim = 128
    config.qk_rope_head_dim = 64
    config.v_head_dim = 128
    config.n_routed_experts = 384
    config.moe_intermediate_size = 2048
    config.n_shared_experts = 1
    config.first_k_dense_replace = 1
    config.num_experts_per_tok = 8
    config.n_group = 1
    config.topk_group = 1
    config.routed_scaling_factor = 2.827
    config.vocab_size = 163840
    config.rope_theta = 50000.0
    config.rope_scaling = {"type": "yarn", "factor": 32, "mscale": 1.0, "mscale_all_dim": 1.0}
    config.initializer_range = 0.006
    config.rms_norm_eps = 1e-6
    config.hidden_act = "silu"
    config.max_position_embeddings = 131072
    config.torch_dtype = "bfloat16"
    config.bos_token_id = 163584
    config.eos_token_id = 163585
    config.aux_loss_alpha = 1e-3
    config.generation_config = None
    return config


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for Kimi K2.5 VL."""
    config = Mock()
    config.hidden_size = 1152
    config.merge_kernel_size = (2, 2)
    return config


@pytest.fixture
def mock_hf_config(mock_text_config, mock_vision_config):
    """Create a mock HF config for Kimi K2.5 VL."""
    config = Mock()
    config.text_config = mock_text_config
    config.vision_config = mock_vision_config
    config.media_placeholder_token_id = 163605
    config.pad_token_id = 163839
    config.ignore_index = -100
    config.torch_dtype = "bfloat16"
    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    pretrained._model_name_or_path = "/path/to/kimi_k25_vl"
    pretrained.generation_config = None
    return pretrained


@pytest.fixture
def kimi_bridge():
    """Create a KimiK25VLBridge instance."""
    return KimiK25VLBridge()


class TestKimiK25VLBridgeInitialization:
    """Test KimiK25VLBridge initialization."""

    def test_bridge_initialization(self, kimi_bridge):
        """Test bridge can be initialized."""
        assert isinstance(kimi_bridge, KimiK25VLBridge)

    def test_bridge_has_required_methods(self, kimi_bridge):
        """Test bridge has required methods."""
        assert hasattr(kimi_bridge, "provider_bridge")
        assert callable(kimi_bridge.provider_bridge)
        assert hasattr(kimi_bridge, "mapping_registry")
        assert callable(kimi_bridge.mapping_registry)


class TestKimiK25VLBridgeProviderBridge:
    """Test provider_bridge method."""

    def test_provider_bridge_basic_config(self, kimi_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, KimiK25VLModelProvider)
        assert provider.num_layers == 61
        assert provider.hidden_size == 7168
        assert provider.vocab_size == 163840

    def test_provider_bridge_moe_config(self, kimi_bridge, mock_hf_pretrained):
        """Test MoE configuration extraction."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_moe_experts == 384
        assert provider.moe_ffn_hidden_size == 2048
        assert provider.moe_router_topk == 8
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True

    def test_provider_bridge_moe_layer_freq(self, kimi_bridge, mock_hf_pretrained):
        """Test MoE layer frequency computation."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)

        # first_k_dense_replace=1: first layer is dense, rest are MoE
        expected = [0] + [1] * 60
        assert provider.moe_layer_freq == expected

    def test_provider_bridge_mla_params(self, kimi_bridge, mock_hf_pretrained):
        """Test MLA parameter extraction."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.multi_latent_attention is True
        assert provider.q_lora_rank == 1536
        assert provider.kv_lora_rank == 512
        assert provider.qk_head_dim == 128
        assert provider.qk_pos_emb_head_dim == 64
        assert provider.v_head_dim == 128

    def test_provider_bridge_token_ids(self, kimi_bridge, mock_hf_pretrained):
        """Test token ID extraction."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.media_placeholder_token_id == 163605
        assert provider.bos_token_id == 163584
        assert provider.eos_token_id == 163585
        assert provider.pad_token_id == 163839

    def test_provider_bridge_vision_config(self, kimi_bridge, mock_hf_pretrained):
        """Test vision config is stored on provider."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.vision_config is mock_hf_pretrained.config.vision_config

    def test_provider_bridge_hf_model_path(self, kimi_bridge, mock_hf_pretrained):
        """Test HF model path is stored on provider."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.hf_model_path == "/path/to/kimi_k25_vl"

    def test_provider_bridge_different_hidden_sizes(self, kimi_bridge, mock_hf_pretrained):
        """Test provider_bridge with different hidden sizes."""
        for hidden_size in [4096, 7168, 8192]:
            mock_hf_pretrained.config.text_config.hidden_size = hidden_size
            provider = kimi_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.hidden_size == hidden_size

    def test_provider_bridge_different_expert_counts(self, kimi_bridge, mock_hf_pretrained):
        """Test provider_bridge with different expert counts."""
        for n_experts in [64, 128, 384]:
            mock_hf_pretrained.config.text_config.n_routed_experts = n_experts
            provider = kimi_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.num_moe_experts == n_experts

    def test_provider_bridge_shared_expert_intermediate_size(self, kimi_bridge, mock_hf_pretrained):
        """Test shared expert intermediate size computation."""
        provider = kimi_bridge.provider_bridge(mock_hf_pretrained)
        # moe_intermediate_size (2048) * n_shared_experts (1)
        assert provider.moe_shared_expert_intermediate_size == 2048


class TestKimiK25VLBridgeMappingRegistry:
    """Test mapping_registry method."""

    def test_mapping_registry_type(self, kimi_bridge):
        """Test mapping_registry returns correct type."""
        registry = kimi_bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_has_language_model_prefix(self, kimi_bridge):
        """Test all language model mappings have language_model prefix."""
        registry = kimi_bridge.mapping_registry()
        for mapping in registry.mappings:
            if isinstance(mapping, AutoMapping):
                if "vision_tower" not in mapping.hf_param and "mm_projector" not in mapping.hf_param:
                    assert "language_model" in mapping.hf_param, f"Language mapping missing prefix: {mapping.hf_param}"

    def test_mapping_registry_has_vision_mappings(self, kimi_bridge):
        """Test registry includes vision tower and projector mappings."""
        registry = kimi_bridge.mapping_registry()

        replicated_params = [m.megatron_param for m in registry.mappings if isinstance(m, ReplicatedMapping)]
        assert "vision_tower.**" in replicated_params
        assert "mm_projector.**" in replicated_params

    def test_mapping_registry_expert_bias(self, kimi_bridge):
        """Test registry includes expert bias mapping."""
        registry = kimi_bridge.mapping_registry()

        auto_hf_params = [m.hf_param for m in registry.mappings if isinstance(m, AutoMapping)]
        assert any("e_score_correction_bias" in p for p in auto_hf_params)


class TestKimiK25VLBridgeWeightConversion:
    """Test weight conversion helpers."""

    def test_is_quantized_expert_key_true(self, kimi_bridge):
        """Test _is_quantized_expert_key for expert weights."""
        assert kimi_bridge._is_quantized_expert_key("model.layers.5.mlp.experts.10.gate_proj.weight") is True
        assert kimi_bridge._is_quantized_expert_key("model.layers.5.mlp.experts.10.up_proj.weight") is True

    def test_is_quantized_expert_key_false_for_shared(self, kimi_bridge):
        """Test _is_quantized_expert_key returns False for shared experts."""
        assert kimi_bridge._is_quantized_expert_key("model.layers.5.mlp.shared_experts.gate_proj.weight") is False

    def test_is_quantized_expert_key_false_for_dense(self, kimi_bridge):
        """Test _is_quantized_expert_key returns False for dense layer 0."""
        assert kimi_bridge._is_quantized_expert_key("model.layers.0.mlp.experts.10.gate_proj.weight") is False

    def test_is_quantized_expert_key_false_for_non_expert(self, kimi_bridge):
        """Test _is_quantized_expert_key returns False for non-expert keys."""
        assert kimi_bridge._is_quantized_expert_key("model.layers.5.self_attn.q_proj.weight") is False
        assert kimi_bridge._is_quantized_expert_key("model.embed_tokens.weight") is False

    @patch("megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge.dequantize_int4")
    def test_load_and_dequantize_uses_source_tensor_device(self, mock_dequantize, kimi_bridge):
        """Quantized loads should dequantize on the source tensor device."""
        packed = torch.zeros((2, 1), dtype=torch.int32)
        scale = torch.ones((2, 1), dtype=torch.float16)
        shape = torch.tensor([2, 8], dtype=torch.int64)
        mock_dequantize.return_value = torch.zeros((2, 8), dtype=torch.bfloat16)

        kimi_bridge._load_and_dequantize(
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight",
            {
                "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_packed": packed,
                "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_scale": scale,
                "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_shape": shape,
            },
        )

        _, kwargs = mock_dequantize.call_args
        assert kwargs["device"] == packed.device
