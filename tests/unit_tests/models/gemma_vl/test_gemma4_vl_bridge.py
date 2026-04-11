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

import math
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig, SiglipVisionConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_text_config_moe():
    """Mock text config for Gemma 4 26B-A4B (MoE model)."""
    config = Mock(spec=[])
    config.num_hidden_layers = 62
    config.hidden_size = 2816
    config.intermediate_size = 2112         # shared expert FFN size
    config.moe_intermediate_size = 704      # routed expert FFN size
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.head_dim = 256
    config.global_head_dim = 512
    config.num_global_key_value_heads = 2
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 262144
    config.max_position_embeddings = 131072
    config.sliding_window = 1024
    config.rope_theta = 1000000.0
    config.query_pre_attn_scalar = 1.0      # not used for scale (softmax_scale=1.0)
    config.rope_scaling = None
    config.rope_local_base_freq = 10000.0
    config.rope_parameters = {"rope_local_base_freq": 10000.0}
    config.hidden_act = "gelu_pytorch_tanh"
    config.torch_dtype = "bfloat16"
    # MoE fields
    config.enable_moe_block = True
    config.num_experts = 128
    config.top_k_experts = 8
    # Attention pattern
    config.layer_types = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
    config.final_logit_softcapping = 30.0
    return config


@pytest.fixture
def mock_text_config_dense():
    """Mock text config for Gemma 4 2B (dense e2b model)."""
    config = Mock(spec=[])
    config.num_hidden_layers = 35
    config.hidden_size = 2048
    config.intermediate_size = 6144
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.head_dim = 256
    config.global_head_dim = 512
    config.num_global_key_value_heads = 2
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 262144
    config.max_position_embeddings = 131072
    config.sliding_window = 1024
    config.rope_theta = 1000000.0
    config.query_pre_attn_scalar = 1.0
    config.rope_scaling = None
    config.rope_local_base_freq = 10000.0
    config.rope_parameters = {"rope_local_base_freq": 10000.0}
    config.hidden_act = "gelu_pytorch_tanh"
    config.torch_dtype = "bfloat16"
    # Dense (no MoE)
    config.enable_moe_block = False
    config.use_double_wide_mlp = False
    config.layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    config.final_logit_softcapping = 30.0
    return config


@pytest.fixture
def mock_vision_config():
    """Mock vision config for Gemma 4 VL."""
    config = SiglipVisionConfig()
    config.hidden_size = 1152
    config.intermediate_size = 4304
    config.num_hidden_layers = 27
    config.num_attention_heads = 16
    config.patch_size = 14
    config.image_size = 896
    return config


@pytest.fixture
def mock_hf_config_moe(mock_text_config_moe, mock_vision_config):
    config = Mock()
    config.text_config = mock_text_config_moe
    config.vision_config = mock_vision_config
    config.vision_soft_tokens_per_image = 280
    config.bos_token_id = 2
    config.eos_token_id = 1
    config.image_token_id = 258_880
    config.video_token_id = 258_884
    return config


@pytest.fixture
def mock_hf_config_dense(mock_text_config_dense, mock_vision_config):
    config = Mock()
    config.text_config = mock_text_config_dense
    config.vision_config = mock_vision_config
    config.vision_soft_tokens_per_image = 280
    config.bos_token_id = 2
    config.eos_token_id = 1
    config.image_token_id = 258_880
    config.video_token_id = 258_884
    return config


@pytest.fixture
def mock_hf_pretrained_moe(mock_hf_config_moe):
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config_moe
    pretrained.generation_config = GenerationConfig()
    return pretrained


@pytest.fixture
def mock_hf_pretrained_dense(mock_hf_config_dense):
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config_dense
    pretrained.generation_config = GenerationConfig()
    return pretrained


@pytest.fixture
def bridge():
    return Gemma4VLBridge()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestGemma4VLBridgeInitialization:
    def test_bridge_initialization(self, bridge):
        assert isinstance(bridge, Gemma4VLBridge)

    def test_bridge_has_required_methods(self, bridge):
        assert callable(getattr(bridge, "provider_bridge", None))
        assert callable(getattr(bridge, "mapping_registry", None))


# ---------------------------------------------------------------------------
# provider_bridge — MoE model
# ---------------------------------------------------------------------------


class TestGemma4VLBridgeProviderBridgeMoE:
    def test_returns_provider(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert isinstance(provider, Gemma4VLModelProvider)

    def test_basic_transformer_config(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.num_layers == 62
        assert provider.hidden_size == 2816
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 4
        assert provider.kv_channels == 256
        assert provider.init_method_std == 0.02
        assert provider.layernorm_epsilon == 1e-6
        assert provider.vocab_size == 262144
        assert provider.seq_length == 131072
        assert provider.window_size == 1024

    def test_moe_config(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.num_moe_experts == 128
        assert provider.moe_router_topk == 8
        assert provider.moe_ffn_hidden_size == 704
        assert provider.moe_shared_expert_intermediate_size == 2112
        assert provider.moe_layer_freq == 1

    def test_softmax_scale_is_one(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.softmax_scale == 1.0

    def test_vl_specific_config(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.image_token_id == 258_880
        assert provider.video_token_id == 258_884
        assert provider.bos_token_id == 2
        assert provider.eos_token_id == 1
        assert provider.vision_soft_tokens_per_image == 280

    def test_dtype_is_bf16(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_global_head_config(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.global_head_dim == 512
        assert provider.num_global_key_value_heads == 2

    def test_qk_layernorm_enabled(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.qk_layernorm is True

    def test_logit_softcapping(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.final_logit_softcapping == 30.0

    def test_vision_config_set(self, bridge, mock_hf_pretrained_moe):
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.vision_config is mock_hf_pretrained_moe.config.vision_config
        assert provider.text_config is mock_hf_pretrained_moe.config.text_config


# ---------------------------------------------------------------------------
# provider_bridge — Dense model
# ---------------------------------------------------------------------------


class TestGemma4VLBridgeProviderBridgeDense:
    def test_returns_provider(self, bridge, mock_hf_pretrained_dense):
        provider = bridge.provider_bridge(mock_hf_pretrained_dense)
        assert isinstance(provider, Gemma4VLModelProvider)

    def test_no_moe_experts(self, bridge, mock_hf_pretrained_dense):
        provider = bridge.provider_bridge(mock_hf_pretrained_dense)
        assert provider.num_moe_experts is None

    def test_dense_ffn_hidden_size(self, bridge, mock_hf_pretrained_dense):
        provider = bridge.provider_bridge(mock_hf_pretrained_dense)
        assert provider.ffn_hidden_size == 6144

    def test_moe_layer_freq_all_zeros(self, bridge, mock_hf_pretrained_dense):
        provider = bridge.provider_bridge(mock_hf_pretrained_dense)
        # All-zero list: no MoE layers
        assert isinstance(provider.moe_layer_freq, list)
        assert all(f == 0 for f in provider.moe_layer_freq)
        assert len(provider.moe_layer_freq) == 35

    def test_is_moe_model_false(self, bridge, mock_hf_pretrained_dense):
        bridge.hf_config = mock_hf_pretrained_dense.config
        assert bridge._is_moe_model() is False

    def test_is_moe_model_true(self, bridge, mock_hf_pretrained_moe):
        bridge.hf_config = mock_hf_pretrained_moe.config
        assert bridge._is_moe_model() is True


# ---------------------------------------------------------------------------
# mapping_registry
# ---------------------------------------------------------------------------


class TestGemma4VLBridgeMappingRegistry:
    def test_returns_registry(self, bridge):
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

    def test_has_mappings(self, bridge):
        registry = bridge.mapping_registry()
        assert len(registry.mappings) > 0

    def _collect_names(self, registry):
        names = []
        for m in registry.mappings:
            if hasattr(m, "megatron_param"):
                names.append(str(m.megatron_param))
            hf = getattr(m, "hf_param", None)
            if isinstance(hf, dict):
                names.extend(str(v) for v in hf.values())
            elif isinstance(hf, str):
                names.append(hf)
        return names

    def test_has_embeddings_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("embed_tokens" in n or "word_embeddings" in n for n in names)

    def test_has_norm_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("norm" in n for n in names)

    def test_has_vision_tower_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("vision_tower" in n for n in names)

    def test_has_embed_vision_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("embed_vision" in n for n in names)

    def test_has_qkv_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("linear_qkv" in n for n in names)

    def test_has_mlp_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("mlp" in n for n in names)

    def test_has_shared_expert_layernorm(self, bridge, mock_hf_config_moe):
        # MoE-specific mappings require hf_config to be set
        bridge.hf_config = mock_hf_config_moe
        names = self._collect_names(bridge.mapping_registry())
        assert any("post_shared_expert_layernorm" in n for n in names)

    def test_has_post_moe_layernorm(self, bridge, mock_hf_config_moe):
        bridge.hf_config = mock_hf_config_moe
        names = self._collect_names(bridge.mapping_registry())
        assert any("post_moe_layernorm" in n for n in names)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGemma4VLBridgeEdgeCases:
    def test_custom_token_ids(self, bridge, mock_hf_pretrained_moe):
        mock_hf_pretrained_moe.config.image_token_id = 99999
        mock_hf_pretrained_moe.config.bos_token_id = 42
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.image_token_id == 99999
        assert provider.bos_token_id == 42

    def test_default_image_token_id(self, bridge, mock_hf_pretrained_moe):
        del mock_hf_pretrained_moe.config.image_token_id
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.image_token_id == 258_880

    def test_default_vision_soft_tokens(self, bridge, mock_hf_pretrained_moe):
        del mock_hf_pretrained_moe.config.vision_soft_tokens_per_image
        provider = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert provider.vision_soft_tokens_per_image == 280

    def test_different_vocab_sizes(self, bridge, mock_hf_pretrained_moe):
        for vocab_size in [256000, 262144, 300000]:
            mock_hf_pretrained_moe.config.text_config.vocab_size = vocab_size
            provider = bridge.provider_bridge(mock_hf_pretrained_moe)
            assert provider.vocab_size == vocab_size

    def test_different_layer_counts(self, bridge, mock_hf_pretrained_moe):
        for num_layers in [32, 46, 62]:
            mock_hf_pretrained_moe.config.text_config.num_hidden_layers = num_layers
            provider = bridge.provider_bridge(mock_hf_pretrained_moe)
            assert provider.num_layers == num_layers
