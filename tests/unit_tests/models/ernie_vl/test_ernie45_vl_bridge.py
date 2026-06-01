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

"""Unit tests for ERNIE 4.5 VL (Vision-Language) MoE bridge."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.ernie_vl.ernie45_vl_bridge import (
    Ernie45VLBridge,
    _ConcatBiasMapping,
    _OffsetGatedMLPMapping,
    _OffsetRowParallelMapping,
)
from megatron.bridge.models.ernie_vl.ernie45_vl_provider import Ernie45VLModelProvider
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_moe_layer import (
    ErnieMultiTypeMoE,
    MultiTypeMoeSubmodules,
)


def _make_vision_config():
    """Create a mock vision config."""
    vision_config = Mock(spec=[])
    vision_config.hidden_size = 1280
    vision_config.num_attention_heads = 16
    vision_config.num_hidden_layers = 32
    vision_config.patch_size = 14
    vision_config.image_size = 384
    vision_config.intermediate_size = 5120
    return vision_config


def _make_flat_vl_config():
    """Create a flat (auto_map) VL config -- used by Thinking model."""
    config = Mock(spec=[])
    config.hidden_size = 2560
    config.intermediate_size = 6912
    config.num_attention_heads = 20
    config.num_key_value_heads = 4
    config.num_hidden_layers = 28
    config.max_position_embeddings = 32768
    config.rms_norm_eps = 1e-05
    config.rope_theta = 500000.0
    config.tie_word_embeddings = False
    config.torch_dtype = "bfloat16"
    config.vocab_size = 189440
    config.initializer_range = 0.02
    config.model_type = "ernie4_5_moe_vl"
    config.architectures = ["Ernie4_5_VLMoeForConditionalGeneration"]
    # Dual-pool MoE
    config.moe_num_experts = [64, 64]
    config.moe_k = 6
    config.moe_intermediate_size = [1536, 512]
    config.moe_num_shared_experts = 2
    config.router_aux_loss_coef = 0.001
    config.mlp_layer_types = ["dense"] + ["moe"] * 27
    # Vision
    config.vision_config = _make_vision_config()
    # VL tokens
    config.image_token_id = 151859
    config.video_token_id = 151860
    config.image_start_token_id = 101304
    config.image_end_token_id = 101305
    config.video_start_token_id = 101306
    config.video_end_token_id = 101307
    # Flat config: no text_config attribute
    config.text_config = config  # self-reference (as done by _normalize_hf_config)
    # RoPE
    config.rope_parameters = {"rope_theta": 500000.0, "mrope_section": [22, 22, 20]}
    config.rope_scaling = None
    # auto_map
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_ernie4_5_vl.Ernie4_5_VLMoeForConditionalGeneration",
    }
    return config


def _make_nested_vl_config():
    """Create a nested (transformers-builtin) VL config."""
    text_config = Mock(spec=[])
    text_config.hidden_size = 2560
    text_config.intermediate_size = 6912
    text_config.num_attention_heads = 20
    text_config.num_key_value_heads = 4
    text_config.num_hidden_layers = 28
    text_config.max_position_embeddings = 32768
    text_config.rms_norm_eps = 1e-05
    text_config.rope_theta = 500000.0
    text_config.torch_dtype = "bfloat16"
    text_config.vocab_size = 189440
    text_config.initializer_range = 0.02
    text_config.model_type = "ernie4_5_moe"
    # MoE
    text_config.moe_num_experts = 4  # nested config uses smaller num for toy
    text_config.moe_k = 6
    text_config.moe_intermediate_size = [1536, 512]
    text_config.moe_num_shared_experts = 2
    text_config.router_aux_loss_coef = 0.001
    text_config.mlp_layer_types = ["dense"] + ["moe"] * 27
    text_config.rope_parameters = {"rope_theta": 500000.0, "mrope_section": [22, 22, 20]}
    text_config.rope_scaling = None

    config = Mock(spec=[])
    config.model_type = "ernie4_5_vl_moe"
    config.architectures = ["Ernie4_5_VLForConditionalGeneration"]
    config.text_config = text_config
    config.vision_config = _make_vision_config()
    config.tie_word_embeddings = True
    config.image_token_id = 151859
    config.video_token_id = 151860
    config.image_start_token_id = 101304
    config.image_end_token_id = 101305
    config.video_start_token_id = 101306
    config.video_end_token_id = 101307
    return config


@pytest.fixture
def flat_config():
    return _make_flat_vl_config()


@pytest.fixture
def nested_config():
    return _make_nested_vl_config()


@pytest.fixture
def mock_pretrained_flat(flat_config):
    pretrained = Mock()
    pretrained.config = flat_config
    return pretrained


@pytest.fixture
def mock_pretrained_nested(nested_config):
    pretrained = Mock()
    pretrained.config = nested_config
    return pretrained


class TestErnie45VLBridgeRegistration:
    """Test bridge class and registration."""

    def test_is_subclass(self):
        assert issubclass(Ernie45VLBridge, MegatronModelBridge)

    def test_instantiation(self):
        bridge = Ernie45VLBridge()
        assert bridge is not None


class TestErnie45VLBridgeGetTextConfig:
    """Test _get_text_config static method."""

    def test_nested_config_returns_text_config(self, nested_config):
        result = Ernie45VLBridge._get_text_config(nested_config)
        # nested_config.text_config is a distinct object
        assert result is nested_config.text_config
        assert result is not nested_config

    def test_flat_config_returns_self(self, flat_config):
        result = Ernie45VLBridge._get_text_config(flat_config)
        # flat config: text_config is self-reference
        assert result is flat_config


class TestErnie45VLBridgeGetNumExperts:
    """Test _get_num_experts static method."""

    def test_list_input(self):
        config = Mock(spec=[])
        config.moe_num_experts = [64, 64]
        assert Ernie45VLBridge._get_num_experts(config) == 64

    def test_int_input(self):
        config = Mock(spec=[])
        config.moe_num_experts = 4
        assert Ernie45VLBridge._get_num_experts(config) == 4


class TestErnie45VLBridgeProviderBridge:
    """Test provider_bridge method."""

    def test_returns_vl_provider_flat(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert isinstance(provider, Ernie45VLModelProvider)

    def test_returns_vl_provider_nested(self, mock_pretrained_nested):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_nested)
        assert isinstance(provider, Ernie45VLModelProvider)

    def test_basic_dimensions_flat(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.num_layers == 28
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 20

    def test_moe_config_flat(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 6
        assert provider.moe_ffn_hidden_size == 1536  # text expert intermediate
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True

    def test_dual_pool_intermediate_sizes(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.moe_intermediate_size == (1536, 512)

    def test_shared_experts_flat(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        # 1536 * 2 = 3072
        assert provider.moe_shared_expert_intermediate_size == 1536 * 2

    def test_rope_mrope(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.position_embedding_type == "mrope"
        assert provider.rotary_base == 500000.0
        assert provider.rotary_interleaved is True
        assert provider.mrope_section == [22, 22, 20]

    def test_moe_layer_freq(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        expected = [0] + [1] * 27
        assert provider.moe_layer_freq == expected

    def test_token_ids_flat(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.image_token_id == 151859
        assert provider.video_token_id == 151860
        assert provider.image_start_token_id == 101304
        assert provider.image_end_token_id == 101305

    def test_tie_word_embeddings_nested(self, mock_pretrained_nested):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_nested)
        # nested config: tie_word_embeddings comes from top-level config
        assert provider.share_embeddings_and_output_weights is True

    def test_tie_word_embeddings_flat(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.share_embeddings_and_output_weights is False

    def test_normalization(self, mock_pretrained_flat):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.normalization == "RMSNorm"

    def test_vision_config_propagated(self, mock_pretrained_flat, flat_config):
        bridge = Ernie45VLBridge()
        provider = bridge.provider_bridge(mock_pretrained_flat)
        assert provider.vision_config is flat_config.vision_config


class TestErnie45VLBridgeMappingRegistry:
    """Test mapping_registry method."""

    def _get_bridge_with_config(self, config, num_experts=64, is_flat=True):
        """Create bridge with hf_config injected."""
        bridge = Ernie45VLBridge()
        bridge.hf_config = config
        return bridge

    def test_returns_mapping_registry_flat(self, flat_config):
        bridge = self._get_bridge_with_config(flat_config)
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)
        assert len(registry.mappings) > 0

    def test_returns_mapping_registry_nested(self, nested_config):
        bridge = self._get_bridge_with_config(nested_config, num_experts=4, is_flat=False)
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)
        assert len(registry.mappings) > 0

    def test_has_language_model_embeddings(self, flat_config):
        bridge = self._get_bridge_with_config(flat_config)
        registry = bridge.mapping_registry()
        megatron_params = [
            m.megatron_param
            for m in registry.mappings
            if hasattr(m, "megatron_param") and isinstance(m.megatron_param, str)
        ]
        assert "language_model.embedding.word_embeddings.weight" in megatron_params

    def test_has_qkv_mapping(self, flat_config):
        from megatron.bridge.models.conversion.param_mapping import QKVMapping

        bridge = self._get_bridge_with_config(flat_config)
        registry = bridge.mapping_registry()
        qkv_mappings = [m for m in registry.mappings if isinstance(m, QKVMapping)]
        assert len(qkv_mappings) > 0

    def test_has_vision_mappings(self, flat_config):
        bridge = self._get_bridge_with_config(flat_config)
        registry = bridge.mapping_registry()
        hf_params = []
        for m in registry.mappings:
            if hasattr(m, "hf_param") and isinstance(m.hf_param, str):
                hf_params.append(m.hf_param)
        vision_params = [p for p in hf_params if "vision" in p]
        assert len(vision_params) > 0, "Should have vision encoder mappings"

    def test_has_offset_expert_mappings(self, flat_config):
        bridge = self._get_bridge_with_config(flat_config)
        registry = bridge.mapping_registry()
        offset_mappings = [
            m for m in registry.mappings if isinstance(m, (_OffsetGatedMLPMapping, _OffsetRowParallelMapping))
        ]
        assert len(offset_mappings) > 0, "Should have offset expert mappings for vision pool"

    def test_has_concat_bias_mapping(self, flat_config):
        bridge = self._get_bridge_with_config(flat_config)
        registry = bridge.mapping_registry()
        concat_bias_mappings = [m for m in registry.mappings if isinstance(m, _ConcatBiasMapping)]
        assert len(concat_bias_mappings) > 0, "Should have ConcatBiasMapping for expert bias"

    def test_has_shared_expert_mappings(self, flat_config):
        bridge = self._get_bridge_with_config(flat_config)
        registry = bridge.mapping_registry()
        megatron_params = [
            m.megatron_param
            for m in registry.mappings
            if hasattr(m, "megatron_param") and isinstance(m.megatron_param, str)
        ]
        shared = [p for p in megatron_params if "shared_experts" in p]
        assert len(shared) > 0


class TestConcatBiasMapping:
    """Test _ConcatBiasMapping class."""

    def test_clear_export_buffer(self):
        _ConcatBiasMapping.clear_export_buffer()
        # Should not raise

    def test_hf_to_megatron_text_slice(self):
        """Test that text slice extracts row 0 from [2, N] bias."""
        from unittest.mock import patch

        mapping = _ConcatBiasMapping(
            megatron_param="language_model.decoder.layers.*.mlp.text_moe_layer.router.expert_bias",
            hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
            slice_name="text",
            num_experts=4,
        )
        # [2, 4] - row 0 is text, row 1 is vision
        concat_bias = torch.randn(2, 4)
        # Patch AutoMapping.hf_to_megatron to return the sliced input as-is
        with patch(
            "megatron.bridge.models.conversion.param_mapping.AutoMapping.hf_to_megatron",
            side_effect=lambda w, m: w,
        ):
            result = mapping.hf_to_megatron(concat_bias, Mock())
        assert result.shape == (4,)
        assert torch.allclose(result, concat_bias[0])

    def test_hf_to_megatron_vision_slice(self):
        """Test that vision slice extracts row 1 from [2, N] bias."""
        from unittest.mock import patch

        mapping = _ConcatBiasMapping(
            megatron_param="language_model.decoder.layers.*.mlp.vision_moe_layer.router.expert_bias",
            hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
            slice_name="vision",
            num_experts=4,
        )
        concat_bias = torch.randn(2, 4)
        with patch(
            "megatron.bridge.models.conversion.param_mapping.AutoMapping.hf_to_megatron",
            side_effect=lambda w, m: w,
        ):
            result = mapping.hf_to_megatron(concat_bias, Mock())
        assert result.shape == (4,)
        assert torch.allclose(result, concat_bias[1])

    def test_hf_to_megatron_squeeze_2d(self):
        """Test slicing logic with text (row 0)."""
        from unittest.mock import patch

        mapping = _ConcatBiasMapping(
            megatron_param="language_model.decoder.layers.*.mlp.text_moe_layer.router.expert_bias",
            hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
            slice_name="text",
            num_experts=4,
        )
        concat_bias = torch.randn(2, 4)
        with patch(
            "megatron.bridge.models.conversion.param_mapping.AutoMapping.hf_to_megatron",
            side_effect=lambda w, m: w,
        ):
            result = mapping.hf_to_megatron(concat_bias, Mock())
        assert result.shape == (4,)


class TestOffsetMappings:
    """Test _OffsetGatedMLPMapping and _OffsetRowParallelMapping."""

    def test_offset_gated_mlp_mapping_instantiation(self):
        mapping = _OffsetGatedMLPMapping(
            megatron_param="language_model.decoder.layers.*.mlp.vision_moe_layer.experts.local_experts.*.linear_fc1.weight",
            gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            up="model.layers.*.mlp.experts.*.up_proj.weight",
            expert_offset=64,
        )
        assert mapping is not None
        assert mapping._expert_offset == 64

    def test_offset_row_parallel_mapping_instantiation(self):
        mapping = _OffsetRowParallelMapping(
            megatron_param="language_model.decoder.layers.*.mlp.vision_moe_layer.experts.local_experts.*.linear_fc2.weight",
            hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
            expert_offset=64,
        )
        assert mapping is not None
        assert mapping._expert_offset == 64

    def test_offset_gated_mlp_mapping_resolve(self):
        """Test that resolve offsets the expert index."""
        mapping = _OffsetGatedMLPMapping(
            megatron_param="language_model.decoder.layers.*.mlp.vision_moe_layer.experts.local_experts.*.linear_fc1.weight",
            gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            up="model.layers.*.mlp.experts.*.up_proj.weight",
            expert_offset=64,
        )
        # Resolve with captures: (layer_idx, local_expert_idx)
        resolved = mapping.resolve(("1", "0"))
        # The HF expert index should be offset: local_expert 0 -> HF expert 64
        assert "64" in resolved.hf_param["gate"]
        assert "64" in resolved.hf_param["up"]

    def test_offset_row_parallel_mapping_resolve(self):
        """Test that resolve offsets the expert index for row parallel."""
        mapping = _OffsetRowParallelMapping(
            megatron_param="language_model.decoder.layers.*.mlp.vision_moe_layer.experts.local_experts.*.linear_fc2.weight",
            hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
            expert_offset=64,
        )
        resolved = mapping.resolve(("1", "0"))
        # The resolved HF param should reference expert 64
        assert "64" in resolved.hf_param


class TestErnie45VLModelProvider:
    """Test Ernie45VLModelProvider class."""

    def test_instantiation(self):
        provider = Ernie45VLModelProvider()
        assert provider is not None

    def test_has_expected_fields(self):
        provider = Ernie45VLModelProvider()
        # Should have standard GPT fields
        assert hasattr(provider, "num_layers")
        assert hasattr(provider, "hidden_size")
        assert hasattr(provider, "num_attention_heads")


class TestErnieMultiTypeMoE:
    """Test ERNIE VL dual-pool MoE construction."""

    def test_accepts_transformer_layer_kwargs(self):
        config = SimpleNamespace(moe_intermediate_size=(64, 32), moe_shared_expert_intermediate_size=128)
        submodules = MultiTypeMoeSubmodules(
            text_moe_layer=object(),
            vision_moe_layer=object(),
            shared_experts=object(),
        )
        pg_collection = object()

        with patch("megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_moe_layer.build_module") as build_module:
            build_module.side_effect = [Mock(), Mock(), Mock()]
            layer = ErnieMultiTypeMoE(
                config=config,
                submodules=submodules,
                layer_number=2,
                pg_collection=pg_collection,
                is_mtp_layer=True,
                name="decoder.layers.1.mlp",
            )

        assert layer.layer_number == 2
        assert layer.is_mtp_layer is True

        text_call, vision_call, shared_call = build_module.call_args_list
        assert text_call.args[0] is submodules.text_moe_layer
        assert text_call.args[1].moe_ffn_hidden_size == 64
        assert text_call.kwargs["pg_collection"] is pg_collection
        assert text_call.kwargs["is_mtp_layer"] is True
        assert text_call.kwargs["name"] == "decoder.layers.1.mlp.text_moe_layer"

        assert vision_call.args[0] is submodules.vision_moe_layer
        assert vision_call.args[1].moe_ffn_hidden_size == 32
        assert vision_call.kwargs["pg_collection"] is pg_collection
        assert vision_call.kwargs["is_mtp_layer"] is True
        assert vision_call.kwargs["name"] == "decoder.layers.1.mlp.vision_moe_layer"

        assert shared_call.args[0] is submodules.shared_experts
        assert shared_call.kwargs["config"] is config
        assert shared_call.kwargs["pg_collection"] is pg_collection
        assert shared_call.kwargs["gate"] is False
        assert shared_call.kwargs["name"] == "decoder.layers.1.mlp.shared_experts"
        assert "is_mtp_layer" not in shared_call.kwargs
