from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.step_vl.modeling_step3_vl.configuration import (
    StepRoboticsVisionEncoderConfig,
)
from megatron.bridge.models.step_vl.step3_vl_bridge import Step3VLBridge
from megatron.bridge.models.step_vl.step3_vl_provider import Step3VLModelProvider


@pytest.fixture
def mock_text_config():
    """Mock Qwen3-8B text config matching stepfun-ai/Step3-VL-10B."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 36
    cfg.hidden_size = 4096
    cfg.intermediate_size = 8192
    cfg.num_attention_heads = 32
    cfg.num_key_value_heads = 8
    cfg.initializer_range = 0.02
    cfg.rms_norm_eps = 1e-6
    cfg.vocab_size = 152064
    cfg.max_position_embeddings = 32768
    cfg.rope_theta = 1000000.0
    cfg.hidden_act = "silu"
    cfg.torch_dtype = "bfloat16"
    return cfg


@pytest.fixture
def mock_vision_config():
    """Mock StepRobotics vision encoder config."""
    return StepRoboticsVisionEncoderConfig(
        width=1536,
        layers=47,
        heads=16,
        image_size=728,
        patch_size=14,
    )


@pytest.fixture
def mock_hf_config(mock_text_config, mock_vision_config):
    cfg = Mock()
    cfg.text_config = mock_text_config
    cfg.vision_config = mock_vision_config
    cfg.tie_word_embeddings = False
    cfg.projector_bias = False
    cfg.image_token_id = 151679
    return cfg


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    return pretrained


@pytest.fixture
def bridge():
    return Step3VLBridge()


class TestStep3VLBridgeInitialization:
    def test_bridge_initialization(self, bridge):
        assert isinstance(bridge, Step3VLBridge)

    def test_bridge_has_required_methods(self, bridge):
        assert callable(bridge.provider_bridge)
        assert callable(bridge.mapping_registry)


class TestStep3VLBridgeProviderBridge:
    def test_returns_correct_provider_type(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert isinstance(provider, Step3VLModelProvider)

    def test_language_model_dims(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.num_layers == 36
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 8192
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8

    def test_qwen3_specific_flags(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_vision_config_passed_through(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert isinstance(provider.vision_config, StepRoboticsVisionEncoderConfig)
        assert provider.vision_config.width == 1536
        assert provider.vision_config.layers == 47

    def test_vlm_specific_fields(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.projector_bias is False
        assert provider.image_token_id == 151679
        assert provider.scatter_embedding_sequence_parallel is False

    def test_tie_word_embeddings_false(self, bridge, mock_hf_pretrained):
        mock_hf_pretrained.config.tie_word_embeddings = False
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.share_embeddings_and_output_weights is False

    def test_tie_word_embeddings_true(self, bridge, mock_hf_pretrained):
        mock_hf_pretrained.config.tie_word_embeddings = True
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.share_embeddings_and_output_weights is True

    def test_custom_image_token_id(self, bridge, mock_hf_pretrained):
        mock_hf_pretrained.config.image_token_id = 99999
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.image_token_id == 99999

    def test_projector_bias_true(self, bridge, mock_hf_pretrained):
        mock_hf_pretrained.config.projector_bias = True
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.projector_bias is True

    def test_different_layer_counts(self, bridge, mock_hf_pretrained):
        for n in [12, 24, 36]:
            mock_hf_pretrained.config.text_config.num_hidden_layers = n
            provider = bridge.provider_bridge(mock_hf_pretrained)
            assert provider.num_layers == n

    def test_missing_optional_fields_use_defaults(self, bridge, mock_hf_pretrained):
        del mock_hf_pretrained.config.projector_bias
        del mock_hf_pretrained.config.image_token_id
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.projector_bias is False
        assert provider.image_token_id == 151679


class TestStep3VLBridgeMappingRegistry:
    def test_returns_correct_type(self, bridge):
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

    def test_contains_vision_model_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("vision_model" in n for n in names)

    def test_contains_projector_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("vit_large_projector" in n for n in names)

    def test_contains_word_embeddings(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("word_embeddings" in n or "embed_tokens" in n for n in names)

    def test_contains_output_layer(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("output_layer" in n or "lm_head" in n for n in names)

    def test_contains_qkv_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("linear_qkv" in n for n in names)

    def test_contains_gated_mlp_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("linear_fc1" in n for n in names)

    def test_contains_mlp_down_proj(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("linear_fc2" in n or "down_proj" in n for n in names)

    def test_contains_qk_layernorm(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("q_layernorm" in n or "q_norm" in n for n in names)
        assert any("k_layernorm" in n or "k_norm" in n for n in names)

    def test_contains_final_norm(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("final_layernorm" in n or "norm.weight" in n for n in names)
