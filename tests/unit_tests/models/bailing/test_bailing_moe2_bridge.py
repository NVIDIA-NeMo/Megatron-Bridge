# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.bridge.models.bailing.bailing_moe2_bridge import BailingMoeV2Bridge


pytestmark = pytest.mark.unit


def test_mtp_mappings_resolve_live_mcore_parameter_names() -> None:
    bridge = BailingMoeV2Bridge()
    bridge.hf_config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    registry = bridge.mapping_registry()

    mtp_parameter_names = [
        "mtp.layers.0.mtp_model_layer.input_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.pre_mlp_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight",
        "mtp.layers.0.mtp_model_layer.mlp.router.weight",
        "mtp.layers.0.mtp_model_layer.mlp.router.expert_bias",
        "mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1.weight0",
        "mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2.weight0",
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc1.weight",
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2.weight",
    ]

    missing = [name for name in mtp_parameter_names if registry.megatron_to_hf_lookup(name) is None]

    assert missing == []


class TestBailingMoeV2BuilderConfig:
    """Builder-backed ModelConfig path for Ling MoE V2."""

    @staticmethod
    def _config(**overrides):
        from megatron.bridge.models.bailing.configuration_bailing_moe_v2 import BailingMoeV2Config

        kwargs = dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=8,
            num_experts=8,
            num_shared_experts=1,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            moe_intermediate_size=32,
            first_k_dense_replace=1,
            max_position_embeddings=4096,
            torch_dtype="bfloat16",
        )
        kwargs.update(overrides)
        config = BailingMoeV2Config(**kwargs)
        config.architectures = ["BailingMoeV2ForCausalLM"]
        return config

    def test_provider_bridge_is_inherited_compatibility_only(self):
        assert "provider_bridge" not in BailingMoeV2Bridge.__dict__

    def test_conversion_uses_builder_config(self):
        assert BailingMoeV2Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_hf_config_to_model_config_uses_direct_mapping(self):
        from unittest.mock import patch

        from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig

        config = self._config()
        bridge = BailingMoeV2Bridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            result = bridge.hf_config_to_model_config(config)

        assert isinstance(result, BridgeGPTModelConfig)
        # The builder's default MoE spec is the decoder block spec, so none is configured here.
        assert result.transformer_layer_spec is None
        assert result.moe_layer_freq == [0, 1, 1, 1]
        assert result.moe_shared_expert_intermediate_size == config.moe_intermediate_size
        assert result.qk_layernorm is True
        assert result.add_qkv_bias is False
        assert result.moe_router_score_function == "sigmoid"
        assert result.moe_router_enable_expert_bias is True
        assert result.position_embedding_type == "rope"

    @pytest.mark.parametrize("use_qkv_bias", [False, True])
    def test_model_config_matches_provider_runtime_config(self, use_qkv_bias):
        from dataclasses import fields
        from unittest.mock import Mock

        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

        from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

        config = self._config(use_qkv_bias=use_qkv_bias)
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        bridge = BailingMoeV2Bridge()

        model_config = bridge.hf_config_to_model_config(config)
        with pytest.warns(FutureWarning, match=r"deprecated.*get_model_config.*get_model"):
            provider = bridge.provider_bridge(mock_pretrained)

        provider_fields = {field.name for field in fields(provider)}
        model_config_fields = {field.name for field in fields(model_config)}
        model_config_fields.update(field.name for field in fields(model_config.transformer))
        for field_name in sorted((provider_fields & model_config_fields) - {"transformer_layer_spec"}):
            assert getattr(model_config, field_name) == getattr(provider, field_name), field_name
        assert model_config.add_qkv_bias is use_qkv_bias
        # The provider keeps its explicit decoder block spec for the dense-first layer pattern.
        assert provider.transformer_layer_spec.func is get_gpt_decoder_block_spec
