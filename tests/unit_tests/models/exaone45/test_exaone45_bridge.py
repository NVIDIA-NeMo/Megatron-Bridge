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

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from megatron.core.transformer import TransformerLayer
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_VisionConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.common.te_layers import TERowParallelLinearLayerNorm
from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.exaone.exaone45.exaone45_bridge import Exaone45Bridge
from megatron.bridge.models.exaone.exaone45.exaone45_config import Exaone45ModelConfig
from megatron.bridge.models.exaone.exaone45.exaone45_provider import (
    Exaone45ModelProvider,
    exaone_45_mtp_block_spec,
    exaone_45_transformer_layer_spec,
)
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.transformer_config import get_vision_model_config
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit


def _make_text_config(*, tie_word_embeddings: bool, mtp_num_layers: int | None = 0) -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=768,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        vocab_size=153600,
        max_position_embeddings=4096,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype="bfloat16",
        attention_bias=False,
        rope_parameters={
            "rope_type": "mrope",
            "rope_theta": 1000000.0,
            "factor": 1.0,
            "mrope_section": [4, 2, 2],
        },
        sliding_window_pattern=2,
        sliding_window=1024,
        layer_types=["sliding_attention", "full_attention"] * 2,
        num_nextn_predict_layers=mtp_num_layers,
        mtp_loss_scaling_factor=0.2,
        mtp_share_layers=False,
    )


def _make_pretrained(
    *,
    top_level_tie_word_embeddings: bool,
    text_config_tie_word_embeddings: bool,
    mtp_num_layers: int | None = 0,
) -> Mock:
    vision_config = Exaone4_5_VisionConfig()
    config = SimpleNamespace(
        text_config=_make_text_config(
            tie_word_embeddings=text_config_tie_word_embeddings,
            mtp_num_layers=mtp_num_layers,
        ),
        vision_config=vision_config,
        tie_word_embeddings=top_level_tie_word_embeddings,
        bos_token_id=1,
        eos_token_id=53,
        vision_start_token_id=73,
        vision_end_token_id=74,
        vision_token_id=67,
        image_token_id=67,
        video_token_id=68,
    )
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    return pretrained


def _mapping_names(bridge: Exaone45Bridge) -> set[str]:
    return {mapping.megatron_param for mapping in bridge.mapping_registry().mappings}


class TestExaone45Bridge:
    def test_vision_config_uses_hf_attention_dimensions(self):
        vision_config = Exaone4_5_VisionConfig(
            hidden_size=64,
            num_heads=8,
            num_key_value_heads=4,
        )
        provider = Exaone45ModelProvider(
            num_layers=2,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            num_query_groups=2,
        )

        config = get_vision_model_config(vision_config, provider)

        assert config.kv_channels == 8
        assert isinstance(config.kv_channels, int)
        assert config.num_query_groups == 4

    def test_autobridge_registration(self):
        config = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=False,
        ).config
        config.architectures = ["Exaone4_5_ForConditionalGeneration"]
        config.model_type = "exaone4_5"

        assert AutoBridge.supports(config)
        assert isinstance(get_model_bridge("Exaone4_5_ForConditionalGeneration", hf_config=config), Exaone45Bridge)

    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_mtp_block_spec_wraps_custom_layer_in_block_spec(self, get_gpt_mtp_block_spec):
        provider = Exaone45ModelProvider(mtp_num_layers=1)
        expected = Mock()
        get_gpt_mtp_block_spec.return_value = expected

        result = exaone_45_mtp_block_spec(provider, vp_stage=2)

        assert result is expected
        block_spec = get_gpt_mtp_block_spec.call_args.args[1]
        assert isinstance(block_spec, TransformerBlockSubmodules)
        layer_spec = block_spec.layer_specs[0]
        assert layer_spec.module is TransformerLayer
        assert layer_spec.submodules.self_attention.submodules.linear_proj is TERowParallelLinearLayerNorm
        assert layer_spec.submodules.mlp.submodules.linear_fc2 is TERowParallelLinearLayerNorm
        get_gpt_mtp_block_spec.assert_called_once_with(
            provider,
            block_spec,
            use_transformer_engine=True,
            vp_stage=2,
        )

    def test_provider_bridge_maps_vlm_config(self):
        pretrained = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=True,
        )

        provider = Exaone45Bridge().provider_bridge(pretrained)

        assert isinstance(provider, Exaone45ModelProvider)
        assert provider.num_layers == 4
        assert provider.hidden_size == 256
        assert provider.ffn_hidden_size == 768
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 2
        assert provider.kv_channels == 32
        assert provider.params_dtype == torch.bfloat16
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == 1000000.0
        assert provider.rope_scaling is True
        assert provider.rope_scaling_factor == 1.0
        assert provider.no_rope_freq == [0, 1, 0, 1]
        assert provider.window_attn_skip_freq == [1, 0, 1, 0]
        assert provider.window_size == (1023, 0)

    def test_provider_bridge_supports_default_rope_without_scaling_factor(self):
        pretrained = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=True,
        )
        pretrained.config.text_config.rope_parameters.pop("factor")

        provider = Exaone45Bridge().provider_bridge(pretrained)

        assert provider.rotary_base == 1000000.0
        assert provider.rope_scaling is False
        assert provider.rope_scaling_factor is None

    def test_provider_bridge_supports_full_attention_without_sliding_window(self):
        pretrained = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=True,
        )
        pretrained.config.text_config.layer_types = ["full_attention"] * 4
        pretrained.config.text_config.sliding_window = None

        provider = Exaone45Bridge().provider_bridge(pretrained)

        assert provider.no_rope_freq == [1, 1, 1, 1]
        assert provider.window_attn_skip_freq == [0, 0, 0, 0]
        assert provider.window_size is None

    @pytest.mark.parametrize(("top_level", "text_level"), [(False, True), (True, False)])
    def test_provider_uses_top_level_weight_tying(self, top_level, text_level):
        pretrained = _make_pretrained(
            top_level_tie_word_embeddings=top_level,
            text_config_tie_word_embeddings=text_level,
        )

        provider = Exaone45Bridge().provider_bridge(pretrained)

        assert provider.share_embeddings_and_output_weights is top_level

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_output_mapping_follows_top_level_weight_tying(self, tie_word_embeddings):
        bridge = Exaone45Bridge()
        bridge.hf_config = _make_pretrained(
            top_level_tie_word_embeddings=tie_word_embeddings,
            text_config_tie_word_embeddings=not tie_word_embeddings,
        ).config

        mapping = bridge.mapping_registry().megatron_to_hf_lookup("language_model.output_layer.weight")

        expected = "model.language_model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight"
        assert mapping.hf_param == expected

    def test_optional_mtp_mappings(self):
        bridge = Exaone45Bridge()
        bridge.hf_config = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=False,
        ).config
        names = _mapping_names(bridge)
        assert not any(".mtp." in name for name in names)

        bridge.hf_config = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=False,
            mtp_num_layers=1,
        ).config
        names = _mapping_names(bridge)
        assert "language_model.mtp.layers.0.enorm.weight" in names
        assert "language_model.mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight" in names
        assert "language_model.mtp.layers.*.mtp_model_layer.self_attention.linear_proj.post_layernorm.weight" in names

    def test_none_mtp_layer_count_disables_mtp_mappings(self):
        bridge = Exaone45Bridge()
        bridge.hf_config = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=False,
            mtp_num_layers=None,
        ).config

        names = _mapping_names(bridge)

        assert not any(".mtp." in name for name in names)

    def test_mapping_registry_contains_vision_and_language_mappings(self):
        bridge = Exaone45Bridge()
        bridge.hf_config = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=False,
        ).config
        registry = bridge.mapping_registry()

        assert (
            registry.hf_to_megatron_lookup("model.language_model.layers.0.self_attn.q_proj.weight").megatron_param
            == "language_model.decoder.layers.0.self_attention.linear_qkv.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.visual.blocks.0.attn.qkv.weight").megatron_param
            == "vision_model.decoder.layers.0.self_attention.linear_qkv.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.visual.patch_embed.proj.weight").megatron_param
            == "vision_model.patch_embed.proj.weight"
        )
        assert (
            registry.hf_to_megatron_lookup(
                "model.language_model.layers.0.post_attention_layernorm.weight"
            ).megatron_param
            == "language_model.decoder.layers.0.self_attention.linear_proj.post_layernorm.weight"
        )
        assert (
            registry.hf_to_megatron_lookup(
                "model.language_model.layers.0.post_feedforward_layernorm.weight"
            ).megatron_param
            == "language_model.decoder.layers.0.mlp.linear_fc2.post_layernorm.weight"
        )


class TestExaone45BridgeModelConfig:
    def test_conversion_uses_builder_config(self):
        assert "provider_bridge" not in Exaone45Bridge.__dict__
        assert Exaone45Bridge.MODEL_CONFIG_CLASS is Exaone45ModelConfig
        assert Exaone45ModelConfig.transformer_config_class is TransformerConfig
        assert Exaone45Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_model_config_maps_vlm_config_without_provider(self):
        config = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=True,
            mtp_num_layers=1,
        ).config
        bridge = Exaone45Bridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(config)

        assert isinstance(model_config, Exaone45ModelConfig)
        assert model_config.transformer_layer_spec is exaone_45_transformer_layer_spec
        assert model_config.num_layers == 4
        assert model_config.kv_channels == 32
        assert model_config.activation_func is F.silu
        assert model_config.normalization == "RMSNorm"
        assert model_config.qk_layernorm is True
        assert model_config.params_dtype is torch.bfloat16
        # VLM weight tying is defined on the top-level config.
        assert model_config.share_embeddings_and_output_weights is False
        assert model_config.position_embedding_type == "rope"
        assert model_config.rotary_base == 1000000.0
        assert model_config.rope_scaling is True
        assert model_config.rope_scaling_factor == 1.0
        assert model_config.no_rope_freq == [0, 1, 0, 1]
        assert model_config.window_attn_skip_freq == [1, 0, 1, 0]
        assert model_config.window_size == (1023, 0)
        assert model_config.moe_layer_freq == [0, 0, 0, 0]
        assert model_config.mtp_num_layers == 1
        assert model_config.mtp_loss_scaling_factor == 0.2
        assert model_config.mtp_use_repeated_layer is False
        assert model_config.vision_config is config.vision_config
        assert model_config.vision_config.torch_dtype is torch.bfloat16
        assert model_config.language_max_sequence_length == 4096
        assert (model_config.image_token_id, model_config.video_token_id) == (67, 68)
        assert (model_config.vision_start_token_id, model_config.vision_end_token_id) == (73, 74)
        # EXAONE 4.5 provider settings that differ from the EXAONE 4.0 text mapping.
        assert model_config.autocast_dtype is None
        assert model_config.attention_softmax_in_fp32 is True
        assert model_config.gradient_accumulation_fusion is False
        assert model_config.bias_activation_fusion is False
        assert model_config.bias_dropout_fusion is False
        assert model_config.apply_rope_fusion is False

    @pytest.mark.parametrize(
        "text_overrides",
        [
            {},
            {"num_nextn_predict_layers": 1, "mtp_share_layers": True},
            {"rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}},
            {"layer_types": ["full_attention"] * 4, "sliding_window": None},
            {"attention_bias": True, "torch_dtype": "float16"},
        ],
        ids=["default", "shared_mtp", "no_rope_factor", "full_attention", "qkv_bias_fp16"],
    )
    def test_model_config_matches_provider(self, text_overrides):
        pretrained = _make_pretrained(
            top_level_tie_word_embeddings=True,
            text_config_tie_word_embeddings=False,
        )
        for name, value in text_overrides.items():
            setattr(pretrained.config.text_config, name, value)
        bridge = Exaone45Bridge()
        model_config = bridge.hf_config_to_model_config(pretrained.config)
        with pytest.warns(FutureWarning):
            provider = bridge.provider_bridge(pretrained)

        assert isinstance(provider, Exaone45ModelProvider)
        assert provider.hf_text_config is pretrained.config.text_config
        assert model_config.language_max_sequence_length == provider.hf_text_config.max_position_embeddings
        model_fields = {f.name for f in fields(model_config)} | {f.name for f in fields(model_config.transformer)}
        shared = {f.name for f in fields(provider)} & model_fields
        assert {"transformer_layer_spec", "vision_config", "window_size", "freeze_mtp_model"} <= shared
        mismatches = {
            name: (getattr(model_config, name), getattr(provider, name))
            for name in sorted(shared)
            if getattr(model_config, name) != getattr(provider, name)
        }
        assert mismatches == {}

    def test_model_config_round_trips_through_serialization(self):
        config = _make_pretrained(
            top_level_tie_word_embeddings=False,
            text_config_tie_word_embeddings=False,
            mtp_num_layers=1,
        ).config
        model_config = Exaone45Bridge().hf_config_to_model_config(config)
        model_config.freeze_vision_model = True

        restored = Exaone45ModelConfig.from_dict(model_config.as_dict())

        assert isinstance(restored, Exaone45ModelConfig)
        assert restored.transformer_layer_spec is exaone_45_transformer_layer_spec
        assert isinstance(restored.vision_config, Exaone4_5_VisionConfig)
        assert restored.vision_config.to_dict() == model_config.vision_config.to_dict()
        for name in (
            "language_max_sequence_length",
            "image_token_id",
            "freeze_vision_model",
            "no_rope_freq",
            "window_attn_skip_freq",
            "rope_scaling_factor",
            "mtp_num_layers",
        ):
            assert getattr(restored, name) == getattr(model_config, name), name
        assert tuple(restored.window_size) == model_config.window_size
