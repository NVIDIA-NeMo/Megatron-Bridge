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
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_bridge import ExaoneMoeBridge
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_config import (
    ExaoneMoeModelConfig,
    ExaoneMoeTransformerConfig,
)
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_provider import (
    ExaoneMoeDecoderLayer,
    ExaoneMoeModelProvider,
    _layer_specific_config,
    _MTPDenseLayerSpecsList,
    build_exaone_moe_layer_spec,
)
from megatron.bridge.models.gpt.gpt_builder import mtp_block_spec
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit


def _make_config(**overrides) -> SimpleNamespace:
    values = {
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "intermediate_size": 768,
        "moe_intermediate_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "n_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "num_shared_experts": 1,
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "vocab_size": 153600,
        "max_position_embeddings": 4096,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "sliding_window": 1024,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_pretrained(config: SimpleNamespace) -> Mock:
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    return pretrained


def _make_k_exaone_2_0_config(**overrides) -> SimpleNamespace:
    """K-EXAONE 2.0 layout: dense-first MoE, per-layer windows and clamps, one repeated MTP layer."""
    values = {
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "sliding_windows": [0, 4096, 128, 0],
        "mlp_layer_types": ["dense", "dense", "sparse", "sparse"],
        "swiglu_limits": [0.0, 0.0, 7.0, 7.0],
        "num_nextn_predict_layers": 1,
        "mtp_num_speculative_steps": 4,
        "mtp_layer_types": ["sliding_attention"],
        "mtp_sliding_windows": [128],
        "mtp_loss_scaling_factor": 0.05,
        "topk_method": "noaux_tc",
    }
    values.update(overrides)
    return _make_config(**values)


class TestExaoneMoeBridge:
    @pytest.mark.parametrize("architecture", ["ExaoneMoeForCausalLM", "ExaoneMoEForCausalLM"])
    def test_autobridge_registration(self, architecture):
        config = _make_config(
            architectures=[architecture],
            model_type="exaone_moe",
        )

        assert AutoBridge.supports(config)
        AutoBridge._validate_config(config, "LGAI-EXAONE/K-EXAONE-236B-A23B")
        assert isinstance(get_model_bridge(architecture, hf_config=config), ExaoneMoeBridge)

    def test_top_level_model_registration(self):
        from megatron.bridge.models import ExaoneMoeBridge as RegisteredExaoneMoeBridge

        assert RegisteredExaoneMoeBridge is ExaoneMoeBridge

    def test_provider_bridge_maps_hf_config(self):
        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config()))

        assert isinstance(provider, ExaoneMoeModelProvider)
        assert provider.num_layers == 4
        assert provider.hidden_size == 256
        assert provider.ffn_hidden_size == 768
        assert provider.moe_ffn_hidden_size == 128
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 2
        assert provider.kv_channels == 32
        assert provider.num_moe_experts == 8
        assert provider.moe_router_topk == 2
        assert provider.moe_router_num_groups == 4
        assert provider.moe_router_group_topk == 2
        assert provider.moe_router_topk_scaling_factor == 2.5
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_shared_expert_intermediate_size == 128
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.share_embeddings_and_output_weights is False
        assert provider.moe_layer_freq == [0, 1, 1, 1]
        assert provider.no_rope_freq == [0, 1, 0, 1]
        assert provider.window_attn_skip_freq == [1, 0, 1, 0]
        assert provider.window_size == (1023, 0)

    def test_provider_bridge_uses_only_mlp_layer_types_for_moe_schedule(self):
        provider = ExaoneMoeBridge().provider_bridge(
            _make_pretrained(
                _make_config(
                    mlp_layer_types=["dense", "sparse", "sparse", "dense"],
                    is_moe_layer=[True, False, True, False],
                    first_k_dense_replace=2,
                )
            )
        )

        assert provider.moe_layer_freq == [0, 1, 1, 0]

    @pytest.mark.parametrize(
        ("config_overrides", "expected"),
        [
            (
                {
                    "mlp_layer_types": None,
                    "is_moe_layer": [False, True, False, True],
                    "first_k_dense_replace": 2,
                },
                [0, 1, 0, 1],
            ),
            (
                {
                    "mlp_layer_types": None,
                    "first_k_dense_replace": 2,
                },
                [0, 0, 1, 1],
            ),
        ],
    )
    def test_provider_bridge_supports_legacy_moe_layer_schedules(self, config_overrides, expected):
        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config(**config_overrides)))

        assert provider.moe_layer_freq == expected

    def test_provider_bridge_uses_sliding_window_fallback(self):
        provider = ExaoneMoeBridge().provider_bridge(
            _make_pretrained(_make_config(layer_types=["full_attention"] * 4, sliding_window=1024))
        )

        assert provider.sliding_windows == [0, 0, 0, 0]
        assert provider.no_rope_freq == [1, 1, 1, 1]
        assert provider.window_attn_skip_freq == [0, 0, 0, 0]
        assert provider.window_size is None

    def test_provider_bridge_infers_layer_types_from_sliding_windows(self):
        provider = ExaoneMoeBridge().provider_bridge(
            _make_pretrained(_make_config(layer_types=None, sliding_windows=[0, 4096, 128, 0]))
        )

        assert provider.layer_types == ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
        assert provider.no_rope_freq == [1, 0, 0, 1]

    def test_provider_bridge_infers_head_dim_and_score_function(self):
        config = _make_config()
        del config.head_dim
        del config.scoring_func

        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(config))

        assert provider.kv_channels == 32
        assert provider.moe_router_score_function == "sigmoid"

    def test_provider_bridge_maps_rope_scaling_and_mtp(self):
        config = _make_config(
            rope_parameters={"rope_type": "linear", "rope_theta": 500000.0, "factor": 4.0},
            num_nextn_predict_layers=1,
            mtp_loss_scaling_factor=0.25,
            mtp_share_layers=True,
        )

        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(config))

        assert provider.rotary_base == 500000.0
        assert provider.rope_scaling is True
        assert provider.rope_scaling_factor == 4.0
        assert provider.mtp_num_layers == 1
        assert provider.mtp_loss_scaling_factor == 0.25
        assert provider.mtp_use_repeated_layer is True
        assert provider.mtp_layer_types == ["full_attention"]
        assert provider.mtp_sliding_windows == [0]

    def test_provider_bridge_maps_k_exaone_2nd_per_layer_config(self):
        config = _make_config(
            layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
            sliding_windows=[0, 4096, 128, 0],
            mlp_layer_types=["dense", "dense", "sparse", "sparse"],
            swiglu_limits=[0.0, 0.0, 7.0, 7.0],
            num_nextn_predict_layers=3,
            mtp_num_speculative_steps=4,
            mtp_layer_types=["sliding_attention"],
            mtp_sliding_windows=[128],
            topk_method="noaux_tc",
        )
        del config.torch_dtype
        config.dtype = "bfloat16"

        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(config))

        assert provider.moe_layer_freq == [0, 0, 1, 1]
        assert provider.no_rope_freq == [1, 0, 0, 1]
        assert provider.window_attn_skip_freq == [0, 1, 1, 0]
        assert provider.window_size == (4095, 0)
        assert provider.sliding_windows == [0, 4096, 128, 0]
        assert provider.swiglu_limits == [0.0, 0.0, 7.0, 7.0]
        assert provider.mtp_num_layers == 4
        assert provider.mtp_use_repeated_layer is True
        assert provider.mtp_layer_types == ["sliding_attention"]
        assert provider.mtp_sliding_windows == [128]

        exported_config = ExaoneMoeBridge.megatron_to_hf_config(provider)
        assert exported_config["num_nextn_predict_layers"] == 1
        assert exported_config["mtp_num_speculative_steps"] == 4
        assert provider.moe_router_load_balancing_type == "none"
        assert provider.moe_aux_loss_coeff == 0.0
        assert provider.params_dtype is torch.bfloat16

    def test_layer_specific_config_applies_main_and_mtp_settings(self):
        provider = ExaoneMoeBridge().provider_bridge(
            _make_pretrained(
                _make_config(
                    layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
                    sliding_windows=[0, 4096, 128, 0],
                    swiglu_limits=[0.0, 0.0, 7.0, 7.0],
                    num_nextn_predict_layers=1,
                    mtp_num_speculative_steps=4,
                    mtp_layer_types=["sliding_attention"],
                    mtp_sliding_windows=[128],
                )
            )
        )

        full_layer = _layer_specific_config(provider, layer_idx=0, is_mtp_layer=False)
        clamped_sliding_layer = _layer_specific_config(provider, layer_idx=2, is_mtp_layer=False)
        mtp_layer = _layer_specific_config(provider, layer_idx=0, is_mtp_layer=True)

        assert full_layer.window_size is None
        assert full_layer.activation_func_clamp_value is None
        assert clamped_sliding_layer.window_size == (127, 0)
        assert clamped_sliding_layer.activation_func_clamp_value == 7.0
        assert mtp_layer.window_size == (127, 0)
        assert mtp_layer.no_rope_freq == [0, 0, 0, 0]

    def test_provider_bridge_uses_dense_mtp_layer_spec(self):
        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config(num_nextn_predict_layers=1)))

        block_spec = provider.transformer_layer_spec(provider, pp_rank=0)

        assert isinstance(block_spec.layer_specs, _MTPDenseLayerSpecsList)
        assert len(block_spec.layer_specs) == provider.num_layers
        assert all(layer_spec.module is ExaoneMoeDecoderLayer for layer_spec in block_spec.layer_specs)
        assert block_spec.layer_specs[-1].module is ExaoneMoeDecoderLayer
        assert block_spec.layer_specs[-1].submodules.mlp.func == MLP.as_mlp_submodule
        assert block_spec.layer_specs[-1] is not block_spec.layer_specs[provider.num_layers - 1]

    def test_layer_spec_builder_is_public_for_checkpoint_config_reload(self):
        provider = ExaoneMoeModelProvider()

        assert provider.transformer_layer_spec is build_exaone_moe_layer_spec
        assert not provider.transformer_layer_spec.__name__.startswith("_")

    @pytest.mark.parametrize(
        ("config_overrides", "message"),
        [
            ({"mlp_layer_types": None}, "mlp_layer_types, is_moe_layer, or first_k_dense_replace"),
            ({"mlp_layer_types": ["dense"]}, "MoE layer schedule"),
            ({"sliding_window": None}, "sliding_windows or sliding_window"),
            ({"sliding_windows": [128]}, "sliding_windows"),
            ({"swiglu_limits": [7.0]}, "swiglu_limits"),
            ({"mtp_num_speculative_steps": 4, "num_nextn_predict_layers": 2}, "repeated MTP"),
            ({"mtp_layer_types": ["sliding_attention"]}, "mtp_sliding_windows"),
            ({"mtp_layer_types": ["sliding_attention"], "mtp_sliding_windows": []}, "mtp_sliding_windows"),
        ],
    )
    def test_provider_bridge_rejects_invalid_per_layer_config(self, config_overrides, message):
        with pytest.raises(ValueError, match=message):
            ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config(**config_overrides)))

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_output_mapping_follows_weight_tying(self, tie_word_embeddings):
        bridge = ExaoneMoeBridge()
        bridge.hf_config = _make_config(tie_word_embeddings=tie_word_embeddings)

        mapping = bridge.mapping_registry().megatron_to_hf_lookup("output_layer.weight")

        expected = "model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight"
        assert mapping.hf_param == expected

    def test_mapping_registry_contains_core_and_moe_mappings(self):
        bridge = ExaoneMoeBridge()
        bridge.hf_config = _make_config()
        registry = bridge.mapping_registry()

        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_proj.weight").hf_param
            == "model.layers.0.self_attn.o_proj.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.self_attn.q_proj.weight").megatron_param
            == "decoder.layers.0.self_attention.linear_qkv.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.experts.3.gate_proj.weight").megatron_param
            == "decoder.layers.0.mlp.experts.linear_fc1.weight3"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.e_score_correction_bias").megatron_param
            == "decoder.layers.0.mlp.router.expert_bias"
        )
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc1.layer_norm_weight").hf_param
            == "model.layers.0.post_attention_layernorm.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.shared_experts.up_proj.weight").megatron_param
            == "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.down_proj.weight").megatron_param
            == "decoder.layers.0.mlp.linear_fc2.weight"
        )

    def test_mapping_registry_prioritizes_mtp_layer_types(self):
        bridge = ExaoneMoeBridge()
        bridge.hf_config = _make_config(num_nextn_predict_layers=1, mtp_layer_types=[])

        assert bridge.mapping_registry().hf_to_megatron_lookup("mtp.layers.0.self_attn.q_proj.weight") is None

        bridge.hf_config = _make_config(num_nextn_predict_layers=0, mtp_layer_types=["sliding_attention"])
        registry = bridge.mapping_registry()

        mapping = registry.hf_to_megatron_lookup("mtp.layers.0.self_attn.q_proj.weight")
        assert mapping.megatron_param == "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"

        dense_fc1_mapping = registry.hf_to_megatron_lookup("mtp.layers.0.mlp.gate_proj.weight")
        assert dense_fc1_mapping.megatron_param == "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight"

        dense_fc1_norm_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight"
        )
        assert dense_fc1_norm_mapping.hf_param == "mtp.layers.0.post_attention_layernorm.weight"

        dense_fc2_mapping = registry.hf_to_megatron_lookup("mtp.layers.0.mlp.down_proj.weight")
        assert dense_fc2_mapping.megatron_param == "mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight"

        assert registry.hf_to_megatron_lookup("mtp.layers.0.mlp.experts.3.gate_proj.weight") is None
        assert registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.mlp.router.weight") is None


class TestExaoneMoeBridgeModelConfig:
    def test_conversion_uses_builder_config(self):
        assert "provider_bridge" not in ExaoneMoeBridge.__dict__
        assert ExaoneMoeBridge.MODEL_CONFIG_CLASS is ExaoneMoeModelConfig
        assert ExaoneMoeModelConfig.transformer_config_class is ExaoneMoeTransformerConfig
        assert ExaoneMoeBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_model_config_maps_hf_config_without_provider(self):
        config = _make_k_exaone_2_0_config()
        del config.torch_dtype
        config.dtype = "bfloat16"
        bridge = ExaoneMoeBridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(config)

        assert isinstance(model_config, ExaoneMoeModelConfig)
        assert isinstance(model_config.transformer, ExaoneMoeTransformerConfig)
        assert model_config.transformer_layer_spec is build_exaone_moe_layer_spec
        assert model_config.position_embedding_type == "rope"
        assert model_config.params_dtype is torch.bfloat16
        assert model_config.moe_layer_freq == [0, 0, 1, 1]
        assert model_config.window_size == (4095, 0)
        assert model_config.transformer.layer_types == config.layer_types
        assert model_config.transformer.sliding_windows == [0, 4096, 128, 0]
        assert model_config.transformer.swiglu_limits == [0.0, 0.0, 7.0, 7.0]
        assert model_config.transformer.mtp_layer_types == ["sliding_attention"]
        assert model_config.transformer.mtp_sliding_windows == [128]
        assert model_config.mtp_num_layers == 4
        assert model_config.mtp_use_repeated_layer is True
        assert model_config.mtp_loss_scaling_factor == 0.05
        assert model_config.moe_router_load_balancing_type == "none"
        assert model_config.moe_aux_loss_coeff == 0.0
        # EXAONE provider settings that differ from the builder config defaults.
        assert model_config.attention_softmax_in_fp32 is True
        assert model_config.gradient_accumulation_fusion is False
        assert model_config.moe_token_dispatcher_type == "alltoall"
        assert model_config.moe_router_pre_softmax is True

    @pytest.mark.parametrize(
        "config_factory",
        [
            _make_config,
            _make_k_exaone_2_0_config,
            lambda: _make_config(mlp_layer_types=None, first_k_dense_replace=2, tie_word_embeddings=True),
            lambda: _make_config(
                rope_parameters={"rope_type": "linear", "rope_theta": 500000.0, "factor": 4.0},
                num_nextn_predict_layers=1,
                mtp_share_layers=True,
            ),
        ],
        ids=["k_exaone", "k_exaone_2_0", "legacy_schedule_tied", "linear_rope_shared_mtp"],
    )
    def test_model_config_matches_provider(self, config_factory):
        config = config_factory()
        bridge = ExaoneMoeBridge()
        model_config = bridge.hf_config_to_model_config(config)
        with pytest.warns(FutureWarning):
            provider = bridge.provider_bridge(_make_pretrained(config))

        model_fields = {f.name for f in fields(model_config)} | {f.name for f in fields(model_config.transformer)}
        shared = {f.name for f in fields(provider)} & model_fields
        assert {"transformer_layer_spec", "sliding_windows", "swiglu_limits", "mtp_sliding_windows"} <= shared
        mismatches = {
            name: (getattr(model_config, name), getattr(provider, name))
            for name in sorted(shared)
            if getattr(model_config, name) != getattr(provider, name)
        }
        assert mismatches == {}

    def test_layer_spec_accepts_model_config_without_mutating_it(self):
        model_config = ExaoneMoeBridge().hf_config_to_model_config(_make_k_exaone_2_0_config())
        before = (model_config.num_moe_experts, list(model_config.moe_layer_freq), model_config.moe_grouped_gemm)

        block_spec = build_exaone_moe_layer_spec(model_config, pp_rank=0)

        after = (model_config.num_moe_experts, list(model_config.moe_layer_freq), model_config.moe_grouped_gemm)
        assert after == before
        assert all(layer_spec.module is ExaoneMoeDecoderLayer for layer_spec in block_spec.layer_specs)
        assert isinstance(block_spec.layer_specs[-1], TransformerBlockSubmodules)

        # The GPT builder derives the MTP block from layer_specs[-1].
        mtp_spec = mtp_block_spec(model_config, block_spec, vp_stage=None, pp_rank=0)
        mtp_layer_spec = mtp_spec.layer_specs[0].submodules.mtp_model_layer
        assert mtp_layer_spec.module is ExaoneMoeDecoderLayer
        assert mtp_layer_spec.submodules.mlp.func == MLP.as_mlp_submodule

    def test_model_config_round_trips_through_serialization(self):
        model_config = ExaoneMoeBridge().hf_config_to_model_config(_make_k_exaone_2_0_config())

        restored = ExaoneMoeModelConfig.from_dict(model_config.as_dict())

        assert isinstance(restored, ExaoneMoeModelConfig)
        assert isinstance(restored.transformer, ExaoneMoeTransformerConfig)
        assert restored.transformer_layer_spec is build_exaone_moe_layer_spec
        for name in ("layer_types", "sliding_windows", "swiglu_limits", "mtp_layer_types", "mtp_sliding_windows"):
            assert getattr(restored.transformer, name) == getattr(model_config.transformer, name), name

    def test_exported_config_keeps_physical_and_repeated_mtp_layers(self):
        model_config = ExaoneMoeBridge().hf_config_to_model_config(_make_k_exaone_2_0_config())

        exported_config = ExaoneMoeBridge.megatron_to_hf_config(model_config)

        assert exported_config["num_nextn_predict_layers"] == 1
        assert exported_config["mtp_num_speculative_steps"] == 4

    def test_dtype_defaults_to_float32_without_dtype_fields(self):
        config = _make_config()
        del config.torch_dtype

        model_config = ExaoneMoeBridge().hf_config_to_model_config(config)

        assert model_config.params_dtype is torch.float32
        assert model_config.bf16 is False
