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

"""Unit tests for Step37Bridge (Step3.7): builder config mapping, builder, and layer-spec wiring."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.bridge.models.conversion.utils import conform_config_to_reference
from megatron.bridge.models.stepfun import step35_bridge as _step35_bridge_mod
from megatron.bridge.models.stepfun import step37_builder as _step37_builder_mod
from megatron.bridge.models.stepfun.configuration_step37 import Step37Config, Step37VisionConfig
from megatron.bridge.models.stepfun.modelling_step37 import model as _step37_model_mod
from megatron.bridge.models.stepfun.modelling_step37 import transformer_block as _step37_block_mod
from megatron.bridge.models.stepfun.modelling_step37.image_insert_embedding import ImageInsertEmbedding
from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.modelling_step37.transformer_block import get_step37_text_layer_spec
from megatron.bridge.models.stepfun.modelling_step37.vision_model import Step37VisionModel
from megatron.bridge.models.stepfun.step35_bridge import Step35Bridge, build_step35_layer_spec
from megatron.bridge.models.stepfun.step35_config import Step35TransformerConfig
from megatron.bridge.models.stepfun.step37_bridge import Step37Bridge
from megatron.bridge.models.stepfun.step37_builder import Step37ModelBuilder
from megatron.bridge.models.stepfun.step37_config import Step37ModelConfig


_LAYER_TYPES = ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
_MTP_LAYER_TYPES = ["full_attention", "sliding_attention"]


def _make_hf_config(**text_overrides) -> Step37Config:
    """A small Step3.7 config: 4 text layers (2 dense, 2 MoE), 2 MTP layers and a 2-layer vision tower."""
    text = dict(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_attention_groups=2,
        num_hidden_layers=4,
        max_seq_len=4096,
        max_position_embeddings=4096,
        vocab_size=1024,
        moe_intermediate_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        share_expert_dim=64,
        head_dim=64,
        moe_layers_enum="2,3",
        rope_theta=[5e6, 1e4, 1e4, 1e4, 5e6, 1e4],
        partial_rotary_factors=[0.5, 1.0, 1.0, 1.0, 0.5, 1.0],
        layer_types=_LAYER_TYPES + _MTP_LAYER_TYPES,
        num_nextn_predict_layers=2,
        sliding_window=64,
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": 8,
            "num_attention_groups": 2,
            "head_dim": 64,
        },
        swiglu_limits=[0.0, 0.0, 7.0, 7.0, 0.0, 0.0],
        swiglu_limits_shared=[0.0, 0.0, 16.0, 16.0, 0.0, 0.0],
        use_moe_router_bias=True,
        need_fp32_gate=True,
        moe_router_activation="sigmoid",
        moe_router_scaling_factor=3.0,
        torch_dtype="bfloat16",
    )
    text.update(text_overrides)
    return Step37Config(
        text_config=text,
        vision_config={"width": 64, "layers": 2, "heads": 2, "image_size": 56, "patch_size": 14},
        image_token_id=128001,
        projector_bias=False,
    )


class TestStep37BridgeModelConfig:
    def test_conversion_uses_builder_config(self):
        assert "provider_bridge" not in Step37Bridge.__dict__
        assert Step37Bridge.MODEL_CONFIG_CLASS is Step37ModelConfig
        assert Step37Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True
        assert Step37ModelConfig.transformer_config_class is Step35TransformerConfig
        assert Step37ModelConfig(transformer=Step35TransformerConfig()).get_builder_cls() is Step37ModelBuilder

    @pytest.mark.parametrize("native_head_wise_gate", [False, True])
    def test_model_config_maps_hf_config_without_provider(self, monkeypatch, native_head_wise_gate):
        monkeypatch.setattr(_step35_bridge_mod, "_mcore_supports_head_wise_attn_gate", lambda: native_head_wise_gate)
        hf_config = _make_hf_config()
        bridge = Step37Bridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(hf_config)

        assert isinstance(model_config, Step37ModelConfig)
        assert isinstance(model_config.transformer, Step35TransformerConfig)
        assert model_config.transformer_layer_spec is build_step35_layer_spec
        assert model_config.moe_layer_freq == [0, 0, 1, 1]
        assert model_config.mtp_num_layers == 2
        assert model_config.position_embedding_type == "rope"
        assert model_config.rotary_base == 5e6
        assert model_config.transformer.rotary_base_per_layer == [5e6, 1e4, 1e4, 1e4, 5e6, 1e4]
        assert model_config.transformer.rotary_percents == [0.5, 1.0, 1.0, 1.0, 0.5, 1.0]
        assert model_config.transformer.swiglu_limits == [0.0, 0.0, 7.0, 7.0, 0.0, 0.0]
        assert model_config.transformer.sliding_attention_setting == {
            "window_size": [64, 0],
            "num_attention_heads": 8,
            "num_query_groups": 2,
            "kv_channels": 64,
        }
        # Step3.7 keeps the main decoder's attention types, its Q/K norm default and its separate
        # lm_head, and always uses the native head-wise gate.
        assert model_config.transformer.layer_types == _LAYER_TYPES
        assert model_config.qk_layernorm is True
        assert model_config.share_embeddings_and_output_weights is False
        assert model_config.transformer.head_wise_attn_gate is True
        assert model_config.attention_output_gate is False
        assert model_config.moe_router_enable_expert_bias is True
        assert model_config.moe_router_dtype == "fp32"
        assert model_config.params_dtype == torch.bfloat16
        assert model_config.vision_config is hf_config.vision_config
        assert model_config.image_token_id == 128001
        assert model_config.understand_projector_stride == 2
        assert model_config.projector_bias is False
        assert model_config.language_max_sequence_length == 4096

    def test_step37_defaults_differ_from_step35_text_mapping(self, monkeypatch):
        monkeypatch.setattr(_step35_bridge_mod, "_mcore_supports_head_wise_attn_gate", lambda: False)
        hf_config = _make_hf_config(rope_scaling={"rope_type": "yarn", "factor": 2.0})

        step35_kwargs = Step35Bridge().hf_config_to_model_config_kwargs(hf_config.text_config)
        step37_kwargs = Step37Bridge().hf_config_to_model_config_kwargs(hf_config)

        assert (step35_kwargs["qk_layernorm"], step37_kwargs["qk_layernorm"]) == (False, True)
        assert (step35_kwargs["attention_output_gate"], "attention_output_gate" in step37_kwargs) == (True, False)
        assert step35_kwargs["layer_types"] == _LAYER_TYPES + _MTP_LAYER_TYPES
        assert step37_kwargs["layer_types"] == _LAYER_TYPES
        assert (step35_kwargs["position_embedding_type"], step37_kwargs["position_embedding_type"]) == ("yarn", "rope")
        assert step37_kwargs["share_embeddings_and_output_weights"] is False

    def test_hf_flags_override_step37_defaults(self):
        model_config = Step37Bridge().hf_config_to_model_config(
            _make_hf_config(use_qk_norm=False, use_head_wise_attn_gate=False, tie_word_embeddings=True)
        )

        assert model_config.qk_layernorm is False
        assert model_config.transformer.head_wise_attn_gate is False
        assert model_config.share_embeddings_and_output_weights is True

    def test_model_config_matches_provider(self):
        hf_config = _make_hf_config()
        bridge = Step37Bridge()
        model_config = bridge.hf_config_to_model_config(hf_config)
        with pytest.warns(FutureWarning):
            provider = bridge.provider_bridge(SimpleNamespace(config=hf_config))

        model_fields = {f.name for f in fields(model_config)} | {f.name for f in fields(model_config.transformer)}
        shared = {f.name for f in fields(provider)} & model_fields
        assert {"vision_config", "image_token_id", "projector_bias", "language_max_sequence_length"} <= shared
        mismatches = {
            name: (getattr(model_config, name), getattr(provider, name))
            for name in sorted(shared)
            if getattr(model_config, name) != getattr(provider, name)
        }
        assert mismatches == {}
        # The provider-only field keeps the HF sliding-attention setting.
        assert provider.attention_other_setting == hf_config.text_config.attention_other_setting

    def test_layer_spec_target_passes_checkpoint_load_validation(self):
        """The serialized spec target must be loadable by the checkpoint instantiate path."""
        from megatron.bridge.utils.instantiate_utils import _validate_target_prefix

        spec = Step37Bridge().hf_config_to_model_config(_make_hf_config()).transformer_layer_spec
        target = f"{spec.__module__}.{spec.__qualname__}"
        _validate_target_prefix(target=target, full_key="transformer_layer_spec")

    def test_model_config_round_trips_through_serialization(self):
        model_config = Step37Bridge().hf_config_to_model_config(_make_hf_config())
        model_config.freeze_vision_model = True

        restored = Step37ModelConfig.from_dict(model_config.as_dict())

        assert isinstance(restored, Step37ModelConfig)
        assert isinstance(restored.transformer, Step35TransformerConfig)
        assert restored.transformer_layer_spec is build_step35_layer_spec
        assert restored.freeze_vision_model is True
        assert isinstance(restored.vision_config, Step37VisionConfig)
        # The vision shape is set in __init__ rather than declared as dataclass fields.
        assert restored.vision_config.to_dict() == model_config.vision_config.to_dict()
        assert (restored.vision_config.width, restored.vision_config.layers) == (64, 2)
        for name in (
            "layer_types",
            "rotary_percents",
            "rotary_base_per_layer",
            "swiglu_limits",
            "head_wise_attn_gate",
        ):
            assert getattr(restored.transformer, name) == getattr(model_config.transformer, name), name


class _BuildRecorders:
    """Stand-ins for the language model, vision tower and projector that Step37Model builds."""

    def __init__(self):
        self.records = {}
        records = self.records

        class LanguageModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                records["language"] = kwargs
                self.share_embeddings_and_output_weights = kwargs["share_embeddings_and_output_weights"]
                self.embedding = nn.Identity()

        class VisionModel(nn.Module):
            def __init__(self, vision_config):
                super().__init__()
                records["vision_config"] = vision_config
                self.weight = nn.Parameter(torch.zeros(2))
                self.register_buffer("freqs_cache", torch.zeros(2), persistent=False)

        class Projector(nn.Module):
            def __init__(self, language_embedding, encoder_output_dim, hidden_size, projector_bias=False):
                super().__init__()
                records["projector"] = (encoder_output_dim, hidden_size, projector_bias)
                self.align_projector = nn.Linear(2, 2, bias=projector_bias)

        self.patches = (
            patch.object(_step37_model_mod, "Step37GPTModel", LanguageModel),
            patch.object(_step37_model_mod, "Step37VisionModel", VisionModel),
            patch.object(_step37_model_mod, "ImageInsertEmbedding", Projector),
        )

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self.records

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


def _pg_collection():
    return SimpleNamespace(cp=None, tp=None, pp=None, embd=None)


class TestStep37ModelBuilder:
    def _model_config(self, **overrides) -> Step37ModelConfig:
        model_config = Step37Bridge().hf_config_to_model_config(_make_hf_config())
        model_config.transformer_layer_spec = SimpleNamespace(name="decoder-spec")
        for name, value in overrides.items():
            setattr(model_config, name, value)
        return model_config

    def test_builds_step37_model_from_model_config(self):
        model_config = self._model_config(projector_bias=True, freeze_vision_model=True)

        with (
            _BuildRecorders() as records,
            patch.object(Step37Model, "freeze") as freeze,
        ):
            model = Step37ModelBuilder(model_config).build_model(_pg_collection(), pre_process=True, post_process=True)

        assert isinstance(model, Step37Model)
        assert model.config is model_config.transformer
        language = records["language"]
        assert language["config"] is model_config.transformer
        assert language["transformer_layer_spec"] is model_config.transformer_layer_spec
        assert language["vocab_size"] == 1024
        assert language["max_sequence_length"] == 4096
        assert language["rotary_base"] == 5e6
        assert language["share_embeddings_and_output_weights"] is False
        assert language["mtp_block_spec"] is None
        assert records["vision_config"] is model_config.vision_config
        assert records["projector"] == (64 * 4, 256, True)
        freeze.assert_called_once_with(
            freeze_language_model=False, freeze_vision_model=True, freeze_vision_projection=False
        )

    def test_casts_vision_tower_and_projector_to_params_dtype(self):
        """Conversion builds without Float16Module, which casts these torch modules on the provider path."""
        with _BuildRecorders():
            model = Step37ModelBuilder(self._model_config()).build_model(_pg_collection(), True, True)

        assert model.vision_model.weight.dtype == torch.bfloat16
        assert model.vision_model.freqs_cache.dtype == torch.bfloat16
        assert model.image_insert_embedding.align_projector.weight.dtype == torch.bfloat16

    def test_later_pipeline_stage_has_no_vision_tower(self):
        with _BuildRecorders() as records:
            model = Step37ModelBuilder(self._model_config()).build_model(
                _pg_collection(), pre_process=False, post_process=True, vp_stage=0
            )

        assert model.vision_model is None
        assert model.image_insert_embedding is None
        assert "vision_config" not in records
        assert records["language"]["pre_process"] is False
        assert model.vp_stage == 0

    def test_resolves_default_layer_spec_with_model_config(self):
        model_config = self._model_config(transformer_layer_spec=None)
        sentinel = SimpleNamespace(name="resolved-spec")

        with (
            _BuildRecorders() as records,
            patch.object(_step37_builder_mod, "get_step37_text_layer_spec", return_value=sentinel) as spec_fn,
        ):
            Step37ModelBuilder(model_config).build_model(_pg_collection(), True, True, vp_stage=None)

        spec_fn.assert_called_once_with(model_config, vp_stage=None)
        assert records["language"]["transformer_layer_spec"] is sentinel

    def test_rejects_other_model_configs_and_missing_vision_config(self):
        with pytest.raises(TypeError, match="Step37ModelConfig"):
            Step37ModelBuilder(SimpleNamespace()).build_model(_pg_collection())
        with pytest.raises(ValueError, match="vision_config"):
            Step37ModelBuilder(self._model_config(vision_config=None)).build_model(_pg_collection())


class TestStep37BridgeReverseConfig:
    def test_checkpoint_mtp_override_is_nested_in_exported_config(self):
        reference_config = Step37Config(text_config={"num_nextn_predict_layers": 3})
        provider = SimpleNamespace(mtp_num_layers=0)

        checkpoint_config = Step37Bridge.megatron_to_hf_config(provider)
        exported_config = conform_config_to_reference(checkpoint_config, reference_config.to_dict())
        synthesized_config = Step37Config(**exported_config)

        assert synthesized_config.text_config.num_nextn_predict_layers == 0

    def test_projector_bias_schema_roundtrips(self):
        reference_config = Step37Config(projector_bias=False)
        provider = SimpleNamespace(projector_bias=True)

        biased_projector = ImageInsertEmbedding(nn.Identity(), 4, 2, projector_bias=provider.projector_bias)
        unbiased_projector = ImageInsertEmbedding(nn.Identity(), 4, 2, projector_bias=False)
        assert biased_projector.align_projector.bias is not None
        assert unbiased_projector.align_projector.bias is None

        mapping = (
            Step37Bridge().mapping_registry().megatron_to_hf_lookup("image_insert_embedding.align_projector.bias")
        )
        checkpoint_config = Step37Bridge.megatron_to_hf_config(provider)
        exported_config = conform_config_to_reference(checkpoint_config, reference_config.to_dict())
        synthesized_config = Step37Config(**exported_config)

        assert (
            mapping.hf_param if mapping is not None else None,
            synthesized_config.projector_bias,
        ) == ("vit_large_projector.bias", True)


class TestStep37VisionMappings:
    def test_optional_vision_parameters_have_checkpoint_mappings(self):
        vision_config = Step37VisionConfig(
            width=8,
            layers=0,
            heads=1,
            image_size=14,
            patch_size=14,
            mlp_ratio=1,
            use_cls_token=True,
            use_ln_pre=False,
            use_ln_post=True,
        )
        vision_model = Step37VisionModel(vision_config)
        parameter_names = set(dict(vision_model.named_parameters()))
        assert {"class_embedding", "ln_post.weight", "ln_post.bias"} <= parameter_names

        disabled_model = Step37VisionModel(
            Step37VisionConfig(
                width=8,
                layers=0,
                heads=1,
                image_size=14,
                patch_size=14,
                mlp_ratio=1,
                use_cls_token=False,
                use_ln_pre=False,
                use_ln_post=False,
            )
        )
        disabled_parameter_names = set(dict(disabled_model.named_parameters()))
        assert "class_embedding" not in disabled_parameter_names
        assert "ln_post.weight" not in disabled_parameter_names
        assert "ln_post.bias" not in disabled_parameter_names

        bridge = Step37Bridge()
        bridge.hf_config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=0, num_nextn_predict_layers=0)
        )
        registry = bridge.mapping_registry()
        for parameter_name in ("class_embedding", "ln_post.weight", "ln_post.bias"):
            megatron_name = f"vision_model.{parameter_name}"
            mapping = registry.megatron_to_hf_lookup(megatron_name)
            assert mapping is not None
            assert mapping.hf_param == megatron_name


class TestGetStep37TextLayerSpec:
    def test_delegates_to_step35_builder(self):
        """Step3.7's text decoder spec is Step-3.5's hybrid layer spec, passed through verbatim."""
        sentinel_cfg = object()
        sentinel_out = object()
        with patch.object(_step37_block_mod, "build_step35_layer_spec", return_value=sentinel_out) as builder:
            out = get_step37_text_layer_spec(sentinel_cfg, num_experts=None)

        builder.assert_called_once_with(sentinel_cfg, num_experts=None)
        assert out is sentinel_out

    def test_module_exposes_public_builder(self):
        assert _step37_block_mod.build_step35_layer_spec is build_step35_layer_spec
