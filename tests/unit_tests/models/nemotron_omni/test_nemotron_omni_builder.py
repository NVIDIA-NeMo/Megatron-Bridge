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

"""Tests for the builder-backed Nemotron Omni model config and builder."""

import dataclasses
import functools
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.activations import squared_relu

from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.nemotron_omni import nemotron_omni_builder as builder_module
from megatron.bridge.models.nemotron_omni import nemotron_omni_provider as provider_module
from megatron.bridge.models.nemotron_omni import nemotron_omni_sound as sound_module
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import Nemotron35SuperVLBridge, NemotronOmniBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_builder import NemotronOmniModelBuilder
from megatron.bridge.models.nemotron_omni.nemotron_omni_config import NemotronOmniModelConfig
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit


class _DictConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


def _omni_hf_config(*, super_vl: bool = False) -> _DictConfig:
    llm_config = _DictConfig(
        torch_dtype="bfloat16",
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        layer_norm_epsilon=1e-6,
        vocab_size=131072,
        max_position_embeddings=4096,
        hybrid_override_pattern="M*EM*E",
        mamba_head_dim=64,
        mamba_num_heads=4,
        n_groups=2,
        ssm_state_size=16,
        moe_intermediate_size=384,
        moe_shared_expert_intermediate_size=768,
        n_routed_experts=8,
        num_experts_per_tok=2,
    )
    config = _DictConfig(
        architectures=["NemotronH_Nano_Omni_Reasoning_V3"],
        llm_config=llm_config,
        vision_config=_DictConfig(separate_video_embedder=True, video_temporal_patch_size=2),
        sound_config=_DictConfig(
            model_type="parakeet",
            hidden_size=128,
            projection_hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            subsampling_factor=8,
            num_mel_bins=128,
        ),
        projector_hidden_size=1024,
        img_context_token_id=18,
        sound_context_token_id=27,
    )
    if super_vl:
        config.architectures = ["NemotronH_Omni_Reasoning_V3"]
        config.sound_config = None
        config.sound_context_token_id = None
        config.video_temporal_patch_size = 2
        llm_config.num_nextn_predict_layers = 1
        llm_config.mtp_hybrid_override_pattern = "*E"
    return config


def _model_config(*, super_vl: bool = False) -> NemotronOmniModelConfig:
    bridge = Nemotron35SuperVLBridge() if super_vl else NemotronOmniBridge()
    return bridge.hf_config_to_model_config(_omni_hf_config(super_vl=super_vl))


def _provider(*, super_vl: bool = False) -> NemotronOmniModelProvider:
    bridge = Nemotron35SuperVLBridge() if super_vl else NemotronOmniBridge()
    with pytest.warns(FutureWarning):
        return bridge.provider_bridge(SimpleNamespace(config=_omni_hf_config(super_vl=super_vl)))


def _transformer_fields(config) -> dict:
    """Return the transformer-config fields of a language or media config for comparison."""
    fields = {}
    for field in dataclasses.fields(TransformerConfig):
        value = getattr(config, field.name)
        if isinstance(value, functools.partial):
            # Copied configs hold copies of the initializer partials, which compare by identity.
            value = (value.func, value.args, sorted(value.keywords.items()))
        fields[field.name] = value
    return fields


class _RecordedModel(SimpleNamespace):
    def freeze(self, **kwargs):
        self.freeze_kwargs = kwargs


def _record_construction(monkeypatch, module) -> dict:
    """Replace the model, projector and sound encoder constructors of ``module`` with recorders."""
    calls = {}

    def model_factory(**kwargs):
        calls["model"] = kwargs
        return _RecordedModel(
            vision_model=torch.nn.Linear(2, 2) if kwargs["add_encoder"] else None,
            sound_model=kwargs["sound_model"],
            freeze_kwargs=None,
        )

    def projector_factory(**kwargs):
        calls["sound_projection"] = kwargs
        return "sound_projection"

    monkeypatch.setattr(module, "NemotronOmniModel", model_factory)
    monkeypatch.setattr(module, "MultimodalProjector", projector_factory)
    monkeypatch.setattr(sound_module, "BridgeSoundEncoder", lambda config: torch.nn.Linear(2, 2))
    return calls


def _single_stage_process_groups():
    stage = Mock()
    stage.rank.return_value = 0
    stage.size.return_value = 1
    return SimpleNamespace(pp=stage, tp=stage, cp=stage)


def test_config_selects_builder_and_round_trips():
    model_config = _model_config()

    assert model_config.get_builder_cls() is NemotronOmniModelBuilder
    data = model_config.as_dict()
    assert data["_builder_"] == NemotronOmniModelConfig.builder
    restored = NemotronOmniModelConfig.from_dict(data)

    assert type(restored) is NemotronOmniModelConfig
    assert restored.as_dict() == data
    assert restored.activation_func is squared_relu
    assert restored.sound_config == model_config.sound_config
    assert restored.image_token_index == 18


def test_media_defaults_match_the_provider():
    media_fields = {field.name for field in dataclasses.fields(NemotronOmniModelConfig)} - {
        field.name for field in dataclasses.fields(BridgeHybridModelConfig)
    }
    model_config = NemotronOmniModelConfig(transformer=TransformerConfig())
    provider = NemotronOmniModelProvider()

    assert media_fields
    for name in sorted(media_fields):
        assert getattr(model_config, name) == getattr(provider, name), name


def test_config_rejects_provider_only_fields():
    model_config = _model_config()

    with pytest.raises(AttributeError, match="tokenizer_type"):
        model_config.tokenizer_type = "nemotron6-moe"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dynamic_resolution": False}, "only supports dynamic_resolution=True"),
        ({"image_token_index": 0}, "positive image_token_index"),
        ({"sound_context_token_id": 0}, "positive sound_context_token_id"),
        ({"sound_config": None}, "requires sound_config"),
    ],
)
def test_finalize_validates_media_settings(overrides, message):
    model_config = _model_config()
    for name, value in overrides.items():
        setattr(model_config, name, value)

    with pytest.raises(ValueError, match=message):
        model_config.finalize()


def test_builder_rejects_other_model_configs():
    model_config = BridgeHybridModelConfig(transformer=TransformerConfig(num_layers=2))

    with pytest.raises(TypeError, match="Expected NemotronOmniModelConfig"):
        NemotronOmniModelBuilder(model_config).build_model(_single_stage_process_groups())


@pytest.mark.parametrize("super_vl", [False, True], ids=["nano", "super_vl"])
def test_builder_constructs_the_model_the_provider_constructs(monkeypatch, super_vl):
    provider = _provider(super_vl=super_vl)
    provider.finalize()
    provider_calls = _record_construction(monkeypatch, provider_module)
    provider.provide(pre_process=True, post_process=True)

    model_config = _model_config(super_vl=super_vl)
    model_config.finalize()
    builder_calls = _record_construction(monkeypatch, builder_module)
    pg_collection = _single_stage_process_groups()
    NemotronOmniModelBuilder(model_config).build_model(pg_collection, pre_process=True, post_process=True)

    provider_kwargs = provider_calls["model"]
    builder_kwargs = builder_calls["model"]
    assert builder_kwargs.pop("model_config") is model_config
    assert builder_kwargs.pop("pg_collection") is pg_collection
    assert builder_kwargs["language_transformer_config"] is model_config.transformer
    for name in ("language_transformer_config", "vision_transformer_config", "vision_projection_config"):
        assert _transformer_fields(builder_kwargs.pop(name)) == _transformer_fields(provider_kwargs.pop(name)), name
    assert repr(builder_kwargs.pop("language_transformer_layer_spec")) == repr(
        provider_kwargs.pop("language_transformer_layer_spec")
    )
    # The remaining kwargs are values and freshly built media modules.
    for name in ("sound_model", "sound_projection"):
        assert (builder_kwargs.pop(name) is None) == (provider_kwargs.pop(name) is None)
    assert builder_kwargs.keys() == provider_kwargs.keys()
    for name, value in builder_kwargs.items():
        assert repr(value) == repr(provider_kwargs[name]), name
    assert ("sound_projection" in builder_calls) == ("sound_projection" in provider_calls) == (not super_vl)
    if not super_vl:
        builder_projection = builder_calls["sound_projection"]
        provider_projection = provider_calls["sound_projection"]
        assert _transformer_fields(builder_projection.pop("config")) == _transformer_fields(
            provider_projection.pop("config")
        )
        assert repr(builder_projection) == repr(provider_projection)


def test_builder_casts_media_encoders_and_freezes(monkeypatch):
    model_config = _model_config()
    model_config.freeze_vision_model = True
    model_config.freeze_sound_projection = True
    model_config.finalize()
    _record_construction(monkeypatch, builder_module)

    model = NemotronOmniModelBuilder(model_config).build_model(
        _single_stage_process_groups(), pre_process=True, post_process=True
    )

    assert model.vision_model.weight.dtype is torch.bfloat16
    assert model.sound_model.weight.dtype is torch.bfloat16
    assert model.freeze_kwargs == {
        "freeze_language_model": False,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
        "freeze_sound_model": False,
        "freeze_sound_projection": True,
    }


def test_builder_places_media_modules_on_the_first_pipeline_stage(monkeypatch):
    model_config = _model_config()
    model_config.pipeline_model_parallel_size = 2
    model_config.pipeline_dtype = torch.bfloat16
    model_config.finalize()
    calls = _record_construction(monkeypatch, builder_module)
    monkeypatch.setattr(builder_module, "is_pp_first_stage", lambda pp_group: False)

    NemotronOmniModelBuilder(model_config).build_model(
        _single_stage_process_groups(), pre_process=False, post_process=True
    )

    assert calls["model"]["add_encoder"] is False
    assert calls["model"]["sound_model"] is None
    assert "sound_projection" not in calls
