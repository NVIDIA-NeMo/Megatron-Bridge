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

"""Unit tests for the builder-backed Gemma 4 vision-language configs and builders."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Gemma4Config, Gemma4TextConfig

from megatron.bridge.models.gemma.gemma4_builder import (
    Gemma4DenseModelBuilder,
    Gemma4ModelBuilder,
    gemma4_moe_layer_spec,
)
from megatron.bridge.models.gemma.gemma4_config import Gemma4ModelConfig
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
from megatron.bridge.models.gemma_vl.gemma4_vl_builder import Gemma4DenseVLModelBuilder, Gemma4VLModelBuilder
from megatron.bridge.models.gemma_vl.gemma4_vl_config import (
    Gemma4DenseVLModelConfig,
    Gemma4DenseVLTransformerConfig,
    Gemma4VLModelConfig,
    Gemma4VLTransformerConfig,
)
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4DenseVLProvider, Gemma4VLModelProvider
from megatron.bridge.recipes.gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.model_load_save import load_model_config


pytestmark = pytest.mark.unit

_BUILDER_MODULE = "megatron.bridge.models.gemma_vl.gemma4_vl_builder"

_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_k_eq_v": True,
    "head_dim": 64,
    "global_head_dim": 128,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 256,
    "intermediate_size": 512,
    "max_position_embeddings": 2048,
    "num_attention_heads": 4,
    "num_hidden_layers": 6,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "vocab_size": 1000,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
    },
}

_VISION_CONFIG = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2}


def _dense_hf_config() -> Gemma4Config:
    """A dense checkpoint with vision and audio encoders, like E2B and E4B."""
    text_config = Gemma4TextConfig(
        **_TEXT_CONFIG,
        enable_moe_block=False,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=1024,
    )
    return Gemma4Config(
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text_config,
        vision_config=_VISION_CONFIG,
        audio_config={},
        eos_token_id=[1, 106],
    )


def _moe_hf_config() -> Gemma4Config:
    """A MoE checkpoint with a vision encoder only, like 26B-A4B."""
    text_config = Gemma4TextConfig(
        **_TEXT_CONFIG,
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=64,
        hidden_size_per_layer_input=0,
    )
    return Gemma4Config(
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text_config,
        vision_config=_VISION_CONFIG,
        eos_token_id=[1, 106],
    )


_CASES = pytest.mark.parametrize(
    ("make_hf_config", "config_class", "transformer_class", "provider_class", "builder_class"),
    [
        (
            _dense_hf_config,
            Gemma4DenseVLModelConfig,
            Gemma4DenseVLTransformerConfig,
            Gemma4DenseVLProvider,
            Gemma4DenseVLModelBuilder,
        ),
        (_moe_hf_config, Gemma4VLModelConfig, Gemma4VLTransformerConfig, Gemma4VLModelProvider, Gemma4VLModelBuilder),
    ],
    ids=["dense", "moe"],
)


class TestGemma4VLModelConfigFromHF:
    @_CASES
    def test_config_types(self, make_hf_config, config_class, transformer_class, provider_class, builder_class):
        model_config = Gemma4VLBridge().hf_config_to_model_config(make_hf_config())

        assert type(model_config) is config_class
        assert type(model_config.transformer) is transformer_class
        assert model_config.get_builder_cls() is builder_class

    @_CASES
    def test_vision_language_fields(
        self, make_hf_config, config_class, transformer_class, provider_class, builder_class
    ):
        hf_config = make_hf_config()
        model_config = Gemma4VLBridge().hf_config_to_model_config(hf_config)

        # Gemma4VLModel reads these from the transformer config it is built with.
        transformer = model_config.transformer
        assert transformer.vision_config is hf_config.vision_config
        assert transformer.text_config is hf_config.text_config
        assert transformer.audio_config is hf_config.audio_config
        assert transformer.image_token_id == 258_880
        assert transformer.audio_token_id == 258_881
        # The other settings stay on the outer config.
        assert "video_token_id" not in {field.name for field in dataclasses.fields(transformer)}
        assert model_config.video_token_id == 258_884
        assert model_config.bos_token_id == 2
        assert model_config.eos_token_id == [1, 106]
        assert model_config.vision_soft_tokens_per_image == 280
        assert not (
            model_config.freeze_language_model
            or model_config.freeze_vision_model
            or model_config.freeze_vision_projection
        )

    def test_moe_language_settings(self):
        model_config = Gemma4VLBridge().hf_config_to_model_config(_moe_hf_config())

        # Gemma4VLModel scatters the merged embeddings for sequence parallelism itself.
        assert model_config.scatter_embedding_sequence_parallel is False
        assert model_config.transformer_layer_spec is gemma4_moe_layer_spec
        assert model_config.rotary_base == (10_000.0, 1_000_000.0)
        assert model_config.num_moe_experts == 8

    def test_dense_language_settings(self):
        model_config = Gemma4VLBridge().hf_config_to_model_config(_dense_hf_config())

        assert model_config.scatter_embedding_sequence_parallel is True
        assert model_config.transformer_layer_spec is None
        assert model_config.per_layer_embed_dim == 32
        assert model_config.num_kv_shared_layers == 2

    @_CASES
    def test_builder_config_matches_legacy_provider(
        self, make_hf_config, config_class, transformer_class, provider_class, builder_class
    ):
        """Every field the provider and the builder config share has the same value."""
        hf_config = make_hf_config()
        provider = Gemma4VLBridge().provider_bridge(SimpleNamespace(config=hf_config))
        model_config = Gemma4VLBridge().hf_config_to_model_config(hf_config)

        assert type(provider) is provider_class
        mismatches = {}
        for owner in (model_config, model_config.transformer):
            for field in dataclasses.fields(owner):
                # The layer spec is chosen by the builder, not read from the provider.
                if field.name in {"transformer", "transformer_layer_spec"} or not hasattr(provider, field.name):
                    continue
                expected, actual = getattr(provider, field.name), getattr(owner, field.name)
                # Both carry the source Hugging Face sub-configs, which cannot be compared by value.
                if actual is not expected and actual != expected:
                    mismatches[field.name] = (expected, actual)
        assert mismatches == {}

    @_CASES
    def test_export_config_matches_legacy_provider(
        self, make_hf_config, config_class, transformer_class, provider_class, builder_class
    ):
        hf_config = make_hf_config()
        provider = Gemma4VLBridge().provider_bridge(SimpleNamespace(config=hf_config))
        model_config = Gemma4VLBridge().hf_config_to_model_config(hf_config)

        exported = Gemma4VLBridge.megatron_to_hf_config(model_config)

        assert exported == Gemma4VLBridge.megatron_to_hf_config(provider)
        assert exported["architectures"] == ["Gemma4ForConditionalGeneration"]

    def test_provider_kwargs_are_the_builder_kwargs(self):
        hf_config = _moe_hf_config()
        bridge = Gemma4VLBridge()

        assert bridge.hf_config_to_provider_kwargs(hf_config) == bridge.hf_config_to_model_config_kwargs(hf_config)

    def test_flat_overrides_route_to_their_owner(self):
        model_config = Gemma4VLBridge().hf_config_to_model_config(_moe_hf_config())

        model_config.freeze_vision_model = True
        model_config.image_token_id = 7
        model_config.tensor_model_parallel_size = 2

        assert "freeze_vision_model" not in vars(model_config.transformer)
        assert model_config.freeze_vision_model is True
        assert model_config.transformer.image_token_id == 7
        assert model_config.transformer.tensor_model_parallel_size == 2
        with pytest.raises(AttributeError):
            model_config.vision_soft_tokens = 1

    @_CASES
    def test_checkpoint_run_config_reload(
        self, make_hf_config, config_class, transformer_class, provider_class, builder_class, tmp_path
    ):
        """A config saved to run_config.yaml reloads as the same config and exports identically."""
        model_config = Gemma4VLBridge().hf_config_to_model_config(make_hf_config())
        model_config.freeze_vision_model = True
        run_config = vanilla_gpt_pretrain_config()
        run_config.model = model_config
        run_config.to_yaml(str(tmp_path / "run_config.yaml"))

        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is config_class
        assert type(loaded.transformer) is transformer_class
        assert loaded.get_builder_cls() is builder_class
        assert loaded.transformer_layer_spec is model_config.transformer_layer_spec
        assert loaded.freeze_vision_model is True
        # The Hugging Face sub-configs are saved as provider checkpoints save them.
        for name in ("vision_config", "text_config", "audio_config"):
            assert type(getattr(loaded, name)) is type(getattr(model_config, name))
        assert loaded.text_config.hidden_size == model_config.text_config.hidden_size
        assert loaded.text_config.pad_token_id == model_config.text_config.pad_token_id
        assert loaded.vision_config.hidden_size == model_config.vision_config.hidden_size
        assert Gemma4VLBridge.megatron_to_hf_config(loaded) == Gemma4VLBridge.megatron_to_hf_config(model_config)
        # Serialization leaves the live config untouched.
        assert model_config.text_config is not None and model_config.vision_config is not None


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _pg_collection(pp_rank: int = 0, pp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(pp=_FakeGroup(pp_rank, pp_size))


@pytest.fixture
def model_class(monkeypatch):
    """Replace Gemma4VLModel in the builder, recording how the builder constructs it."""
    model_class = Mock(return_value=Mock())
    monkeypatch.setattr(f"{_BUILDER_MODULE}.Gemma4VLModel", model_class)
    return model_class


class TestGemma4VLModelBuilder:
    @pytest.mark.parametrize(
        ("make_hf_config", "builder_class", "text_builder_class"),
        [
            (_dense_hf_config, Gemma4DenseVLModelBuilder, Gemma4DenseModelBuilder),
            (_moe_hf_config, Gemma4VLModelBuilder, Gemma4ModelBuilder),
        ],
        ids=["dense", "moe"],
    )
    def test_build_model_constructs_the_vision_language_model(
        self, model_class, monkeypatch, make_hf_config, builder_class, text_builder_class
    ):
        model_config = Gemma4VLBridge().hf_config_to_model_config(make_hf_config())
        pg_collection = _pg_collection()
        language_model = Mock()
        text_build_model = Mock(return_value=language_model)
        monkeypatch.setattr(text_builder_class, "build_model", text_build_model)

        model = builder_class(model_config).build_model(pg_collection, pre_process=True, post_process=True)

        assert model is model_class.return_value
        (config,), kwargs = model_class.call_args
        assert config is model_config.transformer
        assert (kwargs["pre_process"], kwargs["post_process"], kwargs["vp_stage"]) == (True, True, None)
        assert kwargs["pg_collection"] is pg_collection
        model.freeze.assert_not_called()

        # The language model comes from the Gemma 4 text builder, with the same process groups.
        assert kwargs["build_language_model"](pre_process=True, post_process=False, vp_stage=None) is language_model
        text_build_model.assert_called_once_with(pg_collection, pre_process=True, post_process=False, vp_stage=None)

    @pytest.mark.parametrize(
        ("pp_rank", "pp_size", "pre_process", "post_process"),
        [(0, 1, True, True), (0, 2, True, False), (1, 2, False, True)],
    )
    def test_build_model_derives_stage_flags(
        self, model_class, monkeypatch, pp_rank, pp_size, pre_process, post_process
    ):
        # Megatron-Core reads stage ranks from the process group only once distributed is initialized.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        model_config = Gemma4VLBridge().hf_config_to_model_config(_moe_hf_config())

        Gemma4VLModelBuilder(model_config).build_model(_pg_collection(pp_rank, pp_size))

        kwargs = model_class.call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"]) == (pre_process, post_process)

    def test_build_model_freezes_requested_modules(self, model_class):
        model_config = Gemma4VLBridge().hf_config_to_model_config(_moe_hf_config())
        model_config.freeze_language_model = True
        model_config.freeze_vision_projection = True

        model = Gemma4VLModelBuilder(model_config).build_model(_pg_collection(), pre_process=True, post_process=True)

        model.freeze.assert_called_once_with(
            freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=True
        )

    def test_builders_reject_other_configs(self, model_class):
        dense_config = Gemma4VLBridge().hf_config_to_model_config(_dense_hf_config())
        moe_config = Gemma4VLBridge().hf_config_to_model_config(_moe_hf_config())
        text_config = Gemma4ModelConfig(transformer=moe_config.transformer, vocab_size=1000)

        with pytest.raises(TypeError, match="Gemma4VLModelConfig"):
            Gemma4VLModelBuilder(dense_config).build_model(_pg_collection())
        with pytest.raises(TypeError, match="Gemma4VLModelConfig"):
            Gemma4VLModelBuilder(text_config).build_model(_pg_collection())
        with pytest.raises(TypeError, match="Gemma4DenseVLModelConfig"):
            Gemma4DenseVLModelBuilder(moe_config).build_model(_pg_collection())

    @pytest.mark.parametrize("name", ["vision_config", "text_config"])
    def test_build_model_requires_hf_sub_configs(self, model_class, name):
        model_config = Gemma4VLBridge().hf_config_to_model_config(_moe_hf_config())
        setattr(model_config, name, None)

        with pytest.raises(ValueError, match="vision_config and text_config"):
            Gemma4VLModelBuilder(model_config).build_model(_pg_collection())
        model_class.assert_not_called()
