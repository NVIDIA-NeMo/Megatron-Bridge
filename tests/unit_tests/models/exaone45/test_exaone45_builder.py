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

import functools
import types
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.transformer import TransformerLayer
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_VisionConfig

from megatron.bridge.models.common.te_layers import TERowParallelLinearLayerNorm
from megatron.bridge.models.exaone.exaone45.exaone45_bridge import Exaone45Bridge
from megatron.bridge.models.exaone.exaone45.exaone45_builder import Exaone45ModelBuilder
from megatron.bridge.models.exaone.exaone45.exaone45_config import Exaone45ModelConfig
from megatron.bridge.models.exaone.exaone45.exaone45_provider import exaone_45_transformer_layer_spec
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit

_BUILDER_MODULE = "megatron.bridge.models.exaone.exaone45.exaone45_builder"
_MODEL_MODULE = "megatron.bridge.models.exaone.exaone45.modelling_exaone45.model"


def _make_hf_config(*, mtp_num_layers: int = 0) -> SimpleNamespace:
    text_config = SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=768,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        vocab_size=1024,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        rope_parameters={"rope_type": "llama3", "rope_theta": 1000000.0, "factor": 16.0},
        sliding_window=1024,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        num_nextn_predict_layers=mtp_num_layers,
        mtp_loss_scaling_factor=0.05,
        mtp_share_layers=True,
    )
    vision_config = Exaone4_5_VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=8,
        num_key_value_heads=4,
        out_hidden_size=256,
        fullatt_block_indexes=[1],
    )
    return SimpleNamespace(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=False,
        image_token_id=67,
        video_token_id=68,
        vision_start_token_id=73,
    )


def _model_config(**overrides) -> Exaone45ModelConfig:
    model_config = Exaone45Bridge().hf_config_to_model_config(_make_hf_config())
    for name, value in overrides.items():
        setattr(model_config, name, value)
    return model_config


def _pg_collection() -> SimpleNamespace:
    return SimpleNamespace(cp=Mock(), tp=Mock(), pp=Mock(), embd=Mock())


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _comparable(value):
    """Compare config callables (fresh init-method closures per config) by name."""
    if isinstance(value, functools.partial):
        return ("partial", _comparable(value.func), value.args, tuple(sorted(value.keywords.items())))
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        return ("function", value.__module__, value.__qualname__)
    return value


def _config_mismatches(expected, actual) -> dict:
    return {
        f.name: (getattr(expected, f.name), getattr(actual, f.name))
        for f in fields(actual)
        if not f.name.startswith("_")
        and _comparable(getattr(expected, f.name)) != _comparable(getattr(actual, f.name))
    }


class TestExaone45ModelBuilder:
    def test_config_names_builder(self):
        assert Exaone45ModelConfig.builder == f"{_BUILDER_MODULE}.Exaone45ModelBuilder"
        assert _model_config().get_builder_cls() is Exaone45ModelBuilder

    def test_rejects_other_model_configs(self):
        model_config = BridgeGPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
        )

        with pytest.raises(TypeError, match="Exaone45ModelConfig"):
            Exaone45ModelBuilder(model_config).build_model(_pg_collection(), pre_process=True, post_process=True)

    def test_requires_vision_config(self):
        model_config = _model_config(vision_config=None)

        with pytest.raises(ValueError, match="vision_config"):
            Exaone45ModelBuilder(model_config).build_model(_pg_collection(), pre_process=True, post_process=True)

    def test_builds_exaone45_model_from_model_config(self):
        model_config = _model_config()
        pg_collection = _pg_collection()

        with patch(f"{_BUILDER_MODULE}.Exaone45Model") as model_cls:
            model = Exaone45ModelBuilder(model_config).build_model(
                pg_collection, pre_process=True, post_process=False, vp_stage=None
            )

        kwargs = model_cls.call_args.kwargs
        assert kwargs["language_transformer_config"] is model_config.transformer
        assert kwargs["vision_transformer_config"] is model_config.vision_config
        assert kwargs["model_config"] is model_config
        assert kwargs["pg_collection"] is pg_collection
        assert (kwargs["pre_process"], kwargs["post_process"], kwargs["vp_stage"]) == (True, False, None)
        layer_spec = kwargs["language_transformer_layer_spec"]
        assert layer_spec.module is TransformerLayer
        assert layer_spec.submodules.self_attention.submodules.linear_proj is TERowParallelLinearLayerNorm
        assert layer_spec.submodules.mlp.submodules.linear_fc2 is TERowParallelLinearLayerNorm
        # Conversion builds without Float16Module, so the builder casts the fp32 vision encoder itself.
        model.vision_model.to.assert_called_once_with(dtype=torch.bfloat16)
        model.freeze.assert_not_called()

    @pytest.mark.parametrize(
        ("pp_rank", "expected"),
        [(0, (True, False)), (1, (False, True))],
    )
    def test_default_stage_flags_follow_pipeline_rank(self, pp_rank, expected):
        pg_collection = _pg_collection()
        pg_collection.pp = _FakeGroup(rank=pp_rank, size=2)

        with (
            patch(f"{_BUILDER_MODULE}.Exaone45Model") as model_cls,
            patch("torch.distributed.is_initialized", return_value=True),
        ):
            Exaone45ModelBuilder(_model_config()).build_model(pg_collection)

        kwargs = model_cls.call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"]) == expected

    def test_layer_spec_resolution(self):
        spec = exaone_45_transformer_layer_spec(None)
        with_vp_stage = Mock(return_value=spec)

        def spec_with_vp_stage(config, vp_stage=None):
            return with_vp_stage(config, vp_stage=vp_stage)

        for transformer_layer_spec, expected_calls in ((None, 0), (spec, 0), (spec_with_vp_stage, 1)):
            model_config = _model_config(
                transformer_layer_spec=transformer_layer_spec, virtual_pipeline_model_parallel_size=2
            )
            with patch(f"{_BUILDER_MODULE}.Exaone45Model") as model_cls:
                Exaone45ModelBuilder(model_config).build_model(
                    _pg_collection(), pre_process=True, post_process=True, vp_stage=1
                )
            assert model_cls.call_args.kwargs["language_transformer_layer_spec"] == spec
        with_vp_stage.assert_called_once_with(model_config, vp_stage=1)

    def test_applies_freeze_options(self):
        model_config = _model_config(freeze_language_model=True, freeze_mtp_model=True)

        with patch(f"{_BUILDER_MODULE}.Exaone45Model") as model_cls:
            model = Exaone45ModelBuilder(model_config).build_model(
                _pg_collection(), pre_process=True, post_process=True
            )

        assert model is model_cls.return_value
        model.freeze.assert_called_once_with(
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=False,
            freeze_mtp_model=True,
        )

    @pytest.mark.parametrize(("pre_process", "post_process"), [(True, True), (False, True)])
    def test_builder_and_provider_build_the_same_model(self, pre_process, post_process):
        """The builder must hand Exaone45Model the same construction inputs as the provider."""
        hf_config = _make_hf_config()
        bridge = Exaone45Bridge()
        model_config = bridge.hf_config_to_model_config(hf_config)
        with pytest.warns(FutureWarning):
            provider = bridge.provider_bridge(SimpleNamespace(config=hf_config))
        for config in (model_config, provider):
            config.recompute_granularity = "full"
            config.recompute_method = "uniform"
            config.recompute_num_layers = 1
            config.finalize()

        def build(make_model):
            records = {}

            class LanguageModel(torch.nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()
                    records["language"] = kwargs
                    self.config = kwargs["config"]
                    self.share_embeddings_and_output_weights = kwargs["share_embeddings_and_output_weights"]

            class VisionModel(torch.nn.Module):
                def __init__(self, transformer_config, transformer_layer_spec, patch_merger_spec, **kwargs):
                    super().__init__()
                    records["vision"] = transformer_config
                    self.weight = torch.nn.Parameter(torch.zeros(2))

            with (
                patch(f"{_MODEL_MODULE}.Exaone45GPTModel", new=LanguageModel),
                patch(f"{_MODEL_MODULE}.Exaone45VisionModel", new=VisionModel),
            ):
                model = make_model()
            return model, records

        pg_collection = _pg_collection()
        provider._pg_collection = pg_collection
        provider_model, provider_records = build(
            lambda: provider.provide(pre_process=pre_process, post_process=post_process)
        )
        builder_model, builder_records = build(
            lambda: Exaone45ModelBuilder(model_config).build_model(
                pg_collection, pre_process=pre_process, post_process=post_process
            )
        )

        provider_kwargs = dict(provider_records["language"])
        builder_kwargs = dict(builder_records["language"])
        provider_language_config = provider_kwargs.pop("config")
        builder_language_config = builder_kwargs.pop("config")
        assert builder_language_config is model_config.transformer
        assert builder_kwargs == provider_kwargs
        assert builder_kwargs["max_sequence_length"] == 4096
        assert _config_mismatches(provider_language_config, builder_language_config) == {}
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "position_embedding_type"):
            assert getattr(builder_model, name) == getattr(provider_model, name), name
        if pre_process:
            assert _config_mismatches(provider_records["vision"], builder_records["vision"]) == {}
            assert builder_records["vision"].recompute_granularity == "full"
            assert provider_model.vision_model.weight.dtype is torch.float32
            assert builder_model.vision_model.weight.dtype is torch.bfloat16
        else:
            assert "vision" not in provider_records and "vision" not in builder_records

    def test_mtp_block_spec_uses_the_language_transformer_config(self):
        model_config = Exaone45Bridge().hf_config_to_model_config(_make_hf_config(mtp_num_layers=1))
        model_config.finalize()
        mtp_spec = Mock()
        captured = {}

        class LanguageModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)
                self.config = kwargs["config"]
                self.share_embeddings_and_output_weights = kwargs["share_embeddings_and_output_weights"]

        with (
            patch(f"{_MODEL_MODULE}.Exaone45GPTModel", new=LanguageModel),
            patch(
                "megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec", return_value=mtp_spec
            ) as get_mtp_spec,
        ):
            model = Exaone45ModelBuilder(model_config).build_model(
                _pg_collection(), pre_process=False, post_process=True
            )

        assert model.use_mtp_postprocess is True
        assert captured["mtp_block_spec"] is mtp_spec
        assert get_mtp_spec.call_args.args[0] is model_config.transformer
        assert get_mtp_spec.call_args.kwargs == {"use_transformer_engine": True, "vp_stage": None}
