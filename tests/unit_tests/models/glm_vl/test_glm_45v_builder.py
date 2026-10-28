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

"""Unit tests for the builder-backed GLM-4.5V model config and builder."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import Glm4vMoeConfig

from megatron.bridge.models.glm.glm45_bridge import GLM45Bridge
from megatron.bridge.models.glm_vl.glm_45v_bridge import GLM45VBridge
from megatron.bridge.models.glm_vl.glm_45v_builder import GLM45VModelBuilder
from megatron.bridge.models.glm_vl.glm_45v_config import GLM45VModelConfig, GLM45VTransformerConfig
from megatron.bridge.models.glm_vl.glm_45v_provider import GLM45VModelProvider
from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.recipes.gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.model_load_save import load_model_config


pytestmark = pytest.mark.unit

_BUILDER_MODULE = "megatron.bridge.models.glm_vl.glm_45v_builder"


def _hf_config() -> Glm4vMoeConfig:
    """A small GLM-4.5V config with two dense and two MoE layers."""
    return Glm4vMoeConfig(
        architectures=["Glm4vMoeForConditionalGeneration"],
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=4,
            first_k_dense_replace=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            vocab_size=256,
            dtype="bfloat16",
        ),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
    )


def _model_config(**overrides) -> GLM45VModelConfig:
    model_config = GLM45VBridge().hf_config_to_model_config(_hf_config())
    for name, value in overrides.items():
        setattr(model_config, name, value)
    return model_config


class TestGLM45VModelConfigFromHF:
    def test_config_types(self):
        model_config = _model_config()

        assert type(model_config) is GLM45VModelConfig
        assert type(model_config.transformer) is GLM45VTransformerConfig
        assert model_config.get_builder_cls() is GLM45VModelBuilder
        assert GLM45VBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_defaults(self):
        config = GLM45VModelConfig(
            transformer=GLM45VTransformerConfig(num_layers=4, hidden_size=64, num_attention_heads=4)
        )

        assert config.position_embedding_type == "mrope"
        assert config.scatter_embedding_sequence_parallel is False
        assert config.vision_config is None
        assert config.transformer.return_dict is True
        assert config.transformer.spatial_merge_size == 2

    def test_vision_language_fields(self):
        hf_config = _hf_config()
        model_config = GLM45VBridge().hf_config_to_model_config(hf_config)

        # The Hugging Face Glm4vModel methods bound to GLM45VModel read these from its config.
        transformer = model_config.transformer
        assert transformer.vision_config is hf_config.vision_config
        assert transformer.image_token_id == 151363
        assert transformer.video_token_id == 151364
        assert transformer.return_dict is True
        # The remaining token IDs and the freeze options stay on the outer config.
        outer_fields = {field.name for field in dataclasses.fields(model_config)}
        assert {"eos_token_id", "image_start_token_id", "video_end_token_id", "freeze_vision_model"} <= outer_fields

    def test_language_model_uses_the_text_bridge_mapping(self):
        hf_config = _hf_config()
        model_config = GLM45VBridge().hf_config_to_model_config(hf_config)
        text_kwargs = GLM45Bridge().hf_config_to_model_config_kwargs(hf_config.text_config)

        for name in ("num_layers", "hidden_size", "num_moe_experts", "moe_layer_freq", "moe_router_score_function"):
            assert getattr(model_config, name) == text_kwargs[name], name
        # Values the vision-language provider has always used where the text model differs.
        assert model_config.position_embedding_type == "mrope"
        assert model_config.mrope_section == [8, 12, 12]
        assert model_config.mtp_num_layers == 0
        assert model_config.moe_router_num_groups is None
        assert model_config.moe_router_group_topk is None
        assert model_config.transformer_layer_spec is None

    def test_builder_config_matches_legacy_provider(self):
        """Every field the provider and the builder config share has the same value."""
        hf_config = _hf_config()
        with pytest.warns(FutureWarning):
            provider = GLM45VBridge().provider_bridge(SimpleNamespace(config=hf_config))
        model_config = GLM45VBridge().hf_config_to_model_config(hf_config)

        assert type(provider) is GLM45VModelProvider
        mismatches = {}
        for owner in (model_config, model_config.transformer):
            for field in dataclasses.fields(owner):
                # The layer spec is chosen by the builder, not read from the provider.
                if field.name in {"transformer", "transformer_layer_spec"} or not hasattr(provider, field.name):
                    continue
                if getattr(owner, field.name) != getattr(provider, field.name):
                    mismatches[field.name] = (getattr(provider, field.name), getattr(owner, field.name))
        assert mismatches == {}

    def test_builder_path_does_not_use_the_provider(self):
        bridge = GLM45VBridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(_hf_config())

        assert isinstance(model_config, GLM45VModelConfig)

    def test_flat_overrides_route_to_their_owner(self):
        model_config = _model_config()

        model_config.freeze_vision_model = True
        model_config.image_token_id = 7
        model_config.expert_model_parallel_size = 2

        assert "freeze_vision_model" not in vars(model_config.transformer)
        assert model_config.freeze_vision_model is True
        assert model_config.transformer.image_token_id == 7
        assert model_config.transformer.expert_model_parallel_size == 2
        with pytest.raises(AttributeError):
            model_config.vision_start_token_id = 1

    def test_checkpoint_run_config_reload(self, tmp_path):
        model_config = _model_config(freeze_vision_projection=True)
        run_config = vanilla_gpt_pretrain_config()
        run_config.model = model_config
        run_config.to_yaml(str(tmp_path / "run_config.yaml"))

        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is GLM45VModelConfig
        assert type(loaded.transformer) is GLM45VTransformerConfig
        assert loaded.get_builder_cls() is GLM45VModelBuilder
        assert type(loaded.vision_config) is type(model_config.vision_config)
        assert loaded.vision_config.to_dict() == model_config.vision_config.to_dict()
        for name in (
            "freeze_vision_projection",
            "position_embedding_type",
            "mrope_section",
            "moe_layer_freq",
            "eos_token_id",
        ):
            assert getattr(loaded, name) == getattr(model_config, name), name


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
    """Replace GLM45VModel in the builder, recording how the builder constructs it."""
    model_class = Mock(return_value=Mock())
    monkeypatch.setattr(f"{_BUILDER_MODULE}.GLM45VModel", model_class)
    return model_class


class TestGLM45VModelBuilder:
    def test_build_model_constructs_the_vision_language_model(self, model_class, monkeypatch):
        model_config = _model_config()
        pg_collection = _pg_collection()
        language_model = Mock()
        text_build_model = Mock(return_value=language_model)
        monkeypatch.setattr(GPTModelBuilder, "build_model", text_build_model)

        model = GLM45VModelBuilder(model_config).build_model(pg_collection, pre_process=True, post_process=True)

        assert model is model_class.return_value
        (config,), kwargs = model_class.call_args
        assert config is model_config.transformer
        assert (kwargs["pre_process"], kwargs["post_process"], kwargs["vp_stage"]) == (True, True, None)
        assert kwargs["pg_collection"] is pg_collection
        # Without a Float16Module wrapper the vision encoder would stay in fp32.
        model.visual.to.assert_called_once_with(dtype=model_config.params_dtype)
        model.freeze.assert_not_called()

        # The language model comes from the GPT builder, with the same process groups.
        assert kwargs["build_language_model"](pre_process=True, post_process=False, vp_stage=None) is language_model
        text_build_model.assert_called_once_with(pg_collection, pre_process=True, post_process=False, vp_stage=None)

    def test_build_model_skips_cast_without_vision_encoder(self, model_class):
        model = GLM45VModelBuilder(_model_config()).build_model(
            _pg_collection(pp_rank=1, pp_size=2), pre_process=False, post_process=True
        )

        model.visual.to.assert_not_called()

    @pytest.mark.parametrize(
        ("pp_rank", "pp_size", "vp_stage", "vp_size", "pre_process", "post_process"),
        [
            (0, 1, None, None, True, True),
            (0, 2, None, None, True, False),
            (1, 2, None, None, False, True),
            (0, 2, 1, 2, False, False),
            (1, 2, 1, 2, False, True),
        ],
    )
    def test_build_model_derives_stage_flags(
        self, model_class, monkeypatch, pp_rank, pp_size, vp_stage, vp_size, pre_process, post_process
    ):
        # Megatron-Core reads stage ranks from the process group only once distributed is initialized.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        model_config = _model_config(virtual_pipeline_model_parallel_size=vp_size)

        GLM45VModelBuilder(model_config).build_model(_pg_collection(pp_rank, pp_size), vp_stage=vp_stage)

        kwargs = model_class.call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"], kwargs["vp_stage"]) == (
            pre_process,
            post_process,
            vp_stage,
        )

    def test_build_model_freezes_requested_modules(self, model_class):
        model_config = _model_config(freeze_vision_model=True)

        model = GLM45VModelBuilder(model_config).build_model(_pg_collection(), pre_process=True, post_process=True)

        model.freeze.assert_called_once_with(
            freeze_language_model=False, freeze_vision_model=True, freeze_vision_projection=False
        )

    def test_build_model_rejects_other_configs(self, model_class):
        from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
        from megatron.bridge.models.transformer_config import TransformerConfig

        text_config = BridgeGPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4), vocab_size=128
        )
        with pytest.raises(TypeError, match="GLM45VModelConfig"):
            GLM45VModelBuilder(text_config).build_model(_pg_collection())

    def test_build_model_requires_vision_config(self, model_class):
        with pytest.raises(ValueError, match="vision_config"):
            GLM45VModelBuilder(_model_config(vision_config=None)).build_model(_pg_collection())
        model_class.assert_not_called()
