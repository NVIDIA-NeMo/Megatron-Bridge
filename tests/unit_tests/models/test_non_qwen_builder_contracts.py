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

"""Representative build contracts for provider-free non-Qwen VLM builders."""

import pytest
from megatron.core.transformer import TransformerConfig
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.ernie_vl.model_config import Ernie45VLModelBuilder, Ernie45VLModelConfig
from megatron.bridge.models.gemma.model_config import Gemma3ModelBuilder
from megatron.bridge.models.gemma_vl.model_config import Gemma3VLModelBuilder, Gemma3VLModelConfig
from megatron.bridge.models.glm_vl.model_config import GLM45VModelBuilder, GLM45VModelConfig
from megatron.bridge.models.kimi_vl.model_config import KimiK25VLModelBuilder, KimiK25VLModelConfig
from megatron.bridge.models.ministral3.model_config import Ministral3ModelBuilder, Ministral3ModelConfig


pytestmark = pytest.mark.unit


def _transformer() -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)


@pytest.mark.parametrize(
    ("builder_class", "config", "model_target"),
    [
        (
            GLM45VModelBuilder,
            GLM45VModelConfig(transformer=_transformer(), vocab_size=32),
            "megatron.bridge.models.glm_vl.model_config.GLM45VModel",
        ),
        (
            KimiK25VLModelBuilder,
            KimiK25VLModelConfig(transformer=_transformer(), vocab_size=32),
            "megatron.bridge.models.kimi_vl.model_config.KimiK25VLModel",
        ),
    ],
)
def test_vlm_wrapper_builders_forward_stage_and_process_groups(monkeypatch, builder_class, config, model_target):
    language_model = object()
    built_model = object()
    captured = {}
    pg_collection = object()

    monkeypatch.setattr(GPTModelBuilder, "build_model", lambda *args, **kwargs: language_model)
    monkeypatch.setattr(model_target, lambda **kwargs: captured.update(kwargs) or built_model)

    result = builder_class(config).build_model(
        pg_collection,
        pre_process=False,
        post_process=True,
        vp_stage=3,
    )

    assert result is built_model
    assert captured["language_model"] is language_model
    assert captured["pg_collection"] is pg_collection
    assert captured["pre_process"] is False
    assert captured["post_process"] is True
    assert captured["vp_stage"] == 3


def test_gemma3_vl_builder_wraps_the_built_language_stage(monkeypatch):
    config = Gemma3VLModelConfig(transformer=_transformer(), vocab_size=32)
    language_model = object()
    built_model = object()
    captured = {}
    pg_collection = object()

    monkeypatch.setattr(Gemma3ModelBuilder, "build_model", lambda *args, **kwargs: language_model)
    monkeypatch.setattr(
        "megatron.bridge.models.gemma_vl.model_config.Gemma3VLModel",
        lambda **kwargs: captured.update(kwargs) or built_model,
    )

    result = Gemma3VLModelBuilder(config).build_model(
        pg_collection,
        pre_process=True,
        post_process=False,
        vp_stage=1,
    )

    assert result is built_model
    assert captured["language_model"] is language_model
    assert captured["pg_collection"] is pg_collection
    assert captured["pre_process"] is True
    assert captured["post_process"] is False
    assert captured["vp_stage"] == 1


def test_ernie_vl_builder_forwards_both_exact_transformer_configs(monkeypatch):
    language_transformer = _transformer()
    vision_transformer = _transformer()
    config = Ernie45VLModelConfig(
        transformer=language_transformer,
        vision_transformer=vision_transformer,
        vocab_size=32,
        hf_config={},
        vision_config={},
    )
    language_model = object()
    built_model = object()
    captured = {}
    pg_collection = object()

    monkeypatch.setattr(GPTModelBuilder, "build_model", lambda *args, **kwargs: language_model)
    monkeypatch.setattr(
        "megatron.bridge.models.ernie_vl.model_config.Ernie45VLModel",
        lambda **kwargs: captured.update(kwargs) or built_model,
    )

    result = Ernie45VLModelBuilder(config).build_model(
        pg_collection,
        pre_process=False,
        post_process=True,
        vp_stage=2,
    )

    assert result is built_model
    assert captured["language_model"] is language_model
    assert captured["language_transformer_config"] is language_transformer
    assert type(captured["vision_transformer_config"]) is TransformerConfig
    assert captured["pg_collection"] is pg_collection
    assert captured["pre_process"] is False
    assert captured["post_process"] is True
    assert captured["vp_stage"] == 2


def test_ministral_builder_wraps_yarn_language_stage(monkeypatch):
    config = Ministral3ModelConfig(transformer=_transformer(), vocab_size=32)
    language_model = object()
    built_model = object()
    captured = {}
    pg_collection = object()

    monkeypatch.setattr(
        "megatron.bridge.models.ministral3.model_config.build_gpt_with_yarn",
        lambda *args, **kwargs: language_model,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.ministral3.model_config.Ministral3Model",
        lambda **kwargs: captured.update(kwargs) or built_model,
    )

    result = Ministral3ModelBuilder(config).build_model(
        pg_collection,
        pre_process=True,
        post_process=False,
        vp_stage=4,
    )

    assert result is built_model
    assert captured["language_model"] is language_model
    assert captured["pg_collection"] is pg_collection
    assert captured["pre_process"] is True
    assert captured["post_process"] is False
    assert captured["vp_stage"] == 4
