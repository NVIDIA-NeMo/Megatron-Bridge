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

from dataclasses import dataclass

import pytest
import yaml
from megatron.core.quantization.quant_config import GlobMatcher, MatchContext, Matcher, RecipeConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.model_load_save import load_model_config
from megatron.bridge.training.utils.checkpoint_utils import read_run_config
from megatron.bridge.training.utils.config_utils import _ConfigContainerBase
from megatron.bridge.utils.instantiate_utils import instantiate, target_allowlist


pytestmark = pytest.mark.unit


@dataclass
class _RunConfig(_ConfigContainerBase):
    model: GPTModelProvider


class _UnsupportedMatcher(Matcher):
    def match(self, context: MatchContext) -> str | None:
        return None


class _SerializableMatcher(Matcher):
    def __init__(self, pattern: str, config_key: str):
        self.pattern = pattern
        self.config_key = config_key

    def match(self, context: MatchContext) -> str | None:
        return self.config_key if context.module_path == self.pattern else None

    def to_cfg_dict(self) -> dict[str, str]:
        return {
            "_target_": "megatron.core.quantization.quant_config.GlobMatcher",
            "pattern": self.pattern,
            "config_key": self.config_key,
        }


@dataclass
class SerializableDataclassMatcher(Matcher):
    pattern: str
    config_key: str

    def match(self, context: MatchContext) -> str | None:
        return self.config_key if context.module_path == self.pattern else None


class _RecipeConfigSubclass(RecipeConfig):
    pass


def _quant_recipe() -> RecipeConfig:
    return RecipeConfig(
        matchers=[
            GlobMatcher("decoder.layers.0.*", "first_layer"),
            GlobMatcher("*", "fallback"),
        ],
        config_dict={
            "first_layer": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp8_quantization_recipe": "mxfp8"},
            },
            "fallback": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {},
            },
        },
    )


def _model_provider() -> GPTModelProvider:
    return GPTModelProvider(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        quant_recipe=_quant_recipe(),
    )


def test_quant_recipe_round_trips_through_run_config(tmp_path) -> None:
    run_config_path = tmp_path / "run_config.yaml"
    _RunConfig(model=_model_provider()).to_yaml(str(run_config_path))

    serialized = yaml.safe_load(run_config_path.read_text())
    serialized_recipe = serialized["model"]["quant_recipe"]
    assert serialized_recipe == {
        "_target_": "megatron.core.quantization.quant_config.RecipeConfig",
        "config_dict": _quant_recipe().configs,
        "matchers": [
            {
                "_target_": "megatron.core.quantization.quant_config.GlobMatcher",
                "config_key": "first_layer",
                "pattern": "decoder.layers.0.*",
            },
            {
                "_target_": "megatron.core.quantization.quant_config.GlobMatcher",
                "config_key": "fallback",
                "pattern": "*",
            },
        ],
    }

    loaded_model, mlm_args = load_model_config(str(tmp_path))
    assert mlm_args is None
    assert isinstance(loaded_model.quant_recipe, RecipeConfig)
    assert loaded_model.quant_recipe.configs == _quant_recipe().configs
    assert (
        loaded_model.quant_recipe.match_to_config_key(
            MatchContext(module_path="decoder.layers.0.self_attention.linear_qkv", layer_number=0)
        )
        == "first_layer"
    )
    assert (
        loaded_model.quant_recipe.match_to_config_key(
            MatchContext(module_path="decoder.layers.1.self_attention.linear_qkv", layer_number=1)
        )
        == "fallback"
    )


@pytest.mark.parametrize("include_call_flag", [False, True])
def test_legacy_stateless_quant_recipe_is_ignored(tmp_path, caplog, include_call_flag) -> None:
    run_config_path = tmp_path / "run_config.yaml"
    _RunConfig(model=_model_provider()).to_yaml(str(run_config_path))
    run_config = yaml.safe_load(run_config_path.read_text())
    legacy_recipe = {
        "_target_": "megatron.core.quantization.quant_config.RecipeConfig",
    }
    if include_call_flag:
        legacy_recipe["_call_"] = True
    run_config["model"]["quant_recipe"] = legacy_recipe
    run_config_path.write_text(yaml.safe_dump(run_config))

    with caplog.at_level("WARNING"):
        loaded_model, mlm_args = load_model_config(str(tmp_path))

    assert mlm_args is None
    assert loaded_model.quant_recipe is None
    assert "legacy quantization recipe whose state was not preserved" in caplog.text


def test_unsupported_quant_recipe_matcher_fails_loudly() -> None:
    recipe = RecipeConfig(matchers=[_UnsupportedMatcher()], config_dict={})

    with pytest.raises(TypeError, match="Unsupported quantization recipe matcher type"):
        _ConfigContainerBase._convert_value_to_dict(recipe)


def test_custom_serializable_matcher_uses_its_config_mapping() -> None:
    recipe = RecipeConfig(
        matchers=[_SerializableMatcher("decoder.layers.0.*", "custom")],
        config_dict={"custom": {}},
    )

    serialized = _ConfigContainerBase._convert_value_to_dict(recipe)
    restored = instantiate(serialized)

    assert isinstance(restored.matchers[0], GlobMatcher)
    assert restored.matchers[0].pattern == "decoder.layers.0.*"
    assert restored.matchers[0].config_key == "custom"


def test_dataclass_matcher_uses_base_config_serialization() -> None:
    recipe = RecipeConfig(
        matchers=[SerializableDataclassMatcher("decoder.layers.0.*", "custom")],
        config_dict={"custom": {}},
    )
    serialized = _ConfigContainerBase._convert_value_to_dict(recipe)
    matcher_target = serialized["matchers"][0]["_target_"]
    target_allowlist.add_exact(matcher_target)
    try:
        restored = instantiate(serialized)
    finally:
        target_allowlist.remove_exact(matcher_target)

    assert isinstance(restored.matchers[0], SerializableDataclassMatcher)
    assert restored.matchers[0].pattern == "decoder.layers.0.*"
    assert restored.matchers[0].config_key == "custom"


def test_recipe_subclass_without_serializer_fails_loudly() -> None:
    recipe = _RecipeConfigSubclass(matchers=[], config_dict={})

    with pytest.raises(TypeError, match="RecipeConfig subclasses must implement to_cfg_dict"):
        _ConfigContainerBase._convert_value_to_dict(recipe)


def test_upstream_recipe_serializer_takes_precedence(monkeypatch) -> None:
    recipe = _quant_recipe()
    monkeypatch.setattr(recipe, "to_cfg_dict", lambda: {"serializer": "upstream"}, raising=False)

    assert _ConfigContainerBase._convert_value_to_dict(recipe) == {"serializer": "upstream"}


def test_recipe_class_reference_is_not_treated_as_legacy_stub(tmp_path) -> None:
    run_config_path = tmp_path / "run_config.yaml"
    recipe_class_reference = {
        "_target_": "megatron.core.quantization.quant_config.RecipeConfig",
        "_call_": False,
    }
    run_config_path.write_text(yaml.safe_dump({"recipe_class": recipe_class_reference}))

    loaded = read_run_config(str(run_config_path))

    assert loaded["recipe_class"] == recipe_class_reference
