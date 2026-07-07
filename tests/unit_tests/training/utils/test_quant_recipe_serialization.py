from dataclasses import dataclass

import pytest
import yaml
from megatron.core.quantization.quant_config import GlobMatcher, MatchContext, Matcher, RecipeConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.model_load_save import load_model_config
from megatron.bridge.training.utils.config_utils import _ConfigContainerBase


pytestmark = pytest.mark.unit


@dataclass
class _RunConfig(_ConfigContainerBase):
    model: GPTModelProvider


class _UnsupportedMatcher(Matcher):
    def match(self, context: MatchContext) -> str | None:
        return None


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


def test_legacy_stateless_quant_recipe_is_ignored(tmp_path, caplog) -> None:
    run_config_path = tmp_path / "run_config.yaml"
    _RunConfig(model=_model_provider()).to_yaml(str(run_config_path))
    run_config = yaml.safe_load(run_config_path.read_text())
    run_config["model"]["quant_recipe"] = {
        "_call_": True,
        "_target_": "megatron.core.quantization.quant_config.RecipeConfig",
    }
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


def test_upstream_recipe_serializer_takes_precedence(monkeypatch) -> None:
    recipe = _quant_recipe()
    monkeypatch.setattr(recipe, "to_cfg_dict", lambda: {"serializer": "upstream"}, raising=False)

    assert _ConfigContainerBase._convert_value_to_dict(recipe) == {"serializer": "upstream"}
