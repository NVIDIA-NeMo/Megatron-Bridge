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

"""Qwen3.5 text recipes build the language model on a strict builder-backed model config."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import ClassVar

import pytest
from transformers import Qwen3_5Config, Qwen3_5MoeConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.recipes.qwen.gb200 import qwen35
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

# The recipes read published multimodal checkpoint configs, whose language model sits in text_config.
_TEXT_CONFIG = {"max_position_embeddings": 262144, "mtp_num_hidden_layers": 1}
_HF_CONFIGS = {
    "Qwen/Qwen3.5-9B-Base": lambda: Qwen3_5Config(text_config=_TEXT_CONFIG, tie_word_embeddings=False),
    "Qwen/Qwen3.5-35B-A3B-Base": lambda: Qwen3_5MoeConfig(text_config=_TEXT_CONFIG, tie_word_embeddings=False),
    "Qwen/Qwen3.5-35B-A3B": lambda: Qwen3_5MoeConfig(text_config=_TEXT_CONFIG, tie_word_embeddings=False),
}
_DYNAMIC_CP_RECIPE = qwen35.qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config
_RECIPES = [
    value
    for name, value in vars(qwen35).items()
    if not name.startswith("_") and name.endswith("_config") and inspect.isfunction(value)
    if value.__module__ == qwen35.__name__ and value is not _DYNAMIC_CP_RECIPE
]


@dataclass
class _DynamicCPTransformerConfig(TransformerConfig):
    """Transformer config with the dynamic context parallel fields of the Megatron-Core dev pin."""

    dynamic_context_parallel: bool = False
    min_dynamic_context_parallel_size: int = 1


@dataclass(kw_only=True)
class _DynamicCPModelConfig(BridgeHybridModelConfig):
    transformer_config_class: ClassVar[type[TransformerConfig]] = _DynamicCPTransformerConfig


def _reject_provider_api(*args, **kwargs):
    raise AssertionError("Qwen3.5 text recipes must use get_model_config(), not the legacy provider API")


@pytest.fixture
def builder_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real bridges on local HF configs and reject the provider API."""
    patch_recipe_construction_dependencies(monkeypatch)
    monkeypatch.setattr(qwen35, "AutoBridge", AutoBridge)
    monkeypatch.setattr(qwen35.AutoConfig, "from_pretrained", lambda path, **kwargs: _HF_CONFIGS[path]())
    monkeypatch.setattr(AutoBridge, "to_megatron_provider", _reject_provider_api)


def test_recipes_are_discovered():
    assert _RECIPES


@pytest.mark.parametrize("recipe_func", _RECIPES, ids=lambda func: func.__name__)
def test_each_qwen35_text_recipe_uses_strict_builder_config(recipe_func: Callable, builder_only: None):
    """Recipes only set fields that the builder-backed model config declares."""
    cfg = recipe_func()

    assert isinstance(cfg.model, BridgeHybridModelConfig)
    assert cfg.model.experimental_attention_variant == "gdn"
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.share_embeddings_and_output_weights is False


def test_dynamic_cp_recipe_uses_strict_builder_config(builder_only: None, monkeypatch: pytest.MonkeyPatch):
    """With the dev pin's dynamic CP fields declared, the recipe sets only declared fields."""
    monkeypatch.setattr(Qwen35MoEBridge, "MODEL_CONFIG_CLASS", _DynamicCPModelConfig)

    cfg = _DYNAMIC_CP_RECIPE()

    assert isinstance(cfg.model, BridgeHybridModelConfig)
    assert cfg.model.dynamic_context_parallel is True
    assert cfg.model.min_dynamic_context_parallel_size == 1
    assert cfg.model.num_moe_experts == 256


def test_dynamic_cp_recipe_rejects_model_config_without_dynamic_cp(builder_only: None):
    """A pin without dynamic CP fields fails the recipe instead of dropping the setting."""
    if "dynamic_context_parallel" in {field.name for field in fields(TransformerConfig)}:
        pytest.skip("The pinned Megatron-Core declares dynamic context parallelism.")

    with pytest.raises(AttributeError, match="dynamic_context_parallel"):
        _DYNAMIC_CP_RECIPE()
