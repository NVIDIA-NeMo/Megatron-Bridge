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

from typing import Callable

import pytest
from transformers import Exaone4Config

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.recipes.exaone.h100 import exaone4
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


pytestmark = pytest.mark.unit

_EXAONE4_RECIPE_FUNCS = [getattr(exaone4, name) for name in exaone4.__all__]


class _BuilderOnlyBridge:
    """Return a real strict ModelConfig while rejecting the provider API."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs) -> "_BuilderOnlyBridge":
        return _BuilderOnlyBridge()

    def get_model_config(self) -> BridgeGPTModelConfig:
        config = Exaone4Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=65536,
            vocab_size=102400,
        )
        config.architectures = ["Exaone4ForCausalLM"]
        return AutoBridge.from_hf_config(config).get_model_config()

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("EXAONE 4.0 recipes must use get_model_config(), not the legacy provider API")


@pytest.mark.parametrize("recipe_func", _EXAONE4_RECIPE_FUNCS, ids=lambda func: func.__name__)
def test_each_exaone4_recipe_uses_strict_builder_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Every EXAONE 4.0 recipe should configure a ModelConfig without provider fallback."""
    patch_recipe_module_global(monkeypatch, recipe_func, "AutoBridge", _BuilderOnlyBridge)

    if "peft" in recipe_func.__name__:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()

    assert isinstance(cfg.model, BridgeGPTModelConfig)
    assert cfg.model.qk_layernorm is True
    assert cfg.model.normalization == "RMSNorm"
