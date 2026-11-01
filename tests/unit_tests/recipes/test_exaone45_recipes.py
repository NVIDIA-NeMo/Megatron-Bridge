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
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config

from megatron.bridge import AutoBridge
from megatron.bridge.models.exaone.exaone45.exaone45_config import Exaone45ModelConfig
from megatron.bridge.recipes.exaone.h100 import exaone45
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


pytestmark = pytest.mark.unit

_EXAONE45_RECIPE_FUNCS = [getattr(exaone45, name) for name in exaone45.__all__]


class _BuilderOnlyBridge:
    """Return a real strict ModelConfig while rejecting the provider API."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs) -> "_BuilderOnlyBridge":
        return _BuilderOnlyBridge()

    def get_model_config(self) -> Exaone45ModelConfig:
        config = Exaone4_5_Config(
            text_config={
                "model_type": "exaone4",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "max_position_embeddings": 65536,
                "vocab_size": 1024,
                "sliding_window": 32,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "num_nextn_predict_layers": 1,
            },
            vision_config={"depth": 2, "hidden_size": 64, "num_heads": 8, "out_hidden_size": 64},
        )
        config.architectures = ["Exaone4_5_ForConditionalGeneration"]
        return AutoBridge.from_hf_config(config).get_model_config()

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("EXAONE 4.5 recipes must use get_model_config(), not the legacy provider API")


@pytest.mark.parametrize("recipe_func", _EXAONE45_RECIPE_FUNCS, ids=lambda func: func.__name__)
def test_each_exaone45_recipe_uses_strict_builder_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Every EXAONE 4.5 recipe should configure a ModelConfig without provider fallback."""
    patch_recipe_module_global(monkeypatch, recipe_func, "AutoBridge", _BuilderOnlyBridge)

    if "peft" in recipe_func.__name__:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()

    assert isinstance(cfg.model, Exaone45ModelConfig)
    assert cfg.model.seq_length == 4096
    assert cfg.model.language_max_sequence_length == 65536
    assert cfg.model.vision_config is not None
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.freeze_mtp_model is False
    assert cfg.model.transformer_impl == "transformer_engine"
    assert cfg.model.mtp_num_layers == 1
