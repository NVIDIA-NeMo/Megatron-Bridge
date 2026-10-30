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

import importlib
import inspect

import pytest
from transformers import ExaoneMoeConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_config import ExaoneMoeModelConfig
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_HF_PATH = "LGAI-EXAONE/K-EXAONE-2.0-750B-A37B"


def _k_exaone_hf_config() -> ExaoneMoeConfig:
    """A small K-EXAONE architecture: dense-first MoE, hybrid attention, one MTP layer."""
    return ExaoneMoeConfig(
        architectures=["ExaoneMoeForCausalLM"],
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=768,
        moe_intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        num_shared_experts=1,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=128,
        vocab_size=1024,
        max_position_embeddings=4096,
        num_nextn_predict_layers=1,
        dtype="bfloat16",
    )


class _BuilderOnlyBridge:
    """Return a real strict ModelConfig while rejecting the provider API."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs) -> "_BuilderOnlyBridge":
        return _BuilderOnlyBridge()

    def get_model_config(self) -> ExaoneMoeModelConfig:
        return AutoBridge.from_hf_config(_k_exaone_hf_config()).get_model_config()

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("K-EXAONE recipes must use get_model_config(), not the legacy provider API")


@pytest.fixture(autouse=True)
def _patch_recipe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")
    patch_recipe_construction_dependencies(monkeypatch)


def test_k_exaone_2_0_pipeline_layout_balances_decoder_and_mtp_layers() -> None:
    recipe_module = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")

    layout = recipe_module.get_k_exaone_2_0_pipeline_layout(16, None)

    assert layout is not None
    assert len(layout) == 16
    assert sum(stage.count("decoder") for stage in layout) == 78
    assert sum(stage.count("mtp") for stage in layout) == 4
    assert layout[0].count("embedding") == 1
    assert layout[-1].count("loss") == 1
    assert layout[-1].count("mtp") == 4


def test_k_exaone_2_0_pipeline_layout_builder_is_public_for_checkpoint_reload() -> None:
    recipe_module = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")

    builder = recipe_module.get_k_exaone_2_0_pipeline_layout

    assert not builder.__name__.startswith("_")


@pytest.mark.parametrize(
    ("recipe_name", "tp", "pp", "ep", "expected_gpus"),
    [
        ("exaone_moe_2_0_750b_a37_pretrain_512gpu_h100_bf16_config", 4, 16, 32, 512),
        ("exaone_moe_2_0_750b_a37_sft_512gpu_h100_bf16_config", 4, 16, 32, 512),
        ("exaone_moe_2_0_750b_a37_peft_128gpu_h100_bf16_config", 4, 8, 16, 128),
    ],
)
def test_k_exaone_2_0_recipe_topology(
    recipe_name: str,
    tp: int,
    pp: int,
    ep: int,
    expected_gpus: int,
) -> None:
    recipe_module = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")
    recipe = getattr(recipe_module, recipe_name)()

    assert recipe.tokenizer.tokenizer_model == _HF_PATH
    assert recipe.model.tensor_model_parallel_size == tp
    assert recipe.model.pipeline_model_parallel_size == pp
    assert recipe.model.expert_model_parallel_size == ep
    assert recipe.model.sequence_parallel is True
    assert pp * max(tp, ep) == expected_gpus
    assert sum(stage.count("decoder") for stage in recipe.model.pipeline_model_parallel_layout) == 78
    assert sum(stage.count("mtp") for stage in recipe.model.pipeline_model_parallel_layout) == 4


def test_k_exaone_2_0_public_aliases() -> None:
    recipe_module = importlib.import_module("megatron.bridge.recipes.exaone")
    h100_module = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")

    assert (
        recipe_module.exaone_moe_2_0_750b_a37_pretrain_config
        is h100_module.exaone_moe_2_0_750b_a37_pretrain_512gpu_h100_bf16_config
    )
    assert (
        recipe_module.exaone_moe_2_0_750b_a37_sft_config
        is h100_module.exaone_moe_2_0_750b_a37_sft_512gpu_h100_bf16_config
    )
    assert (
        recipe_module.exaone_moe_2_0_750b_a37_peft_config
        is h100_module.exaone_moe_2_0_750b_a37_peft_128gpu_h100_bf16_config
    )


@pytest.mark.parametrize(
    ("alias_name", "target_name"),
    [
        (
            "exaone_moe_pretrain_config",
            "exaone_moe_2_0_750b_a37_pretrain_512gpu_h100_bf16_config",
        ),
        (
            "exaone_moe_sft_config",
            "exaone_moe_2_0_750b_a37_sft_512gpu_h100_bf16_config",
        ),
        (
            "exaone_moe_peft_config",
            "exaone_moe_2_0_750b_a37_peft_128gpu_h100_bf16_config",
        ),
    ],
)
def test_k_exaone_2_0_short_aliases_resolve(alias_name: str, target_name: str) -> None:
    recipes = importlib.import_module("megatron.bridge.recipes")
    h100_module = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")

    assert getattr(recipes, alias_name) is getattr(h100_module, target_name)


_EXAONE_MOE_RECIPE_MODULE = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")
_EXAONE_MOE_RECIPE_NAMES = sorted(
    name
    for name, value in vars(_EXAONE_MOE_RECIPE_MODULE).items()
    if not name.startswith("_") and name.endswith("_config") and inspect.isfunction(value)
    if value.__module__ == _EXAONE_MOE_RECIPE_MODULE.__name__
)


@pytest.mark.parametrize("recipe_name", _EXAONE_MOE_RECIPE_NAMES)
def test_each_k_exaone_recipe_uses_strict_builder_config(recipe_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recipes only set fields that the builder-backed model config declares."""
    recipe_module = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone_moe")
    monkeypatch.setattr(recipe_module, "AutoBridge", _BuilderOnlyBridge)

    cfg = getattr(recipe_module, recipe_name)()

    assert isinstance(cfg.model, ExaoneMoeModelConfig)
    assert cfg.model.qk_layernorm is True
    assert cfg.model.num_moe_experts == 8
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.sequence_parallel is True
    if "2_0" in recipe_name:
        assert cfg.model._pipeline_model_parallel_layout_builder is recipe_module.get_k_exaone_2_0_pipeline_layout
        assert cfg.model.pipeline_model_parallel_layout is not None
