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

import pytest

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_HF_PATH = "LGAI-EXAONE/K-EXAONE-2.0-750B-A37B"


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
