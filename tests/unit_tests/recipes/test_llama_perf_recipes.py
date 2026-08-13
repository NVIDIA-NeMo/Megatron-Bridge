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
import importlib.util
import pkgutil
from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.llama.gb200.llama3 import (
    llama3_8b_pretrain_8gpu_gb200_nvfp4_config,
    llama3_70b_pretrain_64gpu_gb200_nvfp4_config,
)
from megatron.bridge.perf_recipes.llama.h100.llama3 import (
    llama3_70b_pretrain_64gpu_h100_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import (
    patch_recipe_construction_dependencies,
    patch_recipe_module_global,
)


def _finetune_perf_recipes() -> list[Callable[[], ConfigContainer]]:
    recipes = []
    llama_package = importlib.import_module("megatron.bridge.perf_recipes.llama")
    for module_info in pkgutil.iter_modules(llama_package.__path__):
        module_name = f"{llama_package.__name__}.{module_info.name}.llama3"
        if not module_info.ispkg or importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        recipes.extend(
            recipe
            for name in dir(module)
            if ("_sft_" in name or "_lora_" in name)
            and name.endswith("_config")
            and callable(recipe := getattr(module, name))
            and recipe.__module__ == module_name
        )
    return recipes


class _FakeModelCfg:
    cross_entropy_fusion_impl = "te"
    context_parallel_size = 1
    use_te_rng_tracker = False

    def finalize(self) -> None:
        return None


class _FakeBridge:
    @staticmethod
    def from_hf_pretrained(*args, **kwargs) -> "_FakeBridge":
        return _FakeBridge()

    def get_model_config(self) -> _FakeModelCfg:
        return _FakeModelCfg()

    def to_megatron_provider(self, load_weights: bool = False) -> _FakeModelCfg:
        raise AssertionError("Llama recipes must use get_model_config(), not the legacy provider API")


@pytest.mark.unit
@pytest.mark.parametrize("recipe_func", _finetune_perf_recipes(), ids=lambda recipe: recipe.__name__)
def test_llama3_finetune_perf_recipes_use_offline_packing_specs(
    recipe_func: Callable[[], ConfigContainer], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_recipes = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    patch_recipe_module_global(monkeypatch, base_recipes, "AutoBridge", _FakeBridge)

    config = recipe_func()

    assert config.dataset.offline_packing_specs is not None
    assert config.dataset.enable_offline_packing is True
    assert config.dataset.offline_packing_specs.packed_sequence_size == config.dataset.seq_length
    assert not hasattr(config.dataset, "packed_sequence_specs")
    assert config.dataset.dataset_kwargs["pad_to_max_length"] is True


@pytest.mark.unit
def test_llama3_70b_h100_fp8cs_sets_explicit_pipeline_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 64-GPU H100 FP8-CS recipe pins an explicit pipeline layout.

    The transformer layers do not split evenly across PP8/VP5 once the embedding and loss stages
    are placed, so the default split leaves the first and last stages heavier than the middle
    ones. Pipeline throughput is set by the slowest stage, so the layout is pinned rather than
    derived.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    cfg = llama3_70b_pretrain_64gpu_h100_fp8cs_config()

    assert cfg.model.pipeline_model_parallel_layout == "Et*2|(t*2|)*34,t*3|(t*2|)*3,tL"

    # The layout is only valid for this parallelism; pin it alongside.
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.model.virtual_pipeline_model_parallel_size == 5
    assert cfg.model.context_parallel_size == 1
    assert cfg.train.global_batch_size == 256
    assert cfg.train.micro_batch_size == 1


@pytest.mark.unit
def test_llama3_70b_gb200_nvfp4_captures_whole_transformer_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 64-GPU GB200 NVFP4 recipe captures the whole Transformer layer per graph.

    Megatron-Core expresses whole-layer coverage as an empty ``cuda_graph_modules``; the field's
    ``"full"`` default normalizes to the same empty list, and ``"full"`` itself is deprecated.
    ``clear_cuda_graph_modules()`` also clears the deprecated ``cuda_graph_scope`` so the config
    stays off the conversion path in ``TransformerConfig.__post_init__``.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    cfg = llama3_70b_pretrain_64gpu_gb200_nvfp4_config()

    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_modules == []
    assert cfg.model.cuda_graph_scope is None

    # TE RNG trackers are derived from graphs being active, not assigned by the recipe.
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True

    # Parallelism and batch are unchanged by the capture-scope setting.
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 5
    assert cfg.train.global_batch_size == 256


@pytest.mark.unit
def test_llama3_8b_gb200_nvfp4_runs_dpa_in_fp8_current_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 8-GPU GB200 NVFP4 recipe runs dot-product attention in FP8 current scaling.

    NVFP4BlockScaling takes ``fp8_dpa`` from ``fp8_dot_product_attention`` independently of
    ``config.fp8``, so the flag is live under the FP4 recipe rather than inert.
    ``NVTE_DPA_FP8_RECIPE`` selects the FP8 recipe TE uses for that attention path, and must be
    part of the inline ``env_vars`` dict so the recipe-environment test sees it.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    cfg = llama3_8b_pretrain_8gpu_gb200_nvfp4_config()

    assert cfg.mixed_precision.fp8_dot_product_attention is True
    assert cfg.mixed_precision.fp4 == "e2m1"
    assert cfg.mixed_precision.fp4_recipe == "nvfp4"
    assert cfg.env_vars["NVTE_DPA_FP8_RECIPE"] == "Float8CurrentScaling"

    # Parallelism and batch are unchanged by this setting.
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.train.global_batch_size == 128
    assert cfg.train.micro_batch_size == 4
