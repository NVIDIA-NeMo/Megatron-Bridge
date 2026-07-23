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

"""Unit tests for GLM-5 flat performance recipes."""

import importlib
import inspect
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from megatron.bridge.perf_recipes.glm_moe_dsa import (
    glm51_sft_192gpu_gb200_bf16_config,
    glm51_sft_416gpu_h100_bf16_config,
    glm52_sft_192gpu_gb200_bf16_config,
    glm52_sft_416gpu_h100_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit

_RECIPES = [
    glm51_sft_192gpu_gb200_bf16_config,
    glm52_sft_192gpu_gb200_bf16_config,
    glm51_sft_416gpu_h100_bf16_config,
    glm52_sft_416gpu_h100_bf16_config,
]

_H100_RECIPES = [
    glm51_sft_416gpu_h100_bf16_config,
    glm52_sft_416gpu_h100_bf16_config,
]

_BRIDGE_DSA_VALUES = {
    "dsa_indexer_n_heads": 7,
    "dsa_indexer_head_dim": 11,
    "dsa_indexer_topk": 13,
    "dsa_indexer_rope_interleaved": False,
    "dsa_indexer_rotate_activation": True,
    "dsa_indexer_k_norm_epsilon": 0.25,
    "dsa_indexer_loss_coeff": 0.5,
    "dsa_indexer_use_sparse_loss": False,
}


class _FakeAutoBridge:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @classmethod
    def from_hf_pretrained(cls, model_id: str) -> "_FakeAutoBridge":
        return cls(model_id)

    def to_megatron_provider(self, load_weights: bool = False) -> SimpleNamespace:
        del load_weights
        is_glm52 = self.model_id.endswith("GLM-5.2")
        return SimpleNamespace(
            model_id=self.model_id,
            num_layers=78,
            experimental_attention_variant="dsa",
            dsa_indexer_topk_freq=4 if is_glm52 else 1,
            dsa_indexer_skip_topk_offset=3 if is_glm52 else 0,
            use_transformer_engine_op_fuser=False,
            use_te_rng_tracker=False,
            **_BRIDGE_DSA_VALUES,
        )


def _build_recipe(recipe_func: Callable[[], ConfigContainer], monkeypatch: pytest.MonkeyPatch) -> ConfigContainer:
    recipe_module = importlib.import_module(recipe_func.__module__)
    monkeypatch.setattr(recipe_module, "AutoBridge", _FakeAutoBridge)
    return recipe_func()


@pytest.mark.parametrize("recipe_func", _RECIPES, ids=lambda recipe: recipe.__name__)
def test_glm5_perf_recipes_are_flat_and_preserve_bridge_dsa_fields(
    recipe_func: Callable[[], ConfigContainer], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recipes stay parameterless and only override performance-owned settings."""
    assert not inspect.signature(recipe_func).parameters

    cfg = _build_recipe(recipe_func, monkeypatch)

    assert cfg.model.moe_token_dispatcher_type == "flex"
    expected_dispatcher_backend = "hybridep" if "_gb200_" in recipe_func.__name__ else "deepep"
    assert cfg.model.moe_flex_dispatcher_backend == expected_dispatcher_backend
    assert cfg.model.dsa_kernel_backend == "cudnn"
    assert cfg.model.mtp_num_layers == 1
    if recipe_func.__name__.startswith("glm52_"):
        assert cfg.model.dsa_indexer_topk_freq == 4
        assert cfg.model.dsa_indexer_skip_topk_offset == 3
    else:
        assert cfg.model.dsa_indexer_topk_freq == 1
        assert cfg.model.dsa_indexer_skip_topk_offset == 0
    for field, expected in _BRIDGE_DSA_VALUES.items():
        assert getattr(cfg.model, field) == expected


@pytest.mark.parametrize("recipe_func", _H100_RECIPES, ids=lambda recipe: recipe.__name__)
def test_glm5_h100_parallel_topology(
    recipe_func: Callable[[], ConfigContainer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both 416-GPU H100 recipes use the TP1/PP13/VPP2/CP32 topology."""
    cfg = _build_recipe(recipe_func, monkeypatch)

    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 64
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 13
    assert cfg.model.virtual_pipeline_model_parallel_size == 2
    assert cfg.model.context_parallel_size == 32
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.sequence_parallel is False
    assert cfg.model.mtp_num_layers == 1
    assert (
        416
        // (
            cfg.model.tensor_model_parallel_size
            * cfg.model.pipeline_model_parallel_size
            * cfg.model.context_parallel_size
        )
        == 1
    )
    assert (
        416
        // (
            cfg.model.pipeline_model_parallel_size
            * cfg.model.expert_model_parallel_size
            * cfg.model.expert_tensor_parallel_size
        )
        == 1
    )


def test_glm51_h100_uses_balanced_default_pipeline_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """GLM-5.1 needs no custom layout because it does not share DSA indices."""
    cfg = _build_recipe(glm51_sft_416gpu_h100_bf16_config, monkeypatch)

    assert cfg.model.dsa_indexer_topk_freq == 1
    assert cfg.model.pipeline_model_parallel_layout is None
    assert (
        cfg.model.num_layers
        // cfg.model.pipeline_model_parallel_size
        // cfg.model.virtual_pipeline_model_parallel_size
        == 3
    )


def test_glm52_h100_pipeline_layout_keeps_dsa_index_sharing_within_each_vpp_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GLM-5.2 layout never shares DSA indices across PP/VPP chunks."""
    cfg = _build_recipe(glm52_sft_416gpu_h100_bf16_config, monkeypatch)

    layout = cfg.model.pipeline_model_parallel_layout
    parsed_layout = PipelineParallelLayerLayout(layout, cfg.model.pipeline_model_parallel_size)
    parsed_layout.validate_layer_layout(cfg.model.num_layers, cfg.model.mtp_num_layers)
    assert parsed_layout.virtual_pipeline_model_parallel_size == 2
    assert parsed_layout.layout[-1][-1][-2:] == [LayerType.mtp, LayerType.loss]

    decoder_count = sum(
        stage.count(LayerType.decoder)
        for pipeline_stage in parsed_layout.layout
        for stage in pipeline_stage
    )
    assert decoder_count == cfg.model.num_layers
