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
import torch
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from megatron.bridge.perf_recipes.glm_moe_dsa import (
    glm51_sft_192gpu_gb200_bf16_config,
    glm51_sft_416gpu_h100_bf16_config,
    glm52_pretrain_192gpu_gb200_fp8mx_config,
    glm52_sft_192gpu_gb200_bf16_config,
    glm52_sft_416gpu_h100_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit

_RECIPES = [
    glm51_sft_192gpu_gb200_bf16_config,
    glm52_pretrain_192gpu_gb200_fp8mx_config,
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
    def from_hf_pretrained(cls, model_id: str, revision: str | None = None) -> "_FakeAutoBridge":
        if revision is not None:
            assert len(revision) == 40
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


def _dsa_source_layer_id(layer_id: int, *, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the zero-based source layer defined by MCore's DSA sharing contract."""
    # Mirrors the private MCore `_validate_dsa_index_share_pipeline_split`
    # helper that guarded this recipe before removal from the pinned revision.
    # MCore defines DSA sharing with one-based layers: layers through
    # max(skip_topk_offset, 1) compute their own indices, then each topk_freq
    # group reuses the indices computed by its first layer.
    layer_number = layer_id + 1
    sharing_offset = max(skip_topk_offset, 1)
    if layer_number <= sharing_offset:
        return layer_id
    return layer_number - ((layer_number - sharing_offset) % topk_freq) - 1


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
    perf_demo_dsa_overrides = {
        "dsa_indexer_topk": 2048,
        "dsa_indexer_loss_coeff": 0.0,
        "dsa_indexer_use_sparse_loss": False,
    }
    for field, expected in _BRIDGE_DSA_VALUES.items():
        if recipe_func is glm52_pretrain_192gpu_gb200_fp8mx_config:
            expected = perf_demo_dsa_overrides.get(field, expected)
        assert getattr(cfg.model, field) == expected


def test_glm52_gb200_pretrain_perf_demo_matches_measured_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full-model perf demo preserves the measured 192-GPU configuration."""
    cfg = _build_recipe(glm52_pretrain_192gpu_gb200_fp8mx_config, monkeypatch)

    assert cfg.model.num_layers == 78
    assert cfg.dataset.seq_length == 4096
    assert cfg.model.seq_length == 4096
    assert cfg.model.qk_pos_emb_head_dim == 64
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 6
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.train.global_batch_size == 3072
    assert cfg.train.micro_batch_size == 1

    layout = PipelineParallelLayerLayout(
        cfg.model.pipeline_model_parallel_layout,
        cfg.model.pipeline_model_parallel_size,
    )
    layout.validate_layer_layout(cfg.model.num_layers, cfg.model.mtp_num_layers)
    decoder_counts = [
        sum(chunk.count(LayerType.decoder) for chunk in physical_stage) for physical_stage in layout.layout
    ]
    assert decoder_counts == [10, 12, 12, 12, 16, 16]

    assert cfg.mixed_precision.fp8 == "e4m3"
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_hybridep_num_sms == 32
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 32
    assert cfg.model.moe_permute_fusion_into_hybridep is False
    assert cfg.model.dsa_indexer_topk == 2048
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False

    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.high_priority_a2a_comm_stream is False
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.comm_overlap.overlap_p2p_comm is False
    assert cfg.comm_overlap.batch_p2p_comm is True
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32


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

    decoder_offset = 0
    for vpp_rank in range(parsed_layout.virtual_pipeline_model_parallel_size):
        for pp_rank in range(parsed_layout.pipeline_model_parallel_size):
            stage = parsed_layout.layout[pp_rank][vpp_rank]
            decoder_count = stage.count(LayerType.decoder)
            if decoder_count:
                local_layer_ids = range(decoder_offset, decoder_offset + decoder_count)
                local_layer_id_set = set(local_layer_ids)
                for layer_id in local_layer_ids:
                    source_layer_id = _dsa_source_layer_id(
                        layer_id,
                        skip_topk_offset=cfg.model.dsa_indexer_skip_topk_offset,
                        topk_freq=cfg.model.dsa_indexer_topk_freq,
                    )
                    assert source_layer_id in local_layer_id_set
                decoder_offset += decoder_count

    assert decoder_offset == cfg.model.num_layers
