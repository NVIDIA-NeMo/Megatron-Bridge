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
import torch.nn.functional as F
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from megatron.bridge.perf_recipes.glm_moe_dsa import (
    glm51_sft_192gpu_gb200_bf16_config,
    glm51_sft_416gpu_h100_bf16_config,
    glm52_50b_pretrain_8gpu_gb200_bf16_config,
    glm52_50b_pretrain_8gpu_gb200_fp8mx_config,
    glm52_sft_192gpu_gb200_bf16_config,
    glm52_sft_416gpu_h100_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.theoretical_memory_utils import estimate_training_memory


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
    def from_hf_pretrained(cls, model_id: str, revision: str | None = None) -> "_FakeAutoBridge":
        if model_id.endswith("GLM-5.2") and revision is not None:
            assert len(revision) == 40
        return cls(model_id)

    def to_megatron_provider(self, load_weights: bool = False) -> SimpleNamespace:
        del load_weights
        is_glm52 = self.model_id.endswith("GLM-5.2")
        return SimpleNamespace(
            model_id=self.model_id,
            num_layers=78,
            hidden_size=6144,
            ffn_hidden_size=12288,
            num_attention_heads=64,
            num_query_groups=64,
            kv_channels=192,
            qk_pos_emb_head_dim=64,
            num_moe_experts=256,
            moe_ffn_hidden_size=2048,
            moe_shared_expert_intermediate_size=2048,
            moe_layer_freq=[0, 0, 0] + [1] * 75,
            moe_router_topk=8,
            gated_linear_unit=True,
            activation_func=F.silu,
            share_embeddings_and_output_weights=False,
            vocab_size=154880,
            make_vocab_size_divisible_by=1280,
            should_pad_vocab=True,
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


@pytest.mark.parametrize(
    ("recipe_func", "fp8_recipe"),
    [
        (glm52_50b_pretrain_8gpu_gb200_bf16_config, None),
        (glm52_50b_pretrain_8gpu_gb200_fp8mx_config, "mxfp8"),
    ],
    ids=["bf16", "fp8mx"],
)
def test_glm52_50b_proxy_pretrain_recipes(
    recipe_func: Callable[[], ConfigContainer],
    fp8_recipe: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GB200 proxy preserves GLM-5.2 dimensions while scaling depth and expert count."""
    cfg = _build_recipe(recipe_func, monkeypatch)

    assert cfg.dataset.blend is None
    assert cfg.dataset.seq_length == 8192
    assert cfg.model.seq_length == 8192
    assert cfg.model.num_layers == 18
    assert cfg.model.moe_layer_freq == [0, 0, 0] + [1] * 15
    assert cfg.model.num_moe_experts == 64
    assert cfg.model.moe_router_topk == 8
    assert cfg.model.dsa_indexer_topk == 512
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.hidden_size == 6144
    assert cfg.model.qk_pos_emb_head_dim == 64
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.train.global_batch_size == 64
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_hybridep_num_sms == 16
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 128
    assert cfg.dist.distributed_timeout_minutes == 30
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.average_in_collective is False
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.optimizer.lr == 3e-5
    assert cfg.optimizer.min_lr == 3e-5
    assert cfg.scheduler.lr_warmup_iters == 0
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.comm_overlap.overlap_grad_reduce is False
    assert cfg.mixed_precision.fp8_recipe == (fp8_recipe or "tensorwise")
    if fp8_recipe is None:
        assert cfg.mixed_precision.fp8 is None
        assert cfg.model.use_transformer_engine_op_fuser is False
        assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag is False
    else:
        assert cfg.mixed_precision.fp8 == "e4m3"
        assert cfg.model.use_transformer_engine_op_fuser is True
        assert cfg.model.moe_mlp_glu_interleave_size == 32
        assert cfg.model.fp8_output_proj is True
        assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag is True
        assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
        assert cfg.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
        assert cfg.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1

    parameter_count = estimate_training_memory(cfg, include_activation=False).total_parameters
    assert 45e9 < parameter_count < 55e9


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
