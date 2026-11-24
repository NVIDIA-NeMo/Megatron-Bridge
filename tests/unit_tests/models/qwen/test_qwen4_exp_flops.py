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

"""Hand-counted language/vision FLOPs and training-dispatch coverage."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.qwen.modeling_qwen4_exp.model import Qwen4ExpHybridModel
from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpModelConfig
from megatron.bridge.models.qwen.modeling_qwen4_exp.vl_model import Qwen4ExpVLModel, Qwen4ExpVLModelConfig
from megatron.bridge.training.utils.flop_utils import (
    num_floating_point_operations,
    vit_flops,
    vit_flops_from_grid_thw,
    vit_flops_from_patch_stats,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def model_config():
    # One GDN + one QSA block, both MoE; one PLE, two residual streams.
    config = SimpleNamespace(
        num_layers=2,
        hidden_size=8,
        seq_length=8,
        linear_attention_freq=[1, 0],
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=4,
        attention_output_gate=True,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_num_value_heads=1,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        num_moe_experts=4,
        moe_layer_freq=1,
        moe_ffn_hidden_size=4,
        ffn_hidden_size=16,
        moe_latent_size=None,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=4,
        gated_linear_unit=True,
        mhc_num_residual_streams=2,
        mhc_gated_residual_rank=3,
        ple_layer_ids=[2],
        ple_embed_dim=4,
        vocab_size=16,
        should_pad_vocab=False,
        ple_freeze_ngram_embeddings=True,
        qsa_indexer_n_heads=2,
        qsa_indexer_kv_heads=1,
        qsa_indexer_head_dim=4,
        qsa_indexer_budget=8,
        qsa_indexer_compress_ratio=4,
        qsa_force_sparse=False,
    )
    config.transformer = config
    config.estimate_flops = lambda **kwargs: Qwen4ExpModelConfig.estimate_flops(config, **kwargs)
    return config


@pytest.mark.parametrize("backward", [False, True])
def test_language_hand_count_and_model_dispatch(model_config, backward):
    # Per-token: QSA projections512 + GDN528 + MoE1152 + GR/PLE1408 + logits256.
    # Runtime subsequences3+5 have sum squares34, rather than8**2.
    expected = (3856 * 8 + 16 * 34) * (3 if backward else 1)
    assert Qwen4ExpHybridModel.estimate_flops(
        model_config,
        total_real_tokens_in_batch=8,
        seqlen_squared_sum_in_batch=34,
    ) == expected / (3 if backward else 1)
    if backward:
        cfg = SimpleNamespace(model=model_config, peft=None)
        assert num_floating_point_operations(cfg, batch_size=2, seqlen_sum=8, seqlen_squared_sum=34) == expected
    assert model_config.estimate_flops(batch_size=0) == 0


def test_detached_indexer_is_forward_only_and_frozen_lookup_keeps_projections(model_config):
    model_config.seq_length = 12
    with pytest.warns(UserWarning, match="dense-attention surrogate"):
        forward = model_config.estimate_flops(batch_size=1, include_backward=False)
    with pytest.warns(UserWarning, match="dense-attention surrogate"):
        training = model_config.estimate_flops(batch_size=1)
    assert forward == 3856 * 12 + 16 * 144 + 192 * 12
    assert training == 3 * (forward - 192 * 12) + 192 * 12
    model_config.ple_freeze_ngram_embeddings = False
    with pytest.warns(UserWarning):
        assert model_config.estimate_flops(batch_size=1) == training


def test_vision_composition_and_global_accounting(model_config):
    model_config.vision_config = dict(
        depth=2, hidden_size=4, intermediate_size=8, spatial_merge_size=2, out_hidden_size=8
    )
    model_config.estimate_vision_flops = lambda **kwargs: Qwen4ExpVLModelConfig.estimate_vision_flops(
        model_config, **kwargs
    )
    # Two images with4/8patches: transformer2*(256*12+16*80), merger2*3*16*(16+8).
    vision = 3 * (2 * (256 * 12 + 16 * 80) + 2 * 3 * 16 * 24)
    stats = dict(patch_sum=12, patch_squared_sum=80, merged_token_sum=3)
    cfg = SimpleNamespace(model=model_config, peft=None)
    assert vit_flops_from_patch_stats(cfg, **stats) == vision
    language = model_config.estimate_flops(batch_size=1)
    language_stats = dict(total_real_tokens_in_batch=8, seqlen_squared_sum_in_batch=64)
    assert Qwen4ExpVLModel.estimate_flops(model_config, **language_stats, **stats) == (language + vision) / 3
    assert Qwen4ExpVLModel.estimate_flops(model_config, **language_stats) == language / 3


@pytest.mark.parametrize("merge_size", [2, 4])
def test_mapping_vision_config_merge_size(model_config, merge_size):
    model_config.vision_config = {"spatial_merge_size": merge_size}
    model_config.estimate_vision_flops = lambda **stats: stats["merged_token_sum"]
    cfg = SimpleNamespace(model=model_config, peft=None)
    expected = 32 // merge_size**2
    assert vit_flops_from_grid_thw(cfg, torch.tensor([[1, 4, 8]])) == expected
    assert vit_flops(cfg, batch_size=1, num_patches=32) == expected
    assert num_floating_point_operations(cfg, batch_size=1, num_vision_patches=32) == (
        model_config.estimate_flops(batch_size=1) + expected
    )
