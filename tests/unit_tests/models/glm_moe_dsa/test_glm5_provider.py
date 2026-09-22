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

"""Tests of GLM5ModelProvider's hybrid layer pattern derivation and model construction."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from megatron.core.transformer.experimental_attention_variant.dsa import source_dsa_compute_layer
from transformers import GlmMoeDsaConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.glm_moe_dsa.glm5_provider import GLM5ModelProvider, glm_hybrid_pattern, split_glm_pattern


pytestmark = pytest.mark.unit


def _provider(**overrides):
    config = GlmMoeDsaConfig(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=6,
        first_k_dense_replace=2,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        intermediate_size=256,
        moe_intermediate_size=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        indexer_rope_interleave=True,
        index_topk_freq=2,
        index_skip_topk_offset=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
    )
    provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
    for key, value in overrides.items():
        assert hasattr(provider, key)
        setattr(provider, key, value)
    return provider


def test_provider_preserves_mla_and_derives_physical_layers():
    provider = _provider(moe_token_dispatcher_type="alltoall")
    assert isinstance(provider, GLM5ModelProvider)
    assert provider.hybrid_layer_pattern == "D-D-DEDEDEDE"
    provider.finalize()
    assert provider.num_layers == 12
    assert provider.qk_head_dim == 16
    assert provider.qk_pos_emb_head_dim == 16
    assert provider.qk_layernorm
    assert provider.moe_layer_freq == [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]


def test_mtp_finalize_is_idempotent():
    provider = _provider(mtp_num_layers=1, moe_token_dispatcher_type="alltoall")
    provider.finalize()
    provider.finalize()
    assert provider.hybrid_layer_pattern.endswith("/DE")
    assert provider.hybrid_layer_pattern.count("/") == 1
    assert provider.num_layers == 12
    provider.mtp_num_layers = None
    provider.finalize()
    assert "/" not in provider.hybrid_layer_pattern


def test_pipeline_boundary_must_start_a_compute_group():
    provider = _provider(pipeline_model_parallel_size=2)
    provider.hybrid_layer_pattern = "D-D-DE|DEDEDE"
    with pytest.raises(ValueError, match="reuse layer"):
        provider.finalize()
    provider.hybrid_layer_pattern = split_glm_pattern(provider.hybrid_layer_pattern, [4, 2])
    provider.finalize()
    assert [len(segment) for segment in provider.hybrid_layer_pattern.split("|")] == [8, 4]


def test_invalid_patterns_fail_before_construction():
    with pytest.raises(ValueError):
        glm_hybrid_pattern(num_layers=2, first_k_dense_replace=3)
    with pytest.raises(ValueError):
        split_glm_pattern("D-DE", [1])


def test_automatic_pipeline_partition_keeps_sharing_groups_whole():
    provider = _provider(pipeline_model_parallel_size=2, moe_token_dispatcher_type="alltoall")
    provider.finalize()
    offset = 0
    for segment in provider.hybrid_layer_pattern.split("/")[0].split("|"):
        count = segment.count("D")
        for logical in range(offset + 1, offset + count + 1):
            assert source_dsa_compute_layer(logical, 1, 2) in range(offset + 1, offset + count + 1)
        offset += count
    assert offset == 6


def test_selective_attention_recompute_rejects_lost_forward_state():
    provider = _provider(recompute_granularity="selective", recompute_modules=["core_attn"])
    with pytest.raises(ValueError, match="forward-local DSA state"):
        provider.finalize()


def test_provider_constructs_native_hybrid_model():
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    from megatron.bridge.models.hybrid import hybrid_provider

    provider = _provider()
    provider._pg_collection = SimpleNamespace(pp=object())
    assert hybrid_provider.MCoreHybridModel is HybridModel
    with patch.object(hybrid_provider, "MCoreHybridModel", autospec=True) as constructor:
        result = provider.provide(pre_process=True, post_process=True)
    assert result is constructor.return_value
    assert constructor.call_args.kwargs["config"] is provider
    assert constructor.call_args.kwargs["hybrid_stack_spec"].module is not HybridStack


@pytest.mark.parametrize("tp,expected", [(1, False), (2, True)])
def test_tensor_parallel_enables_sequence_parallel_for_absorbed_mla(tp, expected):
    provider = _provider(tensor_model_parallel_size=tp, moe_token_dispatcher_type="alltoall")
    provider.finalize()
    assert provider.sequence_parallel is expected
