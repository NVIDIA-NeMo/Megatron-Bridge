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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge
from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3MoELayer
from megatron.bridge.models.kimi.kimi_k3_pipeline import (
    bank_num_rows,
    pack_stage_boundary,
    unpack_stage_boundary,
)
from megatron.bridge.models.kimi.kimi_k3_provider import KimiK3ModelProvider


@pytest.fixture
def kimi_k3_text_config() -> SimpleNamespace:
    """Return the official K3 architecture truncated to four language layers."""
    return SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        attn_res_block_size=12,
        first_k_dense_replace=1,
        head_dim=256,
        hidden_act="situ",
        hidden_dropout=0.0,
        hidden_size=7168,
        initializer_range=0.006,
        intermediate_size=33792,
        kv_lora_rank=512,
        latent_moe_use_norm=True,
        linear_attn_config={
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "kda_layers": [1, 2, 3],
            "num_heads": 96,
            "short_conv_kernel_size": 4,
        },
        max_position_embeddings=1048576,
        moe_intermediate_size=3072,
        moe_layer_freq=1,
        moe_router_activation_func="sigmoid",
        num_attention_heads=96,
        num_expert_group=1,
        num_experts=896,
        num_experts_per_token=16,
        num_hidden_layers=4,
        num_key_value_heads=96,
        num_shared_experts=2,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-5,
        routed_expert_hidden_size=3584,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
        topk_group=1,
        torch_dtype="bfloat16",
        use_grouped_topk=True,
        v_head_dim=128,
        vocab_size=163840,
    )


@pytest.fixture
def kimi_k3_pretrained(kimi_k3_text_config: SimpleNamespace) -> Mock:
    """Return a config-only K3 wrapper."""
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        model_type="kimi_k3",
        text_config=kimi_k3_text_config,
        torch_dtype="bfloat16",
    )
    return pretrained


def test_provider_bridge_configures_four_layer_proxy(kimi_k3_pretrained: Mock) -> None:
    """The provider preserves K3's heterogeneous attention and latent-MoE layout."""
    provider = KimiK3Bridge().provider_bridge(kimi_k3_pretrained)

    assert isinstance(provider, KimiK3ModelProvider)
    assert provider.num_layers == 4
    assert provider.position_embedding_type == "none"
    assert provider.kimi_kda_layers == (1, 2, 3)
    assert provider.moe_layer_freq == [0, 1, 1, 1]
    assert provider.moe_latent_size == 3584
    assert provider.moe_shared_expert_intermediate_size == 6144
    assert provider.moe_router_topk == 16
    assert provider.moe_router_num_groups == 1
    assert provider.moe_router_group_topk == 1
    assert provider.use_te_activation_func is True
    assert provider.bf16 is True
    assert provider.params_dtype == torch.bfloat16


def test_mapping_registry_covers_kda_latent_moe_and_attn_res(kimi_k3_pretrained: Mock) -> None:
    """Custom K3 weights resolve in both conversion directions."""
    bridge = KimiK3Bridge()
    bridge.provider_bridge(kimi_k3_pretrained)
    registry = bridge.mapping_registry()

    cases = {
        "decoder.layers.0.self_attention.q_conv1d.weight": ("language_model.model.layers.0.self_attn.q_conv1d.weight"),
        "decoder.layers.2.mlp.routed_expert_norm.weight": (
            "language_model.model.layers.2.block_sparse_moe.routed_expert_norm.weight"
        ),
        "decoder.layers.3.output_attn_res_proj.weight": "language_model.model.output_attn_res_proj.weight",
    }
    for megatron_name, hf_name in cases.items():
        mapping = registry.megatron_to_hf_lookup(megatron_name)
        assert mapping is not None
        assert mapping.hf_param == hf_name
        reverse = registry.hf_to_megatron_lookup(hf_name)
        assert reverse is not None
        assert reverse.megatron_param == megatron_name


def test_stage_boundary_pack_unpack_and_bank_schedule() -> None:
    """AttnRes state survives pipeline packing without mixing rows."""
    prefix_sum = torch.randn(5, 2, 16, dtype=torch.bfloat16)
    block_residual = torch.randn(5, 2, 3, 16, dtype=torch.bfloat16)

    packed = pack_stage_boundary(prefix_sum, block_residual)
    prefix_out, bank_out = unpack_stage_boundary(packed, hidden_size=16, num_rows=3)

    torch.testing.assert_close(prefix_out, prefix_sum, rtol=0, atol=0)
    torch.testing.assert_close(bank_out, block_residual, rtol=0, atol=0)
    assert [bank_num_rows(layer_idx, 12) for layer_idx in (1, 12, 13, 24, 25)] == [1, 1, 2, 2, 3]

    with pytest.raises(ValueError, match="stage-boundary payload width"):
        unpack_stage_boundary(torch.zeros(2, 1, 3 * 16), hidden_size=16, num_rows=3)


def test_latent_moe_normalizes_after_combine_and_before_up_projection() -> None:
    """K3's routed-expert norm is applied to the weighted expert output."""

    class _Dispatcher:
        @staticmethod
        def combine_postprocess(output: torch.Tensor) -> torch.Tensor:
            return output + 1

    layer = SimpleNamespace(
        token_dispatcher=_Dispatcher(),
        routed_expert_norm=lambda output: output * 2,
        fc2_latent_proj=lambda output: (output + 3, None),
        _latent_shared_expert_output=None,
    )
    routed_output = torch.tensor([1.0])
    shared_output = torch.tensor([5.0])

    output = KimiK3MoELayer.postprocess(layer, routed_output, shared_output)

    assert torch.equal(output, torch.tensor([12.0]))
