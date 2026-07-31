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

from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge
from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3MoELayer, KimiK3TransformerLayer
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
        attn_res_block_size=12,
        first_k_dense_replace=1,
        head_dim=256,
        hidden_act="situ",
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
    assert provider.q_lora_rank == 1536
    assert provider.kv_lora_rank == 512
    assert provider.num_moe_experts == 896
    assert provider.moe_ffn_hidden_size == 3072
    assert provider.moe_latent_size == 3584
    assert provider.moe_shared_expert_intermediate_size == 6144
    assert provider.moe_router_topk == 16
    assert provider.moe_router_num_groups == 1
    assert provider.moe_router_group_topk == 1
    assert provider.activation_func is torch.nn.functional.silu
    assert provider.hidden_dropout == 0.0
    assert provider.attention_dropout == 0.0
    assert provider.make_vocab_size_divisible_by == 128
    assert provider.use_te_activation_func is True
    assert provider.variable_seq_lengths is True
    assert provider.bf16 is True
    assert provider.params_dtype == torch.bfloat16


def test_mapping_registry_covers_kda_latent_moe_and_attn_res(kimi_k3_pretrained: Mock) -> None:
    """Custom K3 weights resolve in both conversion directions."""
    bridge = KimiK3Bridge()
    bridge.provider_bridge(kimi_k3_pretrained)
    registry = bridge.mapping_registry()

    cases = {
        "decoder.layers.0.self_attention.q_conv1d.weight": (
            "language_model.model.layers.0.self_attn.q_conv1d.weight",
            ColumnParallelMapping,
        ),
        "decoder.layers.1.self_attention.o_proj.weight": (
            "language_model.model.layers.1.self_attn.o_proj.weight",
            RowParallelMapping,
        ),
        "decoder.layers.2.mlp.routed_expert_norm.weight": (
            "language_model.model.layers.2.block_sparse_moe.routed_expert_norm.weight",
            ReplicatedMapping,
        ),
        "decoder.layers.3.output_attn_res_proj.weight": (
            "language_model.model.output_attn_res_proj.weight",
            ReplicatedMapping,
        ),
    }
    for megatron_name, (hf_name, mapping_type) in cases.items():
        mapping = registry.megatron_to_hf_lookup(megatron_name)
        assert isinstance(mapping, mapping_type)
        assert mapping.hf_param == hf_name
        reverse = registry.hf_to_megatron_lookup(hf_name)
        assert reverse is not None
        assert reverse.megatron_param == megatron_name


def test_kda_a_log_import_drops_zero_padding(kimi_k3_pretrained: Mock) -> None:
    """K3 imports only the 96 active KDA heads from the padded HF tensor."""
    bridge = KimiK3Bridge()
    bridge.hf_config = kimi_k3_pretrained.config
    name = "language_model.model.layers.0.self_attn.A_log"
    active = torch.arange(96, dtype=torch.float32)
    source = torch.cat((active, torch.zeros(32)))

    result = bridge.maybe_modify_loaded_hf_weight(name, {name: source})

    torch.testing.assert_close(result, active, rtol=0, atol=0)


def test_kda_a_log_import_rejects_nonzero_padding(kimi_k3_pretrained: Mock) -> None:
    """K3 fails explicitly if a future HF checkpoint changes the padding contract."""
    bridge = KimiK3Bridge()
    bridge.hf_config = kimi_k3_pretrained.config
    name = "language_model.model.layers.0.self_attn.A_log"
    source = torch.cat((torch.arange(96, dtype=torch.float32), torch.ones(32)))

    with pytest.raises(ValueError, match="zero-padded inactive A_log"):
        bridge.maybe_modify_loaded_hf_weight(name, {name: source})


def test_kda_a_log_export_restores_zero_padding() -> None:
    """K3 exports the 96 active KDA heads in the 128-entry HF layout."""
    bridge = KimiK3Bridge()
    name = "language_model.model.layers.0.self_attn.A_log"
    active = torch.arange(96, dtype=torch.float32)
    task = SimpleNamespace(weight_dtype=None)

    result = bridge.maybe_modify_converted_hf_weight(task, {name: active}, {name: torch.zeros(128)})

    assert result[name].shape == (128,)
    torch.testing.assert_close(result[name][:96], active, rtol=0, atol=0)
    assert torch.count_nonzero(result[name][96:]).item() == 0


def test_export_preserves_unconverted_multimodal_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """K3 keeps the published vision tower and projector in strict HF exports."""
    language = torch.tensor([1.0])
    vision = torch.tensor([2.0])
    projector = torch.tensor([3.0])
    unrelated = torch.tensor([4.0])
    tensors = {
        "vision_tower.encoder.weight": vision,
        "mm_projector.proj.weight": projector,
        "unrelated.weight": unrelated,
    }

    class _Source:
        @staticmethod
        def get_all_keys() -> list[str]:
            return list(tensors)

    class _State:
        source = _Source()

        def __getitem__(self, name: str) -> torch.Tensor:
            return tensors[name]

    monkeypatch.setattr(
        MegatronModelBridge,
        "stream_weights_megatron_to_hf",
        lambda *_args, **_kwargs: iter((HFWeightTuple("language_model.weight", language),)),
    )
    pretrained = SimpleNamespace(state=_State())

    result = list(KimiK3Bridge().stream_weights_megatron_to_hf([], pretrained))

    assert [item.param_name for item in result] == [
        "language_model.weight",
        "vision_tower.encoder.weight",
        "mm_projector.proj.weight",
    ]
    torch.testing.assert_close(result[1].weight, vision, rtol=0, atol=0)
    torch.testing.assert_close(result[2].weight, projector, rtol=0, atol=0)


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


def test_transformer_layer_does_not_forward_input_ids_to_upstream_moe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K3 keeps token IDs out of the upstream MCore MoELayer call."""
    hidden_states = torch.ones(2, 1, 4)
    block_residual = torch.zeros(2, 1, 1, 4)
    padding_mask = torch.ones(1, 2, dtype=torch.bool)
    input_ids = torch.ones(1, 2, dtype=torch.long)
    mlp = Mock(return_value=(torch.zeros_like(hidden_states), None))
    layer = SimpleNamespace(
        layer_number=2,
        config=SimpleNamespace(num_layers=4, hidden_size=4),
        attn_res_block_size=12,
        self_attention_res_proj=Mock(),
        self_attention_res_norm=Mock(),
        input_layernorm=Mock(),
        self_attention=Mock(return_value=(torch.zeros_like(hidden_states), None)),
        mlp_res_proj=Mock(),
        mlp_res_norm=Mock(),
        pre_mlp_layernorm=Mock(),
        mlp=mlp,
        is_stage_exit=False,
        _add_bias=KimiK3TransformerLayer._add_bias,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.kimi.kimi_k3_layers.attn_res_aggregate",
        lambda prefix_sum, *_args: prefix_sum,
    )

    KimiK3TransformerLayer.forward(
        layer,
        hidden_states,
        context=block_residual,
        padding_mask=padding_mask,
        input_ids=input_ids,
    )

    assert mlp.call_args.kwargs == {"padding_mask": padding_mask}
