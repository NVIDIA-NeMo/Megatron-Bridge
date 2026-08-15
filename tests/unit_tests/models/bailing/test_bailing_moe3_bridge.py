# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from torch import nn

from megatron.bridge.models.bailing.bailing_moe3_bridge import (
    _FLASH_MTP_PATTERN,
    _TINY_PATTERN,
    BailingMoeV3Bridge,
    _flash_hybrid_layer_pattern,
    _tiny_hybrid_layer_pattern,
)
from megatron.bridge.models.bailing.bailing_moe3_mappings import (
    _BailingMoe3KDAConv1dMapping,
    _BailingMoe3KDAInProjMapping,
)
from megatron.bridge.models.bailing.bailing_moe3_provider import BailingMoe3HybridProvider
from megatron.bridge.models.bailing.bailing_moe3_spec import (
    BailingMoe3DirectQMLASelfAttention,
    bailing_moe3_hybrid_stack_spec,
)
from megatron.bridge.models.conversion.model_bridge import get_model_bridge


pytestmark = pytest.mark.unit


def _tiny_config(**overrides):
    values = dict(
        architectures=["BailingMoeV3ForCausalLM"],
        model_type="bailing_hybrid",
        num_hidden_layers=24,
        layer_group_size=4,
        first_k_dense_replace=1,
        hidden_size=1536,
        vocab_size=157184,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=512,
        num_shared_experts=1,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        short_conv_kernel_size=4,
        num_nextn_predict_layers=0,
        gated_attention_proj_granularity_type="head_wise",
        use_kda_lora=False,
        no_kda_lora=True,
        use_qkv_bias=False,
        attention_bias=False,
        use_bias=False,
        mlp_bias=False,
        scoring_func="sigmoid",
        max_position_embeddings=131072,
        rope_theta=6000000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        intermediate_size=4608,
        num_hidden_layers_mtp=0,
        hidden_act="silu",
        torch_dtype="bfloat16",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        routed_scaling_factor=2.5,
        n_group=8,
        topk_group=4,
        kda_lower_bound=-5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _flash_config(**overrides):
    values = dict(
        architectures=["BailingMoeV3ForCausalLM"],
        model_type="bailing_hybrid",
        num_hidden_layers=42,
        layer_group_size=6,
        first_k_dense_replace=2,
        hidden_size=2560,
        vocab_size=157184,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_experts=512,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        moe_shared_expert_intermediate_size=768,
        num_shared_experts=1,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        head_dim=128,
        short_conv_kernel_size=4,
        num_nextn_predict_layers=1,
        mtp_use_kda=False,
        mtp_loss_scaling_factor=0.0,
        gated_attention_proj_granularity_type="head_wise",
        use_kda_lora=False,
        no_kda_lora=True,
        use_qkv_bias=False,
        attention_bias=False,
        use_bias=False,
        mlp_bias=False,
        scoring_func="sigmoid",
        max_position_embeddings=262144,
        rope_theta=6000000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        intermediate_size=6144,
        hidden_act="silu",
        torch_dtype="bfloat16",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        routed_scaling_factor=2.5,
        n_group=8,
        topk_group=4,
        kda_lower_bound=-5.0,
        num_kv_heads_for_linear_attn=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tiny_hybrid_pattern_is_fixed() -> None:
    assert _tiny_hybrid_layer_pattern(_tiny_config()) == _TINY_PATTERN


def test_flash_hybrid_pattern_has_dense_prefix_and_grouped_mla() -> None:
    pattern = _flash_hybrid_layer_pattern(_flash_config())

    assert len(pattern) == 84
    assert pattern[:4] == "K-K-"
    assert all(pattern[2 * layer] == "+" for layer in (5, 11, 17, 23, 29, 35, 41))
    assert all(pattern[2 * layer] == "K" for layer in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10))
    assert pattern.count("-") == 2
    assert pattern.count("E") == 40


def test_bailing_moe3_registration_uses_architecture_and_model_type() -> None:
    config = _tiny_config()
    bridge = get_model_bridge("BailingMoeV3ForCausalLM", hf_config=config)

    assert isinstance(bridge, BailingMoeV3Bridge)
    assert bridge.SOURCE_NAME == "BailingMoeV3ForCausalLM"
    assert bridge.MODEL_TYPE == "bailing_hybrid"
    assert bridge.hf_config is config


@pytest.mark.parametrize(
    "override",
    [
        {"num_nextn_predict_layers": 1},
        {"gated_attention_proj_granularity_type": "token_wise"},
        {"use_qkv_bias": True},
        {"layer_group_size": 3},
    ],
)
def test_tiny_validation_fails_closed(override) -> None:
    with pytest.raises(ValueError):
        BailingMoeV3Bridge._validate_config(_tiny_config(**override))


@pytest.mark.parametrize(
    "override",
    [
        {"intermediate_size": 4608},
        {"q_lora_rank": 256},
        {"num_nextn_predict_layers": 0},
    ],
)
def test_flash_validation_fails_closed(override) -> None:
    with pytest.raises(ValueError):
        BailingMoeV3Bridge._validate_config(_flash_config(**override))


def test_tiny_provider_contains_serializable_mla_fields() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_tiny_config()))

    assert isinstance(provider, BailingMoe3HybridProvider)
    assert provider.hybrid_layer_pattern == _TINY_PATTERN
    assert provider.num_layers == 48
    assert provider.q_lora_rank == 256
    assert provider.kv_lora_rank == 512
    assert provider.qk_head_dim == 128
    assert provider.qk_pos_emb_head_dim == 64
    assert provider.v_head_dim == 128


def test_flash_provider_uses_direct_q_and_one_mtp_layer() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_flash_config()))

    assert provider.hybrid_layer_pattern == _flash_hybrid_layer_pattern(_flash_config())
    assert provider.num_layers == 84
    assert provider.q_lora_rank is None
    assert provider.qk_layernorm is True
    assert provider.linear_num_key_heads == 32
    assert provider.num_moe_experts == 512
    assert provider.ffn_hidden_size == 6144
    assert provider.mtp_num_layers == 1
    assert provider.mtp_hybrid_override_pattern == _FLASH_MTP_PATTERN
    assert provider.mtp_loss_scaling_factor == 0.0


def test_flash_spec_uses_plain_q_and_standalone_kv_norm() -> None:
    spec = bailing_moe3_hybrid_stack_spec(_flash_config())
    mla = spec.submodules.mla_layer.submodules.self_attention

    assert mla.module is BailingMoe3DirectQMLASelfAttention
    assert mla.submodules.q_layernorm.__name__ == "IdentityOp"
    assert mla.submodules.kv_layernorm.__name__ == "TENorm"


def test_default_spec_keeps_tiny_low_rank_path_for_legacy_dcp() -> None:
    spec = bailing_moe3_hybrid_stack_spec()
    mla = spec.submodules.mla_layer.submodules.self_attention

    assert mla.module is MLASelfAttention
    assert mla.submodules.q_layernorm.__name__ == "TENorm"


def test_tiny_mapping_registry_covers_logical_to_physical_layout() -> None:
    bridge = BailingMoeV3Bridge()
    bridge.hf_config = _tiny_config()
    registry = bridge.mapping_registry()

    assert registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.in_proj.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.6.self_attention.linear_gate.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.router.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.experts.linear_fc1.weight0") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.shared_experts.linear_fc2.weight") is not None

    mla_mapping = registry.hf_to_megatron_lookup("model.layers.3.attention.q_b_proj.weight")
    assert mla_mapping is not None
    assert mla_mapping.megatron_param == "decoder.layers.6.self_attention.linear_q_up_proj.weight"

    kda_mapping = registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.in_proj.weight")
    assert kda_mapping is not None
    assert list(kda_mapping.hf_param) == ["q", "k", "v", "f", "g"]


def test_flash_mapping_registry_covers_direct_q_and_mtp() -> None:
    bridge = BailingMoeV3Bridge()
    bridge.hf_config = _flash_config()
    registry = bridge.mapping_registry()

    direct_q = registry.hf_to_megatron_lookup("model.layers.5.attention.q_proj.weight")
    assert direct_q is not None
    assert direct_q.megatron_param == "decoder.layers.10.self_attention.linear_q_proj.weight"

    kda = registry.hf_to_megatron_lookup("model.layers.4.attention.q_proj.weight")
    assert kda is not None
    assert kda.megatron_param == "decoder.layers.8.self_attention.in_proj.weight"

    mtp_q = registry.hf_to_megatron_lookup("model.layers.42.attention.q_proj.weight")
    assert mtp_q is not None
    assert mtp_q.megatron_param == "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_q_proj.weight"

    mtp_expert = registry.hf_to_megatron_lookup("model.layers.42.mlp.experts.7.down_proj.weight")
    assert mtp_expert is not None
    assert mtp_expert.megatron_param == "mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc2.weight7"

    mtp_norm = registry.hf_to_megatron_lookup("model.layers.42.enorm.weight")
    assert mtp_norm is not None
    assert mtp_norm.megatron_param == "mtp.layers.0.enorm.weight"


class _FakeTpInProjMapping(_BailingMoe3KDAInProjMapping):
    """Exercise section-wise TP logic without initializing distributed state."""

    _shared_splits = None

    def __init__(self, *args, rank: int, gathered: list[torch.Tensor] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._gathered = gathered

    @property
    def tp_size(self) -> int:
        return 2

    @property
    def tp_rank(self) -> int:
        return self._rank

    def scatter_to_tp_ranks(self, splits, output_shape, dtype, device, src_rank=0):
        if splits is not None:
            type(self)._shared_splits = splits
        return type(self)._shared_splits[self._rank]

    def gather_from_tp_ranks(self, tensor):
        return self._gathered

    def broadcast_from_pp_rank(self, tensor, cache_key=None):
        return tensor


class _FakeTpConvMapping(_BailingMoe3KDAConv1dMapping):
    """Exercise KDA convolution section-wise TP logic without distributed state."""

    _shared_splits = None

    def __init__(self, *args, rank: int, gathered: list[torch.Tensor] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._gathered = gathered

    @property
    def tp_size(self) -> int:
        return 2

    @property
    def tp_rank(self) -> int:
        return self._rank

    def scatter_to_tp_ranks(self, splits, output_shape, dtype, device, src_rank=0):
        if splits is not None:
            type(self)._shared_splits = splits
        return type(self)._shared_splits[self._rank]

    def gather_from_tp_ranks(self, tensor):
        return self._gathered

    def broadcast_from_pp_rank(self, tensor, cache_key=None):
        return tensor


def _kda_config() -> SimpleNamespace:
    return SimpleNamespace(
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_value_head_dim=4,
        linear_num_value_heads=1,
    )


def test_kda_projection_mapping_preserves_semantic_sections_under_tp() -> None:
    config = _kda_config()
    target = nn.Module()
    target.config = config
    target.in_proj = nn.Linear(4, 10, bias=False)
    source = {
        name: torch.arange(4, dtype=torch.float32).reshape(4, 1) + value
        for name, value in zip(("q", "k", "v", "f", "g"), (0, 10, 20, 30, 40), strict=True)
    }

    rank0 = _FakeTpInProjMapping("in_proj.weight", "q", "k", "v", "f", "g", rank=0)
    rank1 = _FakeTpInProjMapping("in_proj.weight", "q", "k", "v", "f", "g", rank=1)
    local0 = rank0.hf_to_megatron(source, target)
    local1 = rank1.hf_to_megatron(source, target)

    assert torch.equal(local0, torch.cat([source[name][:2] for name in ("q", "k", "v", "f", "g")]))
    assert torch.equal(local1, torch.cat([source[name][2:] for name in ("q", "k", "v", "f", "g")]))

    exported = _FakeTpInProjMapping(
        "in_proj.weight",
        "q",
        "k",
        "v",
        "f",
        "g",
        rank=0,
        gathered=[local0, local1],
    ).megatron_to_hf(local0, target)
    for name in source:
        assert torch.equal(exported[name], source[name])


def test_kda_conv_mapping_preserves_semantic_sections_under_tp() -> None:
    config = _kda_config()
    target = nn.Module()
    target.config = config
    target.conv1d = nn.Conv1d(1, 6, kernel_size=2, bias=False)
    source = {
        name: torch.arange(8, dtype=torch.float32).reshape(4, 1, 2) + value
        for name, value in zip(("q", "k", "v"), (0, 10, 20), strict=True)
    }

    rank0 = _FakeTpConvMapping("conv1d.weight", "q", "k", "v", rank=0)
    rank1 = _FakeTpConvMapping("conv1d.weight", "q", "k", "v", rank=1)
    local0 = rank0.hf_to_megatron(source, target)
    local1 = rank1.hf_to_megatron(source, target)

    assert torch.equal(local0, torch.cat([source[name][:2] for name in ("q", "k", "v")]))
    assert torch.equal(local1, torch.cat([source[name][2:] for name in ("q", "k", "v")]))

    exported = _FakeTpConvMapping(
        "conv1d.weight",
        "q",
        "k",
        "v",
        rank=0,
        gathered=[local0, local1],
    ).megatron_to_hf(local0, target)
    for name in source:
        assert torch.equal(exported[name], source[name])
