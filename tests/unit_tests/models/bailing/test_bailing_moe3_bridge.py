# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from torch import nn

from megatron.bridge.models.bailing.bailing_moe3_bridge import (
    _LING3_MTP_PATTERN,
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
    bailing_moe3_hybrid_stack_spec,
)
from megatron.bridge.models.conversion.model_bridge import ModelConfigNotSupportedError, get_model_bridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


pytestmark = pytest.mark.unit

_TINY_GOLDEN_PATTERN = "K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E"
_FLASH_GOLDEN_PATTERN = "K-K-KEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+E"


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
        head_dim=128,
        num_kv_heads_for_linear_attn=0,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=512,
        num_shared_experts=1,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        short_conv_kernel_size=4,
        num_nextn_predict_layers=0,
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
        embedding_dropout=0.0,
        output_dropout=0.0,
        tie_word_embeddings=False,
        use_qk_norm=True,
        linear_silu=True,
        group_norm_size=1,
        moe_router_enable_expert_bias=True,
        router_dtype="fp32",
        norm_topk_prob=True,
        scale_router_input=False,
        topk_method="noaux_tc",
        score_function="sigmoid",
        rotary_dim=64,
        rope_interleave=True,
        rope_scaling={
            "rope_theta": 6_000_000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
        routed_scaling_factor=2.5,
        n_group=8,
        topk_group=4,
        kda_safe_gate=True,
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
        embedding_dropout=0.0,
        output_dropout=0.0,
        tie_word_embeddings=False,
        use_qk_norm=True,
        linear_silu=True,
        group_norm_size=1,
        moe_router_enable_expert_bias=True,
        router_dtype="fp32",
        norm_topk_prob=True,
        scale_router_input=False,
        topk_method="noaux_tc",
        score_function="sigmoid",
        rotary_dim=64,
        rope_interleave=True,
        rope_scaling={
            "rope_theta": 6_000_000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
        routed_scaling_factor=2.5,
        n_group=8,
        topk_group=4,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        num_kv_heads_for_linear_attn=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tiny_hybrid_pattern_is_fixed() -> None:
    assert _tiny_hybrid_layer_pattern(_tiny_config()) == _TINY_GOLDEN_PATTERN


def test_flash_hybrid_pattern_has_dense_prefix_and_grouped_mla() -> None:
    pattern = _flash_hybrid_layer_pattern(_flash_config())

    assert pattern == _FLASH_GOLDEN_PATTERN
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


def test_bailing_moe3_disables_incompatible_default_gpt_builder_config() -> None:
    bridge = BailingMoeV3Bridge()

    with pytest.raises(ModelConfigNotSupportedError, match="sets MODEL_CONFIG_CLASS to None"):
        bridge.hf_config_to_model_config(_tiny_config())


def test_tiny_provider_maps_supported_config_variations() -> None:
    config = _tiny_config(
        hidden_act="gelu",
        attention_dropout=0.1,
        hidden_dropout=0.2,
        intermediate_size=4096,
        vocab_size=157200,
        q_lora_rank=128,
        kv_lora_rank=256,
        num_experts=64,
        num_experts_per_tok=4,
        routed_scaling_factor=1.5,
        n_group=4,
        topk_group=2,
        kda_safe_gate=False,
        kda_lower_bound=None,
        moe_shared_expert_intermediate_size=384,
        mtp_loss_scaling_factor=0.25,
    )

    provider = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=config))

    assert provider.activation_func is torch.nn.functional.gelu
    assert provider.attention_dropout == 0.1
    assert provider.hidden_dropout == 0.2
    assert provider.ffn_hidden_size == 4096
    assert provider.vocab_size == 157200
    assert provider.q_lora_rank == 128
    assert provider.kv_lora_rank == 256
    assert provider.num_moe_experts == 64
    assert provider.moe_router_topk == 4
    assert provider.moe_router_topk_scaling_factor == 1.5
    assert provider.moe_router_num_groups == 4
    assert provider.moe_router_group_topk == 2
    assert provider.moe_router_dtype == "fp32"
    assert provider.moe_router_score_function == "sigmoid"
    assert provider.moe_router_pre_softmax is False
    assert provider.kda_safe_gate is False
    assert provider.kda_lower_bound is None
    assert provider.moe_shared_expert_intermediate_size == 384
    assert provider.mtp_loss_scaling_factor == 0.25


def test_tiny_provider_defaults_linear_heads_when_hf_field_is_absent() -> None:
    config = _tiny_config()
    del config.num_kv_heads_for_linear_attn

    provider = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=config))

    assert provider.linear_num_key_heads == config.num_attention_heads
    assert provider.linear_num_value_heads == config.num_attention_heads


@pytest.mark.parametrize(
    "override",
    [
        {"num_nextn_predict_layers": 2},
        {"num_nextn_predict_layers": 1, "mtp_use_kda": True},
        {"q_lora_rank": None},
        {"gated_attention_proj_granularity_type": "token_wise"},
        {"no_kda_lora": False},
        {"use_qkv_bias": True},
        {"tie_word_embeddings": True},
        {"use_qk_norm": False},
        {"moe_router_enable_expert_bias": False},
        {"router_dtype": "fp64"},
        {"scoring_func": "softmax"},
        {"score_function": "softmax"},
        {"norm_topk_prob": False},
        {"topk_method": "greedy"},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
        {"layer_group_size": 3},
    ],
)
def test_tiny_validation_fails_closed(override) -> None:
    with pytest.raises(ValueError):
        BailingMoeV3Bridge._validate_config(_tiny_config(**override))


@pytest.mark.parametrize(
    "override",
    [
        {"q_lora_rank": 256},
        {"num_nextn_predict_layers": 0},
        {"mtp_use_kda": True},
    ],
)
def test_flash_validation_fails_closed(override) -> None:
    with pytest.raises(ValueError):
        BailingMoeV3Bridge._validate_config(_flash_config(**override))


def test_tiny_provider_contains_serializable_mla_fields() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_tiny_config()))

    assert isinstance(provider, BailingMoe3HybridProvider)
    assert provider.hybrid_layer_pattern == _TINY_GOLDEN_PATTERN
    assert provider.num_layers == 48
    assert provider.q_lora_rank == 256
    assert provider.kv_lora_rank == 512
    assert provider.qk_head_dim == 128
    assert provider.qk_pos_emb_head_dim == 64
    assert provider.v_head_dim == 128
    assert provider.position_embedding_type == "rope"
    assert provider.rotary_percent == 1.0
    assert provider.rotary_scaling_factor == 1.0
    assert provider.mscale_all_dim == 1.0
    assert provider.gated_attention_proj_granularity == "headwise"
    assert provider.moe_router_pre_softmax is False
    assert provider.moe_router_enable_expert_bias is True
    assert set(vars(provider)) <= set(provider.__dataclass_fields__)


def test_bailing_provider_combines_hybrid_construction_and_mla_config() -> None:
    provider = BailingMoe3HybridProvider(
        hidden_size=256,
        num_attention_heads=4,
        hybrid_layer_pattern="K+E-",
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
    )

    assert isinstance(provider, HybridModelProvider)
    assert isinstance(provider, MLATransformerConfig)
    assert provider.multi_latent_attention is True
    assert provider.provide.__func__ is HybridModelProvider.provide
    assert {"q_lora_rank", "hybrid_layer_pattern", "linear_conv_kernel_dim"} <= set(provider.__dataclass_fields__)


def test_bailing_provider_finalize_runs_hybrid_and_mla_validation() -> None:
    provider = BailingMoe3HybridProvider(
        hidden_size=256,
        num_attention_heads=4,
        hybrid_layer_pattern="+",
        attention_output_gate=True,
        mla_down_proj_fusion=True,
    )

    with pytest.raises(ValueError, match="MLA output gating does not support fused down projections"):
        provider.finalize()

    assert provider.num_layers == 1


def test_flash_provider_uses_direct_q_and_one_mtp_layer() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_flash_config()))

    assert provider.hybrid_layer_pattern == _FLASH_GOLDEN_PATTERN
    assert provider.num_layers == 84
    assert provider.q_lora_rank is None
    assert provider.qk_layernorm is True
    assert provider.linear_num_key_heads == 32
    assert provider.num_moe_experts == 512
    assert provider.ffn_hidden_size == 6144
    assert provider.mtp_num_layers == 1
    assert provider.mtp_hybrid_override_pattern == _LING3_MTP_PATTERN
    assert provider.mtp_loss_scaling_factor == 0.0
    assert provider.position_embedding_type == "rope"
    assert provider.rotary_percent == 1.0
    assert set(vars(provider)) <= set(provider.__dataclass_fields__)


def test_flash_spec_uses_mcore_native_direct_q_and_standalone_kv_norm() -> None:
    spec = bailing_moe3_hybrid_stack_spec(_flash_config())
    mla = spec.submodules.mla_layer.submodules.self_attention

    assert mla.module is MLASelfAttention
    assert mla.submodules.q_layernorm is IdentityOp
    assert mla.submodules.kv_layernorm is TENorm


def test_tiny_provider_uses_low_rank_q_and_one_mtp_layer() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(
        SimpleNamespace(config=_tiny_config(num_nextn_predict_layers=1, mtp_use_kda=False))
    )

    assert provider.mtp_num_layers == 1
    assert provider.mtp_hybrid_override_pattern == _LING3_MTP_PATTERN
    assert provider.mtp_loss_scaling_factor == 0.0


def test_tiny_mapping_registry_covers_low_rank_q_mtp() -> None:
    bridge = BailingMoeV3Bridge()
    bridge.hf_config = _tiny_config(num_nextn_predict_layers=1, mtp_use_kda=False)
    registry = bridge.mapping_registry()

    mtp_prefix = "mtp.layers.0.mtp_model_layer.layers.0.self_attention"
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.q_a_proj.weight").megatron_param
        == f"{mtp_prefix}.linear_q_down_proj.weight"
    )
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.q_a_layernorm.weight").megatron_param
        == f"{mtp_prefix}.q_layernorm.weight"
    )
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.q_b_proj.weight").megatron_param
        == f"{mtp_prefix}.linear_q_up_proj.weight"
    )
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.kv_b_proj.weight").megatron_param
        == f"{mtp_prefix}.linear_kv_up_proj.weight"
    )


def test_spec_reuses_mcore_projection_builders_and_overrides_only_ling_norms() -> None:
    spec = bailing_moe3_hybrid_stack_spec(_tiny_config())

    default_kda = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention.submodules
    default_mla = hybrid_stack_spec.submodules.mla_layer.submodules.self_attention.submodules
    kda = spec.submodules.kda_layer.submodules.self_attention.submodules
    mla = spec.submodules.mla_layer.submodules.self_attention.submodules

    assert spec is not hybrid_stack_spec
    assert kda.beta_proj is default_kda.beta_proj
    assert mla.linear_q_down_proj is default_mla.linear_q_down_proj
    assert mla.linear_kv_down_proj is default_mla.linear_kv_down_proj
    assert mla.linear_gate is default_mla.linear_gate
    assert mla.q_layernorm is TENorm
    assert mla.kv_layernorm is TENorm
    assert default_mla.q_layernorm is IdentityOp
    assert default_mla.kv_layernorm is IdentityOp


@pytest.mark.parametrize(
    ("config_factory", "expected_layers", "expected_group_size", "expected_dense_layers", "expected_mtp_layers"),
    [
        (_tiny_config, 24, 4, 1, 0),
        (_flash_config, 42, 6, 2, 1),
    ],
)
def test_megatron_to_hf_config_restores_logical_architecture(
    config_factory,
    expected_layers: int,
    expected_group_size: int,
    expected_dense_layers: int,
    expected_mtp_layers: int,
) -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=config_factory()))
    provider.finalize()

    exported = bridge.megatron_to_hf_config(provider)

    assert exported["num_hidden_layers"] == expected_layers
    assert exported["layer_group_size"] == expected_group_size
    assert exported["first_k_dense_replace"] == expected_dense_layers
    assert exported["gated_attention_proj_granularity_type"] == "head_wise"
    assert exported["no_kda_lora"] is True
    assert exported["num_kv_heads_for_linear_attn"] == 0
    assert exported["scoring_func"] == "sigmoid"
    assert exported["score_function"] == "sigmoid"
    assert exported["norm_topk_prob"] is True
    assert exported["topk_method"] == "noaux_tc"
    assert exported["router_dtype"] == "fp32"
    assert exported["moe_router_enable_expert_bias"] is True
    assert exported["moe_shared_expert_intermediate_size"] in (512, 768)
    assert exported["mtp_loss_scaling_factor"] == 0.0
    assert exported["partial_rotary_factor"] == 0.5
    assert exported["rotary_dim"] == 64
    assert exported["rope_interleave"] is True
    assert exported["num_nextn_predict_layers"] == expected_mtp_layers


@pytest.mark.parametrize(
    ("config_factory", "invalid_mtp_layers"),
    [
        (_tiny_config, 2),
        (_flash_config, 0),
    ],
)
def test_megatron_to_hf_config_rejects_non_public_mtp_layout(config_factory, invalid_mtp_layers: int) -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=config_factory()))
    provider.mtp_num_layers = invalid_mtp_layers

    with pytest.raises(ValueError, match="export requires mtp_num_layers"):
        bridge.megatron_to_hf_config(provider)


def test_megatron_to_hf_config_does_not_infer_norm_topk_prob_from_pre_softmax() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_tiny_config()))
    provider.moe_router_pre_softmax = True

    exported = bridge.megatron_to_hf_config(provider)

    assert exported["norm_topk_prob"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("moe_router_score_function", "softmax"),
        ("moe_router_dtype", "fp64"),
        ("moe_router_load_balancing_type", "aux_loss"),
        ("moe_router_enable_expert_bias", False),
    ],
)
def test_megatron_to_hf_config_rejects_incompatible_router_semantics(name: str, value: object) -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_tiny_config()))
    if not hasattr(provider, name):
        raise ValueError(f"Unknown provider override: {name}")
    setattr(provider, name, value)

    with pytest.raises(ValueError, match=f"export requires {name}"):
        bridge.megatron_to_hf_config(provider)


@pytest.mark.parametrize(
    ("config_factory", "name", "value", "match"),
    [
        (_tiny_config, "attention_output_gate", False, "attention_output_gate"),
        (_tiny_config, "gated_attention_proj_granularity", "elementwise", "gated_attention_proj_granularity"),
        (_tiny_config, "position_embedding_type", "none", "position_embedding_type"),
        (_tiny_config, "qk_layernorm", False, "qk_layernorm"),
        (_tiny_config, "add_bias_linear", True, "add_bias_linear"),
        (_tiny_config, "moe_grouped_gemm", False, "moe_grouped_gemm"),
        (_tiny_config, "rotary_percent", 0.5, "rotary_percent"),
        (_tiny_config, "q_lora_rank", None, "low-rank-Q MLA"),
        (_flash_config, "q_lora_rank", 256, "direct-Q MLA"),
        (_tiny_config, "linear_value_head_dim", 64, "head dimensions"),
        (_tiny_config, "linear_num_value_heads", 8, "head counts"),
        (_tiny_config, "moe_shared_expert_intermediate_size", 0, "shared expert"),
    ],
)
def test_megatron_to_hf_config_rejects_incompatible_structure(
    config_factory,
    name: str,
    value: object,
    match: str,
) -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=config_factory()))
    if not hasattr(provider, name):
        raise ValueError(f"Unknown provider override: {name}")
    setattr(provider, name, value)

    with pytest.raises(ValueError, match=match):
        bridge.megatron_to_hf_config(provider)


def test_megatron_to_hf_config_rejects_incompatible_mtp_pattern() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_flash_config()))
    provider.finalize()
    provider.hybrid_layer_pattern = f"{_FLASH_GOLDEN_PATTERN}/KE"

    with pytest.raises(ValueError, match="MTP pattern suffixes"):
        bridge.megatron_to_hf_config(provider)


def test_tiny_mapping_registry_covers_logical_to_physical_layout() -> None:
    bridge = BailingMoeV3Bridge()
    bridge.hf_config = _tiny_config()
    registry = bridge.mapping_registry()

    assert registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.in_proj.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.6.self_attention.linear_gate.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.router.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.shared_experts.linear_fc2.weight") is not None

    mla_mapping = registry.hf_to_megatron_lookup("model.layers.3.attention.q_b_proj.weight")
    assert mla_mapping is not None
    assert mla_mapping.megatron_param == "decoder.layers.6.self_attention.linear_q_up_proj.weight"

    q_down = registry.hf_to_megatron_lookup("model.layers.3.attention.q_a_proj.weight")
    kv_down = registry.hf_to_megatron_lookup("model.layers.3.attention.kv_a_proj_with_mqa.weight")
    assert isinstance(q_down, AutoMapping)
    assert isinstance(kv_down, AutoMapping)

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

    kv_down = registry.hf_to_megatron_lookup("model.layers.5.attention.kv_a_proj_with_mqa.weight")
    mtp_kv_down = registry.hf_to_megatron_lookup("model.layers.42.attention.kv_a_proj_with_mqa.weight")
    assert isinstance(kv_down, AutoMapping)
    assert isinstance(mtp_kv_down, AutoMapping)


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
