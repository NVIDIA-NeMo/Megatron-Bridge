# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.bridge.models.bailing.bailing_moe2_bridge import BailingMoeV2Bridge


pytestmark = pytest.mark.unit


def test_mtp_mappings_resolve_live_mcore_parameter_names() -> None:
    bridge = BailingMoeV2Bridge()
    bridge.hf_config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    registry = bridge.mapping_registry()

    mtp_parameter_names = [
        "mtp.layers.0.mtp_model_layer.input_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.pre_mlp_layernorm.weight",
        "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight",
        "mtp.layers.0.mtp_model_layer.mlp.router.weight",
        "mtp.layers.0.mtp_model_layer.mlp.router.expert_bias",
        "mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1.weight0",
        "mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2.weight0",
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc1.weight",
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2.weight",
    ]

    missing = [name for name in mtp_parameter_names if registry.megatron_to_hf_lookup(name) is None]

    assert missing == []
