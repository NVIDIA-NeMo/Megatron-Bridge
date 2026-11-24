# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HF architecture and native entry correspondence for Kimi K3."""

from copy import deepcopy

import pytest
from transformers import PretrainedConfig

from megatron.bridge.models.kimi.kimi_k3_hybrid_bridge import KimiK3HybridBridge
from megatron.bridge.models.kimi.kimi_k3_hybrid_config import kimi_k3_hybrid_pattern, kimi_k3_pipeline_pattern
from tests.unit_tests.models.kimi.test_kimi_k3_bridge import kimi_k3_text_config as kimi_k3_text_config


@pytest.mark.parametrize(
    ("kda", "moe", "expected"),
    [([1, 2, 4, 5], [0, 1, 1, 1, 1, 1], "K-KE+EKEKE+E"), ([], [0, 0], "+-+-"), ([1, 2], [1, 1], "KEKE")],
)
def test_pattern_from_hf_blocks(kda, moe, expected):
    assert kimi_k3_hybrid_pattern(num_layers=len(moe), kda_layers=kda, moe_layers=moe) == expected


@pytest.mark.parametrize("kda", [[0], [3], [1, 1]])
def test_pattern_rejects_invalid_kda_indices(kda):
    with pytest.raises(ValueError, match="unique one-based"):
        kimi_k3_hybrid_pattern(num_layers=2, kda_layers=kda, moe_layers=[0, 1])


@pytest.mark.parametrize("frequency", [1, 2])
def test_native_config_roundtrip(kimi_k3_text_config, frequency):
    text = deepcopy(kimi_k3_text_config)
    text.moe_layer_freq = frequency
    hf = PretrainedConfig(text_config=PretrainedConfig(**vars(text)))
    bridge = KimiK3HybridBridge()
    original = bridge.hf_config_to_model_config(hf)
    exported = bridge.megatron_to_hf_config(original)
    exported["text_config"] = PretrainedConfig(**exported["text_config"])
    restored = bridge.hf_config_to_model_config(PretrainedConfig(**exported))
    for field in (
        "hybrid_layer_pattern",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_pos_emb_head_dim",
        "v_head_dim",
        "moe_latent_size",
        "moe_ffn_hidden_size",
        "moe_shared_expert_intermediate_size",
        "num_moe_experts",
        "moe_router_topk",
        "kimi_attn_res_block_size",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "kimi_kda_gate_lower_bound",
    ):
        assert getattr(original, field) == getattr(restored, field), field
    assert original.num_layers == 2 * text.num_hidden_layers
    assert original.kimi_attn_res_block_size == text.attn_res_block_size
    assert original.kimi_linear_num_heads == text.linear_attn_config["num_heads"]
    assert original.kimi_linear_head_dim == text.linear_attn_config["head_dim"]
    assert not getattr(original, "enable_attention_residuals", False)
    assert not original.moe_router_force_load_balancing


def test_provider_uses_native_stack(kimi_k3_text_config):
    from megatron.bridge.models.kimi.kimi_k3_hybrid_config import kimi_k3_hybrid_stack_spec
    from megatron.bridge.models.kimi.kimi_k3_hybrid_provider import KimiK3HybridModelProvider

    bridge = KimiK3HybridBridge()
    hf = PretrainedConfig(text_config=PretrainedConfig(**vars(kimi_k3_text_config)))
    provider = KimiK3HybridModelProvider(**bridge.hf_config_to_provider_kwargs(hf))
    config = bridge.hf_config_to_model_config(hf)
    assert provider.hybrid_stack_spec is kimi_k3_hybrid_stack_spec
    assert provider.hybrid_layer_pattern == config.hybrid_layer_pattern
    assert provider.num_layers == config.num_layers
    assert provider.position_embedding_type == "none"


def test_default_bridge_dispatches_to_hybrid():
    from megatron.bridge.models.conversion.model_bridge import get_model_bridge
    from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge
    from megatron.bridge.models.kimi.kimi_k3_hybrid_config import KimiK3HybridModelConfig

    bridge = get_model_bridge("KimiK3ForConditionalGeneration")
    assert type(bridge) is KimiK3Bridge
    assert bridge.MODEL_CONFIG_CLASS is KimiK3HybridModelConfig


@pytest.mark.parametrize(
    ("pattern", "stages", "counts", "expected"),
    [
        ("K-KE+EKEKE+E", 2, None, "K-KE+E|KEKE+E"),
        ("K-KE+EKEKE+E", 2, (4, 2), "K-KE+EKE|KE+E"),
        ("K-KE+EKE|KE+E", 2, None, "K-KE+EKE|KE+E"),
        ("K-KE+EKE|KE+E", 1, None, "K-KE+EKEKE+E"),
        ("KE" * 93, 4, None, "|".join("KE" * n for n in (24, 23, 23, 23))),
    ],
)
def test_pipeline_uses_complete_source_blocks(pattern, stages, counts, expected):
    assert kimi_k3_pipeline_pattern(pattern, pipeline_size=stages, blocks_per_stage=counts) == expected


@pytest.mark.parametrize(
    ("pattern", "stages", "counts"),
    [
        ("K|EKE", 2, None),
        ("KEKE|", 2, None),
        ("KEKE", 3, None),
        ("KEKE", 2, (1, 2)),
        ("KEKE", 2, (0, 2)),
        ("KK", 1, None),
    ],
)
def test_pipeline_rejects_invalid_boundaries(pattern, stages, counts):
    with pytest.raises(ValueError, match="Kimi K3"):
        kimi_k3_pipeline_pattern(pattern, pipeline_size=stages, blocks_per_stage=counts)


def test_recipes_consume_canonical_model(kimi_k3_text_config, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.kimi.gb300.kimi_k3 import kimi_k3_pretrain_64gpu_gb300_bf16_proxy_config
    from megatron.bridge.recipes.kimi.kimi_k3 import kimi_k3_pretrain_config

    text = deepcopy(kimi_k3_text_config)
    text.num_hidden_layers = 24
    text.linear_attn_config["kda_layers"] = [i for i in range(1, 25) if i % 4]
    hf = PretrainedConfig(text_config=PretrainedConfig(**vars(text)))

    def from_hf(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {"trust_remote_code": True}
        return SimpleNamespace(get_model_config=lambda: KimiK3HybridBridge().hf_config_to_model_config(hf))

    monkeypatch.setenv("KIMI_K3_HF_PATH", str(tmp_path))
    monkeypatch.setattr("megatron.bridge.recipes.kimi.kimi_k3.AutoBridge.from_hf_pretrained", from_hf)
    regular = kimi_k3_pretrain_config()
    assert regular.model.hybrid_layer_pattern.startswith("K-")
    assert not regular.model.moe_router_force_load_balancing
    assert regular.model.pipeline_model_parallel_size == 4
    assert regular.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert regular.tokenizer.tokenizer_model == str(tmp_path)
    proxy = kimi_k3_pretrain_64gpu_gb300_bf16_proxy_config()
    assert proxy.model.hybrid_layer_pattern == "KEKEKE+E" * 6
    assert proxy.model.num_layers == 48
    assert proxy.model.moe_router_force_load_balancing
    assert proxy.model.pipeline_model_parallel_size == 1
    assert proxy.tokenizer.tokenizer_type == "NullTokenizer"
    assert not proxy.ddp.overlap_grad_reduce
    assert proxy.model.kimi_linear_num_heads == regular.model.kimi_linear_num_heads


@pytest.mark.parametrize("partitioned", [False, True])
def test_saved_config_roundtrip(kimi_k3_text_config, partitioned):
    from megatron.bridge.models.kimi.kimi_k3_config import KimiK3TransformerConfig
    from megatron.bridge.models.kimi.kimi_k3_hybrid_config import KimiK3HybridModelConfig, kimi_k3_hybrid_stack_spec

    hf = PretrainedConfig(text_config=PretrainedConfig(**vars(kimi_k3_text_config)))
    config = KimiK3HybridBridge().hf_config_to_model_config(hf)
    config.pipeline_model_parallel_size = 2
    if partitioned:
        config.finalize()
    restored = KimiK3HybridModelConfig.from_dict(config.as_dict())
    assert isinstance(restored.transformer, KimiK3TransformerConfig)
    assert restored.hybrid_stack_spec is kimi_k3_hybrid_stack_spec
    assert restored.as_dict() == config.as_dict()
    restored.finalize()
    assert restored.hybrid_layer_pattern.count("|") == 1


@pytest.mark.parametrize(
    "field,value", [("linear_num_key_heads", 123), ("linear_key_head_dim", 123), ("linear_conv_kernel_dim", 123)]
)
def test_rejects_inconsistent_geometry(kimi_k3_text_config, field, value):
    from megatron.bridge.models.kimi.kimi_k3_hybrid_config import kimi_k3_hybrid_stack_spec

    hf = PretrainedConfig(text_config=PretrainedConfig(**vars(kimi_k3_text_config)))
    config = KimiK3HybridBridge().hf_config_to_model_config(hf)
    assert hasattr(config, field)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="must agree"):
        kimi_k3_hybrid_stack_spec(config)
    with pytest.raises(ValueError, match="must agree"):
        KimiK3HybridBridge.megatron_to_hf_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cuda_graph_impl", "local"),
        ("fine_grained_activation_offloading", True),
        ("overlap_moe_expert_parallel_comm", True),
        ("moe_shared_expert_overlap", True),
        ("enable_mhc_connections", True),
    ],
)
def test_rejects_unsupported_stack_execution(kimi_k3_text_config, field, value):
    from megatron.bridge.models.kimi.kimi_k3_hybrid_config import kimi_k3_hybrid_stack_spec

    hf = PretrainedConfig(text_config=PretrainedConfig(**vars(kimi_k3_text_config)))
    config = KimiK3HybridBridge().hf_config_to_model_config(hf)
    assert hasattr(config, field)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="Kimi K3 Hybrid"):
        kimi_k3_hybrid_stack_spec(config)
