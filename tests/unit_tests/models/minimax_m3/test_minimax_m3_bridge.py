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

"""
Unit tests for the MiniMax-M3 bridge.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.minimax_m3.minimax_m3_bridge import (
    MiniMaxM3Bridge,
    MiniMaxM3ModelProvider,
    MiniMaxM3TextBridge,
    MiniMaxM3TopKRouter,
    TopKRouter,
    _FusedGateUpMapping,
    _promote_router_weights_to_float32,
    minimax_m3_block_spec,
    quick_gelu,
)
from megatron.bridge.models.model_provider import _apply_mixed_precision_wrapper
from megatron.bridge.utils.instantiate_utils import instantiate


pytestmark = pytest.mark.unit

try:
    from transformers import MiniMaxM3VLForCausalLM, MiniMaxM3VLTextConfig
except ImportError:
    MiniMaxM3VLForCausalLM = None
    MiniMaxM3VLTextConfig = None


# Toy text-backbone config (mirrors the shape of MiniMaxAI/MiniMax-M3 text_config)
_MINIMAX_M3_TEXT_CONFIG = {
    "architectures": ["MiniMaxM3SparseForCausalLM"],
    "hidden_size": 64,
    "intermediate_size": 32,
    "dense_intermediate_size": 128,
    "shared_intermediate_size": 32,
    "n_shared_experts": 1,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "hidden_act": "swigluoai",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "rotary_dim": 8,
    "partial_rotary_factor": 0.5,
    "vocab_size": 1024,
    "tie_word_embeddings": False,
    "attention_dropout": 0.0,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "scoring_func": "sigmoid",
    "use_routing_bias": True,
    "use_qk_norm": True,
    "use_gemma_norm": True,
    "qk_norm_type": "per_head",
    "routed_scaling_factor": 2.0,
    "router_aux_loss_coef": 0.001,
    "moe_layer_freq": [0, 1, 1, 1],
    "num_nextn_predict_layers": 1,
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "torch_dtype": "bfloat16",
    "bos_token_id": 1000,
    "eos_token_id": 1001,
    "pad_token_id": None,
}

_MINIMAX_M3_VL_CONFIG = {
    "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
    "model_type": "minimax_m3_vl",
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


_DELETE = object()


def _make_text_config(overrides: dict | None = None) -> Mock:
    values = dict(_MINIMAX_M3_TEXT_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if value is _DELETE:
                values.pop(key, None)
            else:
                values[key] = value
    cfg = Mock(spec=list(values.keys()))
    for k, v in values.items():
        setattr(cfg, k, v)
    return cfg


def _make_pretrained(text_overrides: dict | None = None) -> Mock:
    text_cfg = _make_text_config(text_overrides)

    outer_keys = list(_MINIMAX_M3_VL_CONFIG.keys()) + ["text_config"]
    outer_cfg = Mock(spec=outer_keys)
    for k, v in _MINIMAX_M3_VL_CONFIG.items():
        setattr(outer_cfg, k, v)
    outer_cfg.text_config = text_cfg

    m = Mock(spec=PreTrainedCausalLM)
    m.config = outer_cfg
    m.generation_config = Mock(spec=GenerationConfig)
    return m


class TestMiniMaxM3Bridge:
    """Unit tests for MiniMaxM3Bridge config mapping and mapping registry."""

    @pytest.fixture
    def mock_pretrained(self):
        return _make_pretrained()

    def test_registration(self):
        assert issubclass(MiniMaxM3Bridge, MegatronModelBridge)
        assert issubclass(MiniMaxM3TextBridge, MiniMaxM3Bridge)

    def test_provider_bridge_maps_core_config(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        text_config = mock_pretrained.config.text_config
        assert provider.hidden_size == text_config.hidden_size
        assert provider.num_layers == text_config.num_hidden_layers
        assert provider.num_attention_heads == text_config.num_attention_heads
        assert provider.num_query_groups == text_config.num_key_value_heads
        assert provider.kv_channels == text_config.head_dim
        assert provider.vocab_size == text_config.vocab_size
        assert provider.layernorm_epsilon == text_config.rms_norm_eps
        assert provider.rotary_base == text_config.rope_theta
        assert provider.num_moe_experts == text_config.num_local_experts
        assert provider.moe_router_topk == text_config.num_experts_per_tok
        assert provider.share_embeddings_and_output_weights is False

    def test_provider_bridge_splits_dense_and_expert_ffn_sizes(self, mock_pretrained):
        """intermediate_size is the per-expert size; dense layers use dense_intermediate_size."""
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        text_config = mock_pretrained.config.text_config
        assert provider.ffn_hidden_size == text_config.dense_intermediate_size
        assert provider.moe_ffn_hidden_size == text_config.intermediate_size

    def test_provider_bridge_sets_moe_sigmoid_routing(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_pre_softmax is False
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_router_load_balancing_type == "aux_loss"
        assert provider.moe_router_topk_scaling_factor == 2.0
        assert provider.moe_shared_expert_intermediate_size == 32
        assert provider.moe_shared_expert_overlap is False
        assert provider.moe_layer_freq == [0, 1, 1, 1]

    def test_provider_bridge_derives_moe_layer_freq_from_mlp_layer_types(self):
        """The native transformers config exposes mlp_layer_types instead of moe_layer_freq."""
        mock_pretrained = _make_pretrained(
            {"moe_layer_freq": _DELETE, "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"]}
        )
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.moe_layer_freq == [0, 1, 1, 1]

    def test_provider_bridge_disables_mtp(self, mock_pretrained):
        """The checkpoint config advertises MTP layers but ships no mtp.* weights."""
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.mtp_num_layers is None

    def test_provider_bridge_sets_gemma_style_norm(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.normalization == "RMSNorm"
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.qk_layernorm is True

    def test_provider_bridge_sets_swigluoai_activation(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.gated_linear_unit is True
        assert provider.activation_func is quick_gelu
        assert provider.activation_func_clamp_value == 7.0
        assert provider.glu_linear_offset == 1.0
        assert provider.bias_activation_fusion is False

    def test_hf_to_megatron_activation_swigluoai(self):
        assert MiniMaxM3Bridge.hf_to_megatron_activation("swigluoai") is quick_gelu

    def test_hf_to_megatron_activation_unknown_raises(self):
        with pytest.raises(ValueError):
            MiniMaxM3Bridge.hf_to_megatron_activation("not_a_real_activation")

    def test_provider_bridge_calculates_rotary_percent(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        text_config = mock_pretrained.config.text_config
        expected = text_config.rotary_dim / text_config.head_dim
        assert abs(provider.rotary_percent - expected) < 1e-6

    def test_provider_bridge_rotary_percent_missing_fields(self):
        """When rotary_dim is absent, no AttributeError is raised."""
        mock_pretrained = _make_pretrained({"rotary_dim": None})
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert hasattr(provider, "rotary_percent")

    def test_provider_bridge_dtype_bfloat16(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.autocast_dtype == torch.bfloat16

    def test_provider_bridge_caps_seq_length(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.seq_length == 4096

    def test_provider_bridge_flat_text_config(self):
        """A flat (non-nested) text config is accepted for text-only checkpoints."""
        text_cfg = _make_text_config(
            {
                "model_type": "minimax_m3_vl_text",
                "architectures": ["MiniMaxM3VLForCausalLM"],
                "n_shared_experts": _DELETE,
            }
        )
        m = Mock(spec=PreTrainedCausalLM)
        m.config = text_cfg
        m.generation_config = Mock(spec=GenerationConfig)

        bridge = MiniMaxM3TextBridge()
        provider = bridge.provider_bridge(m)
        assert provider.hidden_size == text_cfg.hidden_size
        assert provider.moe_shared_expert_intermediate_size == text_cfg.shared_intermediate_size
        assert provider.hf_max_position_embeddings == text_cfg.max_position_embeddings

    def test_mapping_registry_contains_critical_weights(self):
        bridge = MiniMaxM3Bridge()
        registry = bridge.mapping_registry()

        megatron_params = [str(m.megatron_param) for m in registry]
        assert any("word_embeddings" in p for p in megatron_params), "Embedding mapping missing"
        assert any("output_layer" in p for p in megatron_params), "LM head mapping missing"
        assert any("linear_qkv.weight" in p for p in megatron_params), "QKV mapping missing"
        assert any("linear_proj" in p for p in megatron_params), "o_proj mapping missing"
        assert any("q_layernorm" in p for p in megatron_params), "Q norm mapping missing"
        assert any("k_layernorm" in p for p in megatron_params), "K norm mapping missing"
        assert any("mlp.router.weight" in p for p in megatron_params), "MoE router mapping missing"
        assert any("mlp.router.expert_bias" in p for p in megatron_params), "Expert bias mapping missing"
        assert any("mlp.experts.linear_fc1" in p for p in megatron_params), "Expert gate/up mapping missing"
        assert any("mlp.experts.linear_fc2" in p for p in megatron_params), "Expert down mapping missing"
        assert any("mlp.shared_experts.linear_fc1" in p for p in megatron_params), "Shared expert mapping missing"
        assert any("mlp.linear_fc1.weight" in p for p in megatron_params), "Dense MLP mapping missing"

    def test_router_hook_preserves_float32_during_native_state_reload(self):
        router = TopKRouter.__new__(TopKRouter)
        torch.nn.Module.__init__(router)
        router.weight = torch.nn.Parameter(torch.empty(8, 64, dtype=torch.bfloat16))
        model = torch.nn.Module()
        model.router = router
        source_weight = torch.randn(8, 64, dtype=torch.float32)

        _promote_router_weights_to_float32([model])
        model.load_state_dict({"router.weight": source_weight})

        assert router.weight.dtype == torch.float32
        assert torch.equal(router.weight, source_weight)

    def test_router_weight_survives_mixed_precision_wrapper(self):
        router = TopKRouter.__new__(TopKRouter)
        torch.nn.Module.__init__(router)
        router.weight = torch.nn.Parameter(torch.empty(8, 64, dtype=torch.bfloat16))
        model = torch.nn.Module()
        model.router = router
        source_weight = torch.randn(8, 64, dtype=torch.float32)

        _promote_router_weights_to_float32([model])
        model.load_state_dict({"router.weight": source_weight})
        wrapped = _apply_mixed_precision_wrapper([model], object(), lambda _config, module: module.bfloat16())

        assert wrapped == [model]
        assert router.weight.dtype == torch.float32
        assert torch.equal(router.weight, source_weight)

    def test_provider_registers_router_dtype_hook_first(self, mock_pretrained):
        provider = MiniMaxM3Bridge().provider_bridge(mock_pretrained)

        assert isinstance(provider, MiniMaxM3ModelProvider)
        assert provider._pre_wrap_hooks[0] is _promote_router_weights_to_float32
        assert provider.transformer_layer_spec.func is minimax_m3_block_spec

    def test_provider_upgrades_legacy_checkpoint_layer_spec_idempotently(self, mock_pretrained):
        provider = MiniMaxM3Bridge().provider_bridge(mock_pretrained)
        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=True)

        provider.__post_init__()

        assert provider.transformer_layer_spec.func is minimax_m3_block_spec
        assert provider.transformer_layer_spec.keywords == {"use_transformer_engine": True}
        assert provider._pre_wrap_hooks.count(_promote_router_weights_to_float32) == 1

    def test_router_gating_casts_input_to_float32_weight_dtype(self, monkeypatch):
        captured_inputs = []

        def fake_gating(_router, hidden_states):
            captured_inputs.append(hidden_states)
            return hidden_states

        monkeypatch.setattr(TopKRouter, "gating", fake_gating)
        router = object.__new__(MiniMaxM3TopKRouter)
        torch.nn.Module.__init__(router)
        router.weight = torch.nn.Parameter(torch.randn(8, 64, dtype=torch.float32))
        hidden_states = torch.randn(2, 3, 64, dtype=torch.bfloat16)

        output = MiniMaxM3TopKRouter.gating(router, hidden_states)

        assert captured_inputs[0].dtype == torch.float32
        assert torch.equal(captured_inputs[0], hidden_states.float())
        assert output is captured_inputs[0]
        assert hidden_states.dtype == torch.bfloat16

    def test_router_gating_does_not_copy_matching_dtype_input(self, monkeypatch):
        captured_inputs = []

        def fake_gating(_router, hidden_states):
            captured_inputs.append(hidden_states)
            return hidden_states

        monkeypatch.setattr(TopKRouter, "gating", fake_gating)
        router = object.__new__(MiniMaxM3TopKRouter)
        torch.nn.Module.__init__(router)
        router.weight = torch.nn.Parameter(torch.randn(8, 64, dtype=torch.float32))
        hidden_states = torch.randn(2, 3, 64, dtype=torch.float32)

        MiniMaxM3TopKRouter.gating(router, hidden_states)

        assert captured_inputs[0] is hidden_states

    def test_block_spec_installs_minimax_router_only_for_moe_layers(self, monkeypatch):
        from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules

        moe_submodules = MoESubmodules(experts=object())
        moe_layer_spec = SimpleNamespace(submodules=SimpleNamespace(mlp=partial(MoELayer, submodules=moe_submodules)))
        dense_mlp_spec = object()
        dense_layer_spec = SimpleNamespace(submodules=SimpleNamespace(mlp=dense_mlp_spec))
        block_spec = SimpleNamespace(layer_specs=[dense_layer_spec, moe_layer_spec])
        calls = []

        def fake_get_gpt_decoder_block_spec(config, use_transformer_engine=True, **kwargs):
            calls.append((config, use_transformer_engine, kwargs))
            return block_spec

        monkeypatch.setattr(
            "megatron.bridge.models.minimax_m3.minimax_m3_bridge.get_gpt_decoder_block_spec",
            fake_get_gpt_decoder_block_spec,
        )

        output = minimax_m3_block_spec(
            "config",
            use_transformer_engine=True,
            vp_stage=2,
            pp_rank=1,
            extra="value",
        )

        assert output is block_spec
        assert calls == [
            (
                "config",
                True,
                {
                    "normalization": None,
                    "qk_l2_norm": False,
                    "vp_stage": 2,
                    "pp_rank": 1,
                    "extra": "value",
                },
            )
        ]
        assert dense_layer_spec.submodules.mlp is dense_mlp_spec
        assert moe_layer_spec.submodules.mlp.func is MoELayer
        assert moe_layer_spec.submodules.mlp.keywords["submodules"].router is MiniMaxM3TopKRouter
        assert moe_submodules.router is TopKRouter

    def test_minimax_router_is_registered_as_replicated(self):
        assert "MiniMaxM3TopKRouter" in AutoMapping._MODULE_TYPE_REGISTRY["replicated"]

    def test_block_spec_serialized_target_can_be_instantiated(self):
        layer_spec = instantiate(
            {
                "_target_": ("megatron.bridge.models.minimax_m3.minimax_m3_bridge.minimax_m3_block_spec"),
                "_partial_": True,
                "use_transformer_engine": True,
            }
        )

        assert layer_spec.func is minimax_m3_block_spec
        assert layer_spec.keywords == {"use_transformer_engine": True}

    def test_mapping_registry_uses_language_model_prefix(self):
        """All HF-side params must live under the multimodal language_model. prefix."""
        bridge = MiniMaxM3Bridge()
        registry = bridge.mapping_registry()

        for mapping in registry:
            hf_param = mapping.hf_param
            hf_params = hf_param.values() if isinstance(hf_param, dict) else [hf_param]
            for p in hf_params:
                assert str(p).startswith("language_model."), f"HF param missing language_model. prefix: {p}"

    def test_mapping_registry_does_not_map_indexer_or_vision(self):
        """Lightning-indexer and vision-tower weights are intentionally unmapped."""
        bridge = MiniMaxM3Bridge()
        registry = bridge.mapping_registry()

        for mapping in registry:
            hf_param = mapping.hf_param
            hf_params = hf_param.values() if isinstance(hf_param, dict) else [hf_param]
            for p in hf_params:
                assert "index_" not in str(p)
                assert "vision_tower" not in str(p)

    def test_megatron_to_hf_config_builds_standalone_text_config(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        config = MiniMaxM3Bridge.megatron_to_hf_config(provider)

        assert config["architectures"] == ["MiniMaxM3VLForCausalLM"]
        assert config["model_type"] == "minimax_m3_vl_text"
        assert config["torch_dtype"] == "bfloat16"
        assert "text_config" not in config
        assert "vision_config" not in config
        assert config["hidden_size"] == 64
        assert config["intermediate_size"] == 32
        assert config["dense_intermediate_size"] == 128
        assert config["shared_intermediate_size"] == 32
        assert config["head_dim"] == 16
        assert config["rotary_dim"] == 8
        assert config["hidden_act"] == "silu"
        assert config["max_position_embeddings"] == 4096
        assert config["mlp_layer_types"] == ["dense", "sparse", "sparse", "sparse"]
        assert config["layer_types"] == ["full_attention"] * 4
        assert config["rope_parameters"] == {
            "rope_theta": 5000000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        }
        assert config["bos_token_id"] == 1000
        assert config["eos_token_id"] == 1001
        assert config["pad_token_id"] is None
        assert "num_nextn_predict_layers" not in config
        assert "mtp_num_hidden_layers" not in config

    def test_exported_config_instantiates_native_text_model_without_indexer(self, mock_pretrained):
        if MiniMaxM3VLForCausalLM is None or MiniMaxM3VLTextConfig is None:
            pytest.skip("Native MiniMax-M3 text model requires Transformers 5.12.1 or newer.")

        provider = MiniMaxM3Bridge().provider_bridge(mock_pretrained)
        exported = MiniMaxM3TextBridge.megatron_to_hf_config(provider)
        config_kwargs = dict(exported)
        config_kwargs.pop("model_type")
        config_kwargs.pop("torch_dtype")

        text_config = MiniMaxM3VLTextConfig(**config_kwargs)
        model = MiniMaxM3VLForCausalLM(text_config)
        state_keys = set(model.state_dict())

        assert text_config.model_type == "minimax_m3_vl_text"
        assert text_config.layer_types == ["full_attention"] * 4
        assert all(layer.self_attn.indexer is None for layer in model.model.layers)
        assert "model.layers.0.mlp.gate_up_proj.weight" in state_keys
        assert "model.layers.1.mlp.experts.gate_up_proj" in state_keys
        assert "model.layers.1.mlp.experts.down_proj" in state_keys
        assert all("index_" not in name for name in state_keys)
        assert MiniMaxM3TextBridge().get_expected_hf_export_keys(text_config) == state_keys

    def test_megatron_to_hf_config_preserves_text_semantics(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        original = bridge.provider_bridge(mock_pretrained)
        exported = MiniMaxM3Bridge.megatron_to_hf_config(original)
        roundtrip_pretrained = Mock(spec=PreTrainedCausalLM)
        roundtrip_pretrained.config = SimpleNamespace(**exported)
        restored = MiniMaxM3TextBridge().provider_bridge(roundtrip_pretrained)

        for field in (
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "num_query_groups",
            "kv_channels",
            "vocab_size",
            "ffn_hidden_size",
            "moe_ffn_hidden_size",
            "num_moe_experts",
            "moe_router_topk",
            "moe_router_score_function",
            "moe_router_enable_expert_bias",
            "moe_router_topk_scaling_factor",
            "moe_shared_expert_intermediate_size",
            "moe_layer_freq",
            "rotary_percent",
            "layernorm_zero_centered_gamma",
            "qk_layernorm",
            "share_embeddings_and_output_weights",
            "hf_max_position_embeddings",
            "hf_bos_token_id",
            "hf_eos_token_id",
            "hf_pad_token_id",
        ):
            assert getattr(restored, field) == getattr(original, field)
        assert restored.activation_func is quick_gelu
        assert restored.activation_func_clamp_value == original.activation_func_clamp_value

    def test_text_mapping_registry_uses_native_transformers_layout(self):
        registry = MiniMaxM3TextBridge().mapping_registry()
        mappings = list(registry)
        hf_params = [
            value
            for mapping in mappings
            for value in (mapping.hf_param.values() if isinstance(mapping.hf_param, dict) else [mapping.hf_param])
        ]

        assert "model.embed_tokens.weight" in hf_params
        assert "lm_head.weight" in hf_params
        assert "model.layers.*.mlp.gate_up_proj.weight" in hf_params
        assert "model.layers.*.mlp.shared_experts.gate_up_proj.weight" in hf_params
        assert "model.layers.*.mlp.experts.gate_up_proj" in hf_params
        assert "model.layers.*.mlp.experts.down_proj" in hf_params
        assert all(not str(name).startswith("language_model.") for name in hf_params)
        assert all("block_sparse_moe" not in str(name) for name in hf_params)
        assert any(isinstance(mapping, FusedGatedExpertMapping) for mapping in mappings)
        assert any(isinstance(mapping, FusedExpertMapping) for mapping in mappings)

    def test_fused_gate_up_mapping_roundtrip_tp1(self):
        mapping = _FusedGateUpMapping(
            megatron_param="decoder.layers.2.mlp.linear_fc1.weight",
            hf_param="model.layers.2.mlp.gate_up_proj.weight",
        )
        hf_weight = torch.arange(60, dtype=torch.float32).reshape(10, 6)

        megatron_weight = mapping.hf_to_megatron(hf_weight, torch.nn.Module())
        restored = mapping.megatron_to_hf(megatron_weight, torch.nn.Module())

        assert torch.equal(megatron_weight, hf_weight)
        assert torch.equal(restored["model.layers.2.mlp.gate_up_proj.weight"], hf_weight)

    def test_fused_gate_up_mapping_rejects_odd_fused_dimension(self):
        mapping = _FusedGateUpMapping(
            megatron_param="decoder.layers.2.mlp.linear_fc1.weight",
            hf_param="model.layers.2.mlp.gate_up_proj.weight",
        )

        with pytest.raises(ValueError, match="even gate/up dimension"):
            mapping.hf_to_megatron(torch.zeros(9, 6), torch.nn.Module())

    def test_fused_gate_up_mapping_resolves_layer_wildcard(self):
        mapping = _FusedGateUpMapping(
            megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
            hf_param="model.layers.*.mlp.gate_up_proj.weight",
        )

        resolved = mapping.resolve(("17",))

        assert resolved.megatron_param == "decoder.layers.17.mlp.linear_fc1.weight"
        assert resolved.hf_param == "model.layers.17.mlp.gate_up_proj.weight"

    def test_hf_export_uses_standalone_text_bridge(self):
        assert MiniMaxM3Bridge.SUPPORTS_HF_PRETRAINED_EXPORT is False
        assert MiniMaxM3TextBridge.SUPPORTS_HF_PRETRAINED_EXPORT is True

    def test_hf_export_preserves_source_tokenizer_artifacts(self):
        assert MiniMaxM3TextBridge.ADDITIONAL_FILE_PATTERNS == [
            "added_tokens.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ]
        assert MiniMaxM3TextBridge.HF_EXPORT_OPTIONAL_ARTIFACTS == ("generation_config",)
        assert MiniMaxM3TextBridge.HF_EXPORT_TRUST_REMOTE_CODE is False
