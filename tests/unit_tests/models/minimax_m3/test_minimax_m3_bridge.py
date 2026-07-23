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

import json
from functools import partial
from itertools import chain
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from safetensors.torch import load_file, save_file
from transformers import GenerationConfig, PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict
from megatron.bridge.models.minimax_m3.minimax_m3_bridge import (
    MiniMaxM3Bridge,
    MiniMaxM3ModelProvider,
    MiniMaxM3TopKRouter,
    MiniMaxM3VLModelProvider,
    TopKRouter,
    _promote_router_weights_to_float32,
    minimax_m3_block_spec,
    quick_gelu,
)
from megatron.bridge.models.model_provider import _apply_mixed_precision_wrapper
from megatron.bridge.utils.instantiate_utils import instantiate


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
}

_MINIMAX_M3_VL_CONFIG = {
    "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
    "model_type": "minimax_m3_vl",
    "image_token_index": 200025,
    "video_token_index": 200026,
    "projector_hidden_size": 64,
    "multimodal_projector_bias": True,
    "img_token_compression_config": {
        "image_token_compression_method": "patch_merge",
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "torch_dtype": "bfloat16",
}

_MINIMAX_M3_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "patch_size": 2,
    "image_size": 4,
    "rope_theta": 10000.0,
    "attention_dropout": 0.0,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "num_channels": 3,
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

    vision_cfg = Mock(spec=list(_MINIMAX_M3_VISION_CONFIG.keys()))
    for key, value in _MINIMAX_M3_VISION_CONFIG.items():
        setattr(vision_cfg, key, value)

    outer_keys = list(_MINIMAX_M3_VL_CONFIG.keys()) + ["text_config", "vision_config", "to_dict"]
    outer_cfg = Mock(spec=outer_keys)
    for k, v in _MINIMAX_M3_VL_CONFIG.items():
        setattr(outer_cfg, k, v)
    outer_cfg.text_config = text_cfg
    outer_cfg.vision_config = vision_cfg
    outer_cfg.to_dict.return_value = {
        **_MINIMAX_M3_VL_CONFIG,
        "text_config": dict(_MINIMAX_M3_TEXT_CONFIG),
        "vision_config": dict(_MINIMAX_M3_VISION_CONFIG),
    }

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
        assert isinstance(provider, MiniMaxM3VLModelProvider)

    def test_provider_bridge_maps_multimodal_config(self, mock_pretrained):
        provider = MiniMaxM3Bridge().provider_bridge(mock_pretrained)

        assert provider.vision_config is mock_pretrained.config.vision_config
        assert provider.image_token_id == 200025
        assert provider.video_token_id == 200026
        assert provider.projector_hidden_size == 64
        assert provider.multimodal_projector_bias is True
        assert provider.spatial_merge_size == 2
        assert provider.temporal_patch_size == 2
        assert provider.hf_config_dict["model_type"] == "minimax_m3_vl"

    def test_provider_bridge_uses_top_level_embedding_tie_contract(self):
        mock_pretrained = _make_pretrained({"tie_word_embeddings": False})
        mock_pretrained.config.tie_word_embeddings = True

        provider = MiniMaxM3Bridge().provider_bridge(mock_pretrained)

        assert provider.share_embeddings_and_output_weights is True

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
        """Text params are nested under language_model on both sides."""
        bridge = MiniMaxM3Bridge()
        registry = bridge.mapping_registry()

        for mapping in registry:
            hf_param = mapping.hf_param
            hf_params = hf_param.values() if isinstance(hf_param, dict) else [hf_param]
            if not all(str(param).startswith("language_model.") for param in hf_params):
                continue
            assert str(mapping.megatron_param).startswith("language_model.")
            for p in hf_params:
                assert str(p).startswith("language_model."), f"HF param missing language_model. prefix: {p}"

    def test_mapping_registry_maps_complete_vision_namespace(self):
        bridge = MiniMaxM3Bridge()
        registry = bridge.mapping_registry()

        replicated = {
            mapping.megatron_param: mapping.hf_param for mapping in registry if isinstance(mapping, ReplicatedMapping)
        }
        assert replicated["vision_tower.**"] == "vision_tower.**"
        assert replicated["multi_modal_projector.**"] == "multi_modal_projector.**"
        assert replicated["patch_merge_mlp.**"] == "patch_merge_mlp.**"

    def test_mapping_registry_covers_every_released_vision_and_projector_key(self):
        registry = MiniMaxM3Bridge().mapping_registry()
        vision_keys = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.pre_layrnorm.weight",
            "vision_tower.vision_model.pre_layrnorm.bias",
        ]
        layer_suffixes = [
            "layer_norm1.weight",
            "layer_norm1.bias",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "layer_norm2.weight",
            "layer_norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        vision_keys.extend(
            f"vision_tower.vision_model.encoder.layers.{layer}.{suffix}"
            for layer in range(32)
            for suffix in layer_suffixes
        )
        projector_keys = [
            f"{module}.linear_{layer}.{parameter}"
            for module in ("multi_modal_projector", "patch_merge_mlp")
            for layer in (1, 2)
            for parameter in ("weight", "bias")
        ]

        assert len(vision_keys) == 515
        assert len(projector_keys) == 8
        for key in [*vision_keys, *projector_keys]:
            mapping = registry.hf_to_megatron_lookup(key)
            assert mapping is not None, key
            assert mapping.megatron_param == key

    def test_mapping_registry_leaves_only_indexer_for_passthrough(self):
        registry = MiniMaxM3Bridge().mapping_registry()

        for mapping in registry:
            hf_param = mapping.hf_param
            hf_params = hf_param.values() if isinstance(hf_param, dict) else [hf_param]
            for p in hf_params:
                assert "index_" not in str(p)

    def test_megatron_to_hf_config_preserves_nested_vlm_contract(self, mock_pretrained):
        bridge = MiniMaxM3Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        config = MiniMaxM3Bridge.megatron_to_hf_config(provider)

        assert config["architectures"] == ["MiniMaxM3SparseForConditionalGeneration"]
        assert config["model_type"] == "minimax_m3_vl"
        assert config["image_token_index"] == 200025
        assert config["video_token_index"] == 200026
        assert config["projector_hidden_size"] == 64
        assert config["vision_config"]["hidden_size"] == 32
        assert config["vision_config"]["spatial_merge_size"] == 2
        assert config["vision_config"]["temporal_patch_size"] == 2
        assert config["text_config"]["hidden_size"] == 64
        assert config["text_config"]["intermediate_size"] == 32
        assert config["text_config"]["dense_intermediate_size"] == 128
        assert config["text_config"]["moe_layer_freq"] == [0, 1, 1, 1]

    def test_config_only_hf_save_rejects_source_dependent_export(self, tmp_path):
        hf_config = PretrainedConfig()
        hf_config.architectures = ["MiniMaxM3SparseForConditionalGeneration"]
        hf_config.model_type = "minimax_m3_vl"
        auto_bridge = AutoBridge(hf_config)
        output_path = tmp_path / "incomplete-hf-export"

        with pytest.raises(NotImplementedError, match="original Hugging Face checkpoint"):
            auto_bridge.save_hf_pretrained([], output_path)

        assert not output_path.exists()

    def test_hf_export_passthrough_is_limited_to_lightning_indexer(self):
        state = {
            "language_model.model.embed_tokens.weight": torch.randn(2, 2),
            "language_model.model.layers.3.self_attn.index_q_proj.weight": torch.randn(2, 2),
            "vision_tower.vision_model.embeddings.patch_embedding.weight": torch.randn(2, 2),
            "multi_modal_projector.linear_1.weight": torch.randn(2, 2),
            "patch_merge_mlp.linear_1.weight": torch.randn(2, 2),
        }
        hf_pretrained = SimpleNamespace(state=state)

        passthrough = dict(MiniMaxM3Bridge().stream_hf_export_passthrough(hf_pretrained))

        assert set(passthrough) == {"language_model.model.layers.3.self_attn.index_q_proj.weight"}
        assert torch.equal(
            passthrough["language_model.model.layers.3.self_attn.index_q_proj.weight"],
            state["language_model.model.layers.3.self_attn.index_q_proj.weight"],
        )

    def test_hf_export_passthrough_requires_source_checkpoint(self):
        with pytest.raises(NotImplementedError, match="original Hugging Face checkpoint"):
            list(MiniMaxM3Bridge().stream_hf_export_passthrough(SimpleNamespace()))

    def test_source_backed_export_completes_original_shards_strictly(self, tmp_path):
        source_path = tmp_path / "source"
        source_path.mkdir()
        shard_tensors = {
            "model-00001-of-00002.safetensors": {
                "language_model.model.embed_tokens.weight": torch.zeros(2, 2),
                "language_model.model.layers.3.self_attn.index_q_proj.weight": torch.full((2, 2), 3.0),
                "vision_tower.vision_model.embeddings.patch_embedding.weight": torch.zeros(2, 2),
            },
            "model-00002-of-00002.safetensors": {
                "language_model.lm_head.weight": torch.zeros(2, 2),
                "multi_modal_projector.linear_1.weight": torch.zeros(2, 2),
                "patch_merge_mlp.linear_1.weight": torch.zeros(2, 2),
            },
        }
        weight_map = {}
        for filename, tensors in shard_tensors.items():
            save_file(tensors, source_path / filename)
            weight_map.update(dict.fromkeys(tensors, filename))
        (source_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )

        source = SafeTensorsStateSource(source_path)
        hf_pretrained = SimpleNamespace(state=StateDict(source))
        converted = [
            (name, torch.full_like(tensor, 7.0))
            for filename, tensors in shard_tensors.items()
            for name, tensor in tensors.items()
            if "index_q_proj" not in name
        ]
        passthrough = MiniMaxM3Bridge().stream_hf_export_passthrough(hf_pretrained)
        output_path = tmp_path / "output"

        source.save_generator(chain(passthrough, converted), output_path, strict=True)

        exported = {}
        for filename in shard_tensors:
            exported.update(load_file(output_path / filename))
        assert set(exported) == set(weight_map)
        assert torch.equal(
            exported["language_model.model.layers.3.self_attn.index_q_proj.weight"],
            shard_tensors["model-00001-of-00002.safetensors"][
                "language_model.model.layers.3.self_attn.index_q_proj.weight"
            ],
        )
        for name, tensor in exported.items():
            if "index_q_proj" not in name:
                assert torch.equal(tensor, torch.full_like(tensor, 7.0))
