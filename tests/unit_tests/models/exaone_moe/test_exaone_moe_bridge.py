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

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_bridge import ExaoneMoeBridge
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_provider import (
    ExaoneMoeModelProvider,
    _MTPDenseLayerSpecsList,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit


def _make_config(**overrides) -> SimpleNamespace:
    values = {
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "intermediate_size": 768,
        "moe_intermediate_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "n_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "vocab_size": 153600,
        "max_position_embeddings": 4096,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "sliding_window": 1024,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_pretrained(config: SimpleNamespace) -> Mock:
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    return pretrained


class TestExaoneMoeBridge:
    @pytest.mark.parametrize("architecture", ["ExaoneMoeForCausalLM", "ExaoneMoEForCausalLM"])
    def test_autobridge_registration(self, architecture):
        config = _make_config(
            architectures=[architecture],
            model_type="exaone_moe",
        )

        assert AutoBridge.supports(config)
        AutoBridge._validate_config(config, "LGAI-EXAONE/K-EXAONE-236B-A23B")
        assert isinstance(get_model_bridge(architecture, hf_config=config), ExaoneMoeBridge)

    def test_top_level_model_registration(self):
        from megatron.bridge.models import ExaoneMoeBridge as RegisteredExaoneMoeBridge

        assert RegisteredExaoneMoeBridge is ExaoneMoeBridge

    def test_provider_bridge_maps_hf_config(self):
        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config()))

        assert isinstance(provider, ExaoneMoeModelProvider)
        assert provider.num_layers == 4
        assert provider.hidden_size == 256
        assert provider.ffn_hidden_size == 768
        assert provider.moe_ffn_hidden_size == 128
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 2
        assert provider.kv_channels == 32
        assert provider.num_moe_experts == 8
        assert provider.moe_router_topk == 2
        assert provider.moe_router_num_groups == 4
        assert provider.moe_router_group_topk == 2
        assert provider.moe_router_topk_scaling_factor == 2.5
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_shared_expert_intermediate_size == 128
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.share_embeddings_and_output_weights is False
        assert provider.moe_layer_freq == [0, 1, 1, 1]
        assert provider.no_rope_freq == [0, 1, 0, 1]
        assert provider.window_attn_skip_freq == [1, 0, 1, 0]
        assert provider.window_size == (1023, 0)

    @pytest.mark.parametrize(
        ("config_overrides", "expected"),
        [
            ({"is_moe_layer": [False, True, False, True]}, [0, 1, 0, 1]),
            ({"mlp_layer_types": ["dense", "sparse", "sparse", "dense"]}, [0, 1, 1, 0]),
            ({"first_k_dense_replace": 2}, [0, 0, 1, 1]),
        ],
    )
    def test_provider_bridge_moe_layer_frequency(self, config_overrides, expected):
        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config(**config_overrides)))

        assert provider.moe_layer_freq == expected

    @pytest.mark.parametrize(
        ("config_overrides", "expected_no_rope_freq", "expected_window_attn_skip_freq"),
        [
            ({"layer_types": ["full_attention"] * 4, "sliding_window": None}, [1, 1, 1, 1], [0, 0, 0, 0]),
            ({"layer_types": None, "sliding_window": None}, [0, 0, 0, 0], [1, 1, 1, 1]),
        ],
    )
    def test_provider_bridge_handles_missing_sliding_window(
        self, config_overrides, expected_no_rope_freq, expected_window_attn_skip_freq
    ):
        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config(**config_overrides)))

        assert provider.no_rope_freq == expected_no_rope_freq
        assert provider.window_attn_skip_freq == expected_window_attn_skip_freq
        assert provider.window_size is None

    def test_provider_bridge_infers_head_dim_and_score_function(self):
        config = _make_config()
        del config.head_dim
        del config.scoring_func

        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(config))

        assert provider.kv_channels == 32
        assert provider.moe_router_score_function == "sigmoid"

    def test_provider_bridge_maps_rope_scaling_and_mtp(self):
        config = _make_config(
            rope_parameters={"rope_type": "linear", "rope_theta": 500000.0, "factor": 4.0},
            num_nextn_predict_layers=1,
            mtp_loss_scaling_factor=0.25,
            mtp_share_layers=True,
        )

        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(config))

        assert provider.rotary_base == 500000.0
        assert provider.rope_scaling is True
        assert provider.rope_scaling_factor == 4.0
        assert provider.mtp_num_layers == 1
        assert provider.mtp_loss_scaling_factor == 0.25
        assert provider.mtp_use_repeated_layer is True

    def test_provider_bridge_uses_dense_mtp_layer_spec(self):
        provider = ExaoneMoeBridge().provider_bridge(_make_pretrained(_make_config(num_nextn_predict_layers=1)))

        block_spec = provider.transformer_layer_spec(provider, pp_rank=0)

        assert isinstance(block_spec.layer_specs, _MTPDenseLayerSpecsList)
        assert len(block_spec.layer_specs) == provider.num_layers
        assert block_spec.layer_specs[-1] is not block_spec.layer_specs[provider.num_layers - 1]

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_output_mapping_follows_weight_tying(self, tie_word_embeddings):
        bridge = ExaoneMoeBridge()
        bridge.hf_config = _make_config(tie_word_embeddings=tie_word_embeddings)

        mapping = bridge.mapping_registry().megatron_to_hf_lookup("output_layer.weight")

        expected = "model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight"
        assert mapping.hf_param == expected

    def test_mapping_registry_contains_core_and_moe_mappings(self):
        bridge = ExaoneMoeBridge()
        bridge.hf_config = _make_config()
        registry = bridge.mapping_registry()

        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_proj.weight").hf_param
            == "model.layers.0.self_attn.o_proj.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.self_attn.q_proj.weight").megatron_param
            == "decoder.layers.0.self_attention.linear_qkv.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.experts.3.gate_proj.weight").megatron_param
            == "decoder.layers.0.mlp.experts.linear_fc1.weight3"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.e_score_correction_bias").megatron_param
            == "decoder.layers.0.mlp.router.expert_bias"
        )
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc1.layer_norm_weight").hf_param
            == "model.layers.0.post_attention_layernorm.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.shared_experts.up_proj.weight").megatron_param
            == "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"
        )
        assert (
            registry.hf_to_megatron_lookup("model.layers.0.mlp.down_proj.weight").megatron_param
            == "decoder.layers.0.mlp.linear_fc2.weight"
        )

    def test_mapping_registry_adds_mtp_mappings_only_when_enabled(self):
        bridge = ExaoneMoeBridge()
        bridge.hf_config = _make_config(num_nextn_predict_layers=0)

        assert bridge.mapping_registry().hf_to_megatron_lookup("mtp.layers.0.self_attn.q_proj.weight") is None

        bridge.hf_config = _make_config(num_nextn_predict_layers=1)
        registry = bridge.mapping_registry()

        mapping = registry.hf_to_megatron_lookup("mtp.layers.0.self_attn.q_proj.weight")
        assert mapping.megatron_param == "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"

        dense_fc1_mapping = registry.hf_to_megatron_lookup("mtp.layers.0.mlp.gate_proj.weight")
        assert dense_fc1_mapping.megatron_param == "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight"

        dense_fc1_norm_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight"
        )
        assert dense_fc1_norm_mapping.hf_param == "mtp.layers.0.post_attention_layernorm.weight"

        dense_fc2_mapping = registry.hf_to_megatron_lookup("mtp.layers.0.mlp.down_proj.weight")
        assert dense_fc2_mapping.megatron_param == "mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight"

        assert registry.hf_to_megatron_lookup("mtp.layers.0.mlp.experts.3.gate_proj.weight") is None
        assert registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.mlp.router.weight") is None
