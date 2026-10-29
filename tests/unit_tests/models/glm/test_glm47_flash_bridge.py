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

"""Unit tests for the GLM-4.7-Flash bridge."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.glm.glm47_flash_bridge import GLM47FlashBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


pytestmark = pytest.mark.unit


@pytest.fixture
def glm47_flash_config():
    """Mock config for GLM-4.7-Flash."""
    return SimpleNamespace(
        architectures=["Glm4MoeLiteForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=3,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=10944,
        kv_lora_rank=512,
        max_position_embeddings=131072,
        model_type="glm4_moe_lite",
        moe_intermediate_size=1408,
        n_routed_experts=128,
        n_shared_experts=2,
        num_attention_heads=96,
        num_experts_per_tok=8,
        num_hidden_layers=47,
        num_key_value_heads=8,
        num_nextn_predict_layers=1,
        partial_rotary_factor=0.5,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        routed_scaling_factor=2.5,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        v_head_dim=128,
        vocab_size=151552,
    )


@pytest.fixture
def mock_pretrained(glm47_flash_config):
    """Create a mock pretrained model for GLM-4.7-Flash."""
    model = Mock(spec=PreTrainedCausalLM)
    model.config = glm47_flash_config
    model.generation_config = Mock(spec=GenerationConfig)
    model.state = Mock()
    model.state.source = Mock()
    model.state.source.get_all_keys.return_value = []
    model.state.source.has_glob.return_value = False
    return model


class TestGLM47FlashBridge:
    """Test cases for GLM47FlashBridge."""

    def test_registration(self):
        """Test that GLM47FlashBridge is registered as a MegatronModelBridge."""
        assert issubclass(GLM47FlashBridge, MegatronModelBridge)

    def test_provider_bridge_maps_flash_config(self, mock_pretrained):
        """Test provider bridge maps GLM-4.7-Flash config and bridge-specific defaults."""
        bridge = GLM47FlashBridge()
        provider = bridge.provider_bridge(mock_pretrained)
        config = mock_pretrained.config

        assert isinstance(provider, MLAModelProvider)
        assert provider.hidden_size == config.hidden_size
        assert provider.num_layers == config.num_hidden_layers
        assert provider.num_attention_heads == config.num_attention_heads
        assert provider.num_query_groups == config.num_key_value_heads
        assert provider.ffn_hidden_size == config.intermediate_size
        assert provider.vocab_size == config.vocab_size
        assert provider.seq_length == config.max_position_embeddings
        assert provider.layernorm_epsilon == config.rms_norm_eps
        assert provider.rotary_base == config.rope_theta
        assert provider.rotary_percent == config.partial_rotary_factor
        assert provider.q_lora_rank == config.q_lora_rank
        assert provider.kv_lora_rank == config.kv_lora_rank
        assert provider.qk_head_dim == config.qk_nope_head_dim
        assert provider.qk_pos_emb_head_dim == config.qk_rope_head_dim
        assert provider.v_head_dim == config.v_head_dim
        assert provider.params_dtype == torch.bfloat16

        assert provider.multi_latent_attention is True
        assert provider.qk_layernorm is True
        assert provider.num_moe_experts == config.n_routed_experts
        assert provider.moe_router_topk == config.num_experts_per_tok
        assert provider.moe_ffn_hidden_size == config.moe_intermediate_size
        assert provider.moe_router_topk_scaling_factor == config.routed_scaling_factor
        assert provider.moe_shared_expert_overlap is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_router_load_balancing_type == "seq_aux_loss"
        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_pre_softmax is True
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_permute_fusion is True
        assert provider.moe_router_dtype == "fp32"
        assert provider.moe_router_bias_update_rate == 0
        assert provider.moe_aux_loss_coeff == 0.001
        assert provider.mtp_num_layers == config.num_nextn_predict_layers
        assert provider.mtp_loss_scaling_factor == 0.3
        assert provider.moe_shared_expert_intermediate_size == (config.moe_intermediate_size * config.n_shared_experts)
        assert provider.moe_layer_freq == [0, 0, 0] + [1] * 44

    def test_mapping_registry_uses_hf_config_for_mtp_mappings(self, glm47_flash_config):
        """Test mapping_registry reads self.hf_config and includes MTP mappings."""
        bridge = GLM47FlashBridge()
        bridge.hf_config = glm47_flash_config

        registry = bridge.mapping_registry()
        megatron_params = {mapping.megatron_param for mapping in registry.mappings}
        hf_params = set()
        for mapping in registry.mappings:
            if not hasattr(mapping, "hf_param"):
                continue
            if isinstance(mapping.hf_param, dict):
                hf_params.update(mapping.hf_param.values())
            else:
                hf_params.add(mapping.hf_param)

        assert "embedding.word_embeddings.weight" in megatron_params
        assert "output_layer.weight" in megatron_params
        assert "decoder.layers.*.mlp.router.expert_bias" in megatron_params
        assert "model.layers.*.mlp.gate.e_score_correction_bias" in hf_params

        assert "mtp.layers.0.enorm.weight" in megatron_params
        assert "mtp.layers.0.hnorm.weight" in megatron_params
        assert "mtp.layers.0.eh_proj.weight" in megatron_params
        assert "mtp.layers.0.final_layernorm.weight" in megatron_params
        assert "mtp.layers.0.mtp_model_layer.mlp.router.expert_bias" in megatron_params

    @pytest.mark.parametrize(
        ("source_name", "alias_name"),
        [
            (
                "model.embed_tokens.weight",
                "model.layers.47.embed_tokens.weight",
            ),
            (
                "lm_head.weight",
                "model.layers.47.shared_head.head.weight",
            ),
        ],
    )
    def test_export_restores_mtp_alias_weights(
        self,
        glm47_flash_config,
        source_name,
        alias_name,
    ):
        """Test export restores the duplicated MTP embedding and output weights."""
        bridge = GLM47FlashBridge()
        bridge.hf_config = glm47_flash_config
        source_weight = torch.randn(4, 4)
        converted_weights = {source_name: source_weight}

        result = bridge.maybe_modify_converted_hf_weight(
            SimpleNamespace(global_param_name="checkpoint.model." + source_name),
            converted_weights,
            {},
        )

        torch.testing.assert_close(result[alias_name], source_weight)
        assert result[alias_name].data_ptr() != source_weight.data_ptr()

    def test_export_does_not_add_mtp_alias_when_mtp_is_disabled(self, glm47_flash_config):
        """Test export leaves checkpoints without MTP layers unchanged."""
        bridge = GLM47FlashBridge()
        bridge.hf_config = SimpleNamespace(**{**vars(glm47_flash_config), "num_nextn_predict_layers": 0})
        source_weight = torch.randn(4, 4)
        converted_weights = {"model.embed_tokens.weight": source_weight}

        result = bridge.maybe_modify_converted_hf_weight(
            SimpleNamespace(global_param_name="embedding.word_embeddings.weight"),
            converted_weights,
            {},
        )

        assert set(result) == {"model.embed_tokens.weight"}
        assert result["model.embed_tokens.weight"] is source_weight
