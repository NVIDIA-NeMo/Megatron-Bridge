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

"""Unit tests for the GLM-5.3-Flash (glm5_next) bridge."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.glm.glm5_next_bridge import (
    Glm5NextBridge,
    HyperConnectionScaleMapping,
    HyperConnectionScaleSliceMapping,
)
from megatron.bridge.models.glm.glm5_next_provider import Glm5NextModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit

# A 4-layer slice of the real architecture: KDA, KDA, KDA, DSA; dense MLP then sparse.
NUM_LAYERS = 4
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
MLP_LAYER_TYPES = ["dense", "sparse", "sparse", "sparse"]


@pytest.fixture
def glm5_next_text_config():
    """Mock text config of a Glm5NextForConditionalGeneration checkpoint."""
    return SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=0,
        hidden_act="silu",
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=8192,
        kv_lora_rank=512,
        max_position_embeddings=131072,
        model_type="glm5_next",
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        swiglu_limit=7.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=151552,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=NUM_LAYERS,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=0,
        v_head_dim=128,
        layer_types=LAYER_TYPES,
        mlp_layer_types=MLP_LAYER_TYPES,
        indexer_types=["full"] * NUM_LAYERS,
        # KDA geometry.
        linear_num_heads=8,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        # DSA lightning indexer.
        index_head_dim=128,
        index_n_heads=64,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        indexer_rope_interleave=False,
        # MoE.
        moe_intermediate_size=1408,
        n_routed_experts=128,
        n_shared_experts=1,
        num_experts_per_tok=8,
        n_group=1,
        # mHC.
        mhc=True,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


@pytest.fixture
def glm5_next_config(glm5_next_text_config):
    """Mock top-level VL config wrapping the text config."""
    return SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=glm5_next_text_config,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )


@pytest.fixture
def mock_pretrained(glm5_next_config):
    """Create a mock pretrained model for GLM-5.3-Flash."""
    model = Mock(spec=PreTrainedCausalLM)
    model.config = glm5_next_config
    model.generation_config = Mock(spec=GenerationConfig)
    model.state = Mock()
    model.state.source = Mock()
    model.state.source.get_all_keys.return_value = []
    model.state.source.has_glob.return_value = False
    return model


class TestGlm5NextBridge:
    """Test cases for Glm5NextBridge."""

    def test_registration(self):
        """Glm5NextBridge is a MegatronModelBridge."""
        assert issubclass(Glm5NextBridge, MegatronModelBridge)

    def test_provider_bridge_maps_config(self, mock_pretrained):
        """provider_bridge reads the nested text config and fills the architecture in."""
        provider = Glm5NextBridge().provider_bridge(mock_pretrained)
        text_config = mock_pretrained.config.text_config

        assert isinstance(provider, Glm5NextModelProvider)
        assert provider.hidden_size == text_config.hidden_size
        assert provider.num_layers == text_config.num_hidden_layers
        assert provider.vocab_size == text_config.vocab_size
        assert provider.layernorm_epsilon == text_config.rms_norm_eps
        assert provider.params_dtype == torch.bfloat16
        assert provider.activation_func_clamp_value == text_config.swiglu_limit

        # head_dim is the (zero) RoPE width and must not become kv_channels.
        assert provider.kv_channels != 0

        # Attention: KDA on the linear_attention layers, DSA on the rest.
        assert provider.linear_attention_freq == [1, 1, 1, 0]
        assert provider.linear_num_key_heads == text_config.linear_num_heads
        assert provider.linear_num_value_heads == text_config.linear_num_heads
        assert provider.linear_key_head_dim == text_config.linear_head_dim
        assert provider.linear_value_head_dim == text_config.linear_head_dim
        assert provider.linear_conv_kernel_dim == text_config.linear_conv_kernel_dim
        assert provider.kda_gate_lower_bound == text_config.linear_lower_bound

        assert provider.multi_latent_attention is True
        assert provider.experimental_attention_variant == "dsa"
        assert provider.dsa_indexer_topk == text_config.index_topk
        assert provider.dsa_indexer_n_heads == text_config.index_n_heads
        assert provider.dsa_indexer_kpool == text_config.index_kpool
        assert provider.dsa_indexer_kpool_always_select_tail is True

        assert provider.moe_layer_freq == [0, 1, 1, 1]
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_shared_expert_intermediate_size == (
            text_config.moe_intermediate_size * text_config.n_shared_experts
        )

        assert provider.enable_mhc_connections is True
        assert provider.mhc_num_residual_streams == text_config.hc_mult
        assert provider.mhc_sinkhorn_iterations == text_config.hc_sinkhorn_iters
        assert provider.mhc_norm_eps == text_config.rms_norm_eps
        assert provider.mhc_norm_eps_inside_sqrt is True

    def test_provider_bridge_rejects_shared_indexers(self, mock_pretrained):
        """Cross-layer DSA index sharing is refused rather than silently ignored."""
        mock_pretrained.config.text_config.indexer_types = ["full", "full", "full", "shared"]
        with pytest.raises(NotImplementedError, match="index sharing"):
            Glm5NextBridge().provider_bridge(mock_pretrained)

    def test_mapping_registry_covers_kda_mhc_and_moe(self, glm5_next_config):
        """The registry names the KDA, mHC and MoE parameters under the VL weight prefix."""
        bridge = Glm5NextBridge()
        bridge.hf_config = glm5_next_config

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
        assert "model.language_model.embed_tokens.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        # KDA.
        for name in ("q_proj.weight", "q_conv1d.weight", "A_log", "dt_bias", "o_norm.weight"):
            assert f"decoder.layers.*.self_attention.{name}" in megatron_params

        # DSA indexer.
        assert "decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight" in megatron_params

        # mHC, both sites.
        for site in ("self_attention_hyper_connection", "mlp_hyper_connection"):
            assert f"decoder.layers.*.{site}.mapping_proj.weight" in megatron_params
            assert f"decoder.layers.*.{site}.alpha_pre" in megatron_params
            assert f"decoder.layers.*.{site}.alpha_post" in megatron_params
            assert f"decoder.layers.*.{site}.alpha_res" in megatron_params
        assert "model.language_model.layers.*.hc_attn_scale" in hf_params
        assert "model.language_model.layers.*.hc_ffn_scale" in hf_params

        # MoE.
        assert "decoder.layers.*.mlp.router.expert_bias" in megatron_params
        assert "model.language_model.layers.*.mlp.gate.e_score_correction_bias" in hf_params


class TestHyperConnectionScaleMappings:
    """The packed [3] hc_*_scale tensor <-> three scalar alpha_* parameters."""

    @staticmethod
    def _module(scale):
        return SimpleNamespace(
            alpha_pre=torch.tensor([scale[0]]),
            alpha_post=torch.tensor([scale[1]]),
            alpha_res=torch.tensor([scale[2]]),
        )

    def test_import_splits_the_packed_tensor(self):
        """Each alpha takes its own element of hc_*_scale."""
        scale = torch.tensor([0.1, 0.2, 0.3])
        module = self._module(scale)

        primary = HyperConnectionScaleMapping(
            megatron_pre="m.alpha_pre",
            megatron_post="m.alpha_post",
            megatron_res="m.alpha_res",
            hf_param="hf.hc_attn_scale",
        )
        post = HyperConnectionScaleSliceMapping("m.alpha_post", "hf.hc_attn_scale", 1)
        res = HyperConnectionScaleSliceMapping("m.alpha_res", "hf.hc_attn_scale", 2)

        torch.testing.assert_close(primary.hf_to_megatron(scale, module), scale[0:1])
        torch.testing.assert_close(post.hf_to_megatron(scale, module), scale[1:2])
        torch.testing.assert_close(res.hf_to_megatron(scale, module), scale[2:3])

    def test_export_is_the_inverse(self):
        """The primary mapping repacks all three alphas; the slices emit nothing."""
        scale = torch.tensor([0.1, 0.2, 0.3])
        module = self._module(scale)

        primary = HyperConnectionScaleMapping(
            megatron_pre="m.alpha_pre",
            megatron_post="m.alpha_post",
            megatron_res="m.alpha_res",
            hf_param="hf.hc_attn_scale",
        )
        exported = primary.megatron_to_hf(module.alpha_pre, module)
        torch.testing.assert_close(exported["hf.hc_attn_scale"], scale)

        slice_mapping = HyperConnectionScaleSliceMapping("m.alpha_post", "hf.hc_attn_scale", 1)
        assert slice_mapping.megatron_to_hf(module.alpha_post, module) == {}

    def test_resolve_expands_wildcards_in_every_name(self):
        """resolve() fills the layer index into all four parameter names."""
        primary = HyperConnectionScaleMapping(
            megatron_pre="decoder.layers.*.self_attention_hyper_connection.alpha_pre",
            megatron_post="decoder.layers.*.self_attention_hyper_connection.alpha_post",
            megatron_res="decoder.layers.*.self_attention_hyper_connection.alpha_res",
            hf_param="model.language_model.layers.*.hc_attn_scale",
        )
        resolved = primary.resolve(("7",))

        assert resolved.megatron_param.startswith("decoder.layers.7.")
        assert resolved._megatron_post == ("decoder.layers.7.self_attention_hyper_connection.alpha_post")
        assert resolved._megatron_res == ("decoder.layers.7.self_attention_hyper_connection.alpha_res")
        assert resolved.hf_param == "model.language_model.layers.7.hc_attn_scale"
